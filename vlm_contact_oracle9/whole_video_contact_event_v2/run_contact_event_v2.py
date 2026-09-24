
#!/usr/bin/env python3

from pathlib import Path
import math

import numpy as np
import pandas as pd
from PIL import Image

import torch
from transformers import (
    AutoProcessor,
    Qwen3VLForConditionalGeneration,
)


ROOT = Path(
    "/shared/ssd_30T/zhuoyingyang/physact/sam3_robowm"
)

OLD = (
    ROOT
    / "vlm_contact_oracle9"
    / "whole_video_causal_v1"
)

INPUT_CSV = (
    OLD
    / "WHOLE_VIDEO_CONTACT_CANDIDATES.csv"
)

OUT = (
    ROOT
    / "vlm_contact_oracle9"
    / "whole_video_contact_event_v2"
)

OUT.mkdir(
    parents=True,
    exist_ok=True,
)

CAND_OUT = (
    OUT
    / "CONTACT_EVENT_CANDIDATES_V2.csv"
)

VIDEO_OUT = (
    OUT
    / "CONTACT_EVENT_VIDEO_SCORES_V2.csv"
)

MODEL_PATH = (
    "/shared/ssd_30T/zhuoyingyang/models/"
    "Qwen3-VL-8B-Instruct"
)


# ============================================================
# ONSET-INDEPENDENT CONTACT-EVENT PROMPT
#
# IMPORTANT:
# - t is NOT claimed to be response onset.
# - target motion must NOT be used as contact evidence.
# - 2D overlap alone is insufficient.
# ============================================================

PROMPT = r"""
You are examining three chronological HIGH-RESOLUTION ROI images
from a robot manipulation video.

Target object: {target}

Image 1 = t-2
Image 2 = t-1
Image 3 = candidate time t

IMPORTANT:
candidate time t is NOT known to be the target-response onset.
Do not assume that the target is responding at t.

Judge ONLY the visible spatial relationship between the robot
end-effector and the target object.

Use Images 1 and 2 to understand approach direction, local geometry,
gap closing, and depth ordering before Image 3.

Do NOT infer contact from target motion.
Do NOT infer contact from later consequences.
Mere 2D overlap is NOT sufficient evidence of physical contact.

Choose exactly ONE answer:

A

CLEARLY SEPARATED / NO CONTACT EVENT SUPPORTED BY t:

At Image 3, a visible gap remains between the closest robot and
target-object surfaces, or the images do not show the robot reaching
the target surface.

B

DEPTH AMBIGUOUS:

At Image 3, the robot and target appear extremely close or overlap in
2D projection, but the images do not reliably establish whether the
surfaces truly meet or whether a small 3D gap remains.

C

SURFACE-CONTACT EVENT SUPPORTED BY t:

At Image 3, a specific robot end-effector surface visibly meets the
target-object surface at a plausible contact location, with no visible
separating gap and sufficient visual evidence to support actual surface
contact. Earlier images may show the gap closing or the end-effector
approaching that same contact location.

Return exactly one character:

A, B, or C.
""".strip()


def softmax3(values):

    m = max(values)

    e = [
        math.exp(
            x - m
        )
        for x
        in values
    ]

    z = sum(e)

    return [
        x / z
        for x
        in e
    ]


@torch.inference_mode()
def candidate_logprob(
    processor,
    model,
    images,
    prompt,
    candidate,
):

    messages = [{
        "role": "user",
        "content": [
            {
                "type":
                    "image"
            },
            {
                "type":
                    "image"
            },
            {
                "type":
                    "image"
            },
            {
                "type":
                    "text",

                "text":
                    prompt,
            },
        ],
    }]

    prefix_text = (
        processor
        .apply_chat_template(

            messages,

            tokenize=False,

            add_generation_prompt=True,
        )
    )

    prefix = processor(

        text=[
            prefix_text
        ],

        images=
            images,

        padding=
            True,

        return_tensors=
            "pt",
    )

    full = processor(

        text=[
            prefix_text
            +
            candidate
        ],

        images=
            images,

        padding=
            True,

        return_tensors=
            "pt",
    )

    prefix_len = (
        prefix[
            "input_ids"
        ].shape[1]
    )

    full = {

        k:
            (
                v.to(
                    model.device
                )
                if hasattr(
                    v,
                    "to"
                )
                else v
            )

        for k, v
        in full.items()
    }

    ids = full[
        "input_ids"
    ]

    out = model(
        **full,
        use_cache=False,
    )

    logp = torch.log_softmax(
        out.logits,
        dim=-1,
    )

    total = 0.0

    for pos in range(
        prefix_len,
        ids.shape[1],
    ):

        token = int(
            ids[
                0,
                pos
            ]
        )

        total += float(
            logp[
                0,
                pos - 1,
                token,
            ].item()
        )

    return total


def pairwise_auc(
    labels,
    scores,
):

    pos = [
        float(s)
        for y, s
        in zip(
            labels,
            scores,
        )
        if int(y) == 1
    ]

    neg = [
        float(s)
        for y, s
        in zip(
            labels,
            scores,
        )
        if int(y) == 0
    ]

    if (
        not pos
        or
        not neg
    ):
        return float(
            "nan"
        )

    wins = 0.0
    total = 0

    for p in pos:

        for n in neg:

            total += 1

            if p > n:

                wins += 1.0

            elif p == n:

                wins += 0.5

    return (
        wins / total
    )


# ============================================================
# LOAD EXISTING WHOLE-VIDEO CANDIDATES
# ============================================================

src = pd.read_csv(
    INPUT_CSV
)

print("=" * 110)
print("WHOLE-VIDEO CONTACT-EVENT V2")
print("=" * 110)

print(
    "Candidate rows:",
    len(src),
)

print(
    "Cases:",
    src["case"].nunique(),
)

print()
print(
    "No human onset is used."
)

print(
    "Candidate frames are the existing geometry-only proposals."
)

print(
    "Prompt is onset-independent."
)

print(
    "Missing response evidence will ABSTAIN, not receive score 0."
)


# ============================================================
# MODEL
# ============================================================

print()
print(
    "Loading Qwen3-VL-8B-Instruct..."
)

processor = (
    AutoProcessor
    .from_pretrained(

        MODEL_PATH,

        trust_remote_code=True,
    )
)

model = (
    Qwen3VLForConditionalGeneration
    .from_pretrained(

        MODEL_PATH,

        torch_dtype=
            torch.bfloat16,

        device_map=
            "auto",

        trust_remote_code=
            True,
    )
    .eval()
)


# ============================================================
# RESUME
# ============================================================

done = {}

rows = []


if CAND_OUT.exists():

    old = pd.read_csv(
        CAND_OUT
    )

    for _, r in old.iterrows():

        key = (
            str(
                r["split"]
            ),
            str(
                r["case"]
            ),
            int(
                r["frame"]
            ),
        )

        done[
            key
        ] = r.to_dict()

        rows.append(
            r.to_dict()
        )

    print(
        "Resume rows:",
        len(rows),
    )


# ============================================================
# SCORE CONTACT EVENT
# ============================================================

for idx, r in src.iterrows():

    split = str(
        r[
            "split"
        ]
    )

    case = str(
        r[
            "case"
        ]
    )

    frame = int(
        r[
            "frame"
        ]
    )

    target = str(
        r[
            "target"
        ]
    )

    key = (
        split,
        case,
        frame,
    )

    if key in done:
        continue


    roi_dir = Path(
        r[
            "roi_dir"
        ]
    )

    paths = [

        roi_dir
        / "t_minus_2_roi.png",

        roi_dir
        / "t_minus_1_roi.png",

        roi_dir
        / "candidate_roi.png",
    ]


    for p in paths:

        if not p.exists():

            raise RuntimeError(
                f"Missing ROI: {p}"
            )


    images = [

        Image.open(
            p
        ).convert(
            "RGB"
        )

        for p
        in paths
    ]


    prompt = PROMPT.replace(
        "{target}",
        target,
    )


    ll_A = candidate_logprob(
        processor,
        model,
        images,
        prompt,
        "A",
    )

    ll_B = candidate_logprob(
        processor,
        model,
        images,
        prompt,
        "B",
    )

    ll_C = candidate_logprob(
        processor,
        model,
        images,
        prompt,
        "C",
    )


    pA, pB, pC = softmax3(
        [
            ll_A,
            ll_B,
            ll_C,
        ]
    )


    event_score = 100.0 * (
        pC
        +
        0.5 * pB
    )


    out = r.to_dict()

    out[
        "event_p_separated"
    ] = pA

    out[
        "event_p_ambiguous"
    ] = pB

    out[
        "event_p_contact"
    ] = pC

    out[
        "contact_event_score"
    ] = event_score


    rows.append(
        out
    )

    done[
        key
    ] = out


    pd.DataFrame(
        rows
    ).to_csv(
        CAND_OUT,
        index=False,
    )


    print(
        f"[{len(done):04d}/"
        f"{len(src):04d}] "
        f"{split:8s} "
        f"{case} "
        f"t={frame:3d} "
        f"A={pA:.3f} "
        f"B={pB:.3f} "
        f"C={pC:.3f} "
        f"EVENT={event_score:6.2f}"
    )


cand = pd.DataFrame(
    rows
)


# ============================================================
# VIDEO LEVEL
#
# IMPORTANT CHANGE:
#
# If a video has no HYBRID response evidence at all:
#   ABSTAIN.
#
# Do NOT assign causal score = 0.
#
# For each window:
#   among geometry/contact candidates that have FUTURE response
#   evidence, take maximum contact-event score.
#
# Earlier motion episodes are ignored rather than used as vetoes.
# ============================================================

video_rows = []


for (
    split,
    case,
), q in cand.groupby(
    [
        "split",
        "case",
    ]
):

    first = q.iloc[
        0
    ]

    vr = {

        "split":
            split,

        "case":
            case,

        "audit":
            first[
                "audit"
            ],

        "eval_gt":
            first[
                "eval_gt"
            ],

        "target":
            first[
                "target"
            ],

        "motion_any_video":
            int(
                first[
                    "motion_any_video"
                ]
            ),

        "max_event_score":
            float(
                q[
                    "contact_event_score"
                ].max()
            ),
    }


    best_event = q.loc[
        q[
            "contact_event_score"
        ].idxmax()
    ]

    vr[
        "max_event_frame"
    ] = int(
        best_event[
            "frame"
        ]
    )


    for w in [
        12,
        24,
        36,
    ]:

        post_col = (
            f"post_{w}"
        )

        usable = q[
            pd.to_numeric(
                q[
                    post_col
                ],
                errors="coerce",
            )
            ==
            1
        ]


        if (
            int(
                vr[
                    "motion_any_video"
                ]
            )
            ==
            0
            or
            len(
                usable
            )
            ==
            0
        ):

            vr[
                f"future_pair_score_{w}"
            ] = np.nan

            vr[
                f"future_pair_frame_{w}"
            ] = np.nan

            vr[
                f"future_pair_status_{w}"
            ] = "ABSTAIN"

        else:

            best = usable.loc[
                usable[
                    "contact_event_score"
                ].idxmax()
            ]

            vr[
                f"future_pair_score_{w}"
            ] = float(
                best[
                    "contact_event_score"
                ]
            )

            vr[
                f"future_pair_frame_{w}"
            ] = int(
                best[
                    "frame"
                ]
            )

            vr[
                f"future_pair_status_{w}"
            ] = "SCORED"


    video_rows.append(
        vr
    )


video = pd.DataFrame(
    video_rows
)

video.to_csv(
    VIDEO_OUT,
    index=False,
)


# ============================================================
# EVALUATION
#
# Fresh20 is exploratory/fixed evaluation already inspected.
# We report all windows; do NOT choose one by Fresh20 AUROC.
# ============================================================

def evaluate(
    name,
    q,
):

    q = q[
        pd.to_numeric(
            q[
                "eval_gt"
            ],
            errors="coerce",
        ).notna()
    ].copy()

    q[
        "eval_gt"
    ] = pd.to_numeric(
        q[
            "eval_gt"
        ]
    ).astype(
        int
    )


    print()
    print("=" * 110)
    print(name)
    print("=" * 110)

    print(
        "Binary videos:",
        len(q),
    )

    print(
        "CONTACT:",
        int(
            (
                q[
                    "eval_gt"
                ]
                ==
                1
            ).sum()
        ),
    )

    print(
        "NO_CONTACT:",
        int(
            (
                q[
                    "eval_gt"
                ]
                ==
                0
            ).sum()
        ),
    )


    # --------------------------------------------------------
    # Contact-event only control
    # --------------------------------------------------------

    auc = pairwise_auc(

        q[
            "eval_gt"
        ].tolist(),

        q[
            "max_event_score"
        ].tolist(),
    )

    print(
        f"{'max_event_score':28s} "
        f"coverage={len(q)}/{len(q)} "
        f"AUROC={auc:.4f}"
    )


    # --------------------------------------------------------
    # Contact-event -> future response pair
    # --------------------------------------------------------

    for w in [
        12,
        24,
        36,
    ]:

        col = (
            f"future_pair_score_{w}"
        )

        valid = q[
            q[
                col
            ].notna()
        ]

        if (
            valid[
                "eval_gt"
            ].nunique()
            <
            2
        ):

            auc = float(
                "nan"
            )

        else:

            auc = pairwise_auc(

                valid[
                    "eval_gt"
                ].tolist(),

                valid[
                    col
                ].tolist(),
            )

        print(
            f"{col:28s} "
            f"coverage="
            f"{len(valid)}/{len(q)} "
            f"AUROC={auc:.4f}"
        )


    print()
    print(
        q[
            [
                "case",
                "audit",
                "motion_any_video",
                "max_event_score",
                "future_pair_score_12",
                "future_pair_score_24",
                "future_pair_score_36",
            ]
        ].to_string(
            index=False
        )
    )


evaluate(

    "PRIMARY9 — CONTACT EVENT V2",

    video[
        video[
            "split"
        ]
        ==
        "PRIMARY9"
    ],
)


evaluate(

    "FRESH20 CLEAR BINARY — CONTACT EVENT V2",

    video[
        video[
            "split"
        ]
        ==
        "FRESH20"
    ],
)


evaluate(

    "COMBINED — CONTACT EVENT V2",

    video,
)


print()
print("=" * 110)
print("REFERENCE")
print("=" * 110)

print(
    "Old auto-onset Contact V6:"
)

print(
    "  Primary9 AUROC = 0.450"
)

print(
    "  Fresh20 AUROC = 0.600"
)

print()

print(
    "Oracle-onset frozen Contact V6:"
)

print(
    "  Primary9 AUROC = 0.850"
)

print(
    "  Fresh20 AUROC = 0.923"
)

print()

print(
    "V2 uses NO human onset."
)

print(
    "Missing response evidence -> ABSTAIN."
)

print(
    "Earlier motion does NOT veto a later contact->response pair."
)

print()

print(
    "Saved candidates:",
    CAND_OUT,
)

print(
    "Saved video scores:",
    VIDEO_OUT,
)

