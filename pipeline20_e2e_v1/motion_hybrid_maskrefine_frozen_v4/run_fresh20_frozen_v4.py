#!/usr/bin/env python3

from pathlib import Path
import pickle
import re
import gc

import numpy as np
import pandas as pd
from pycocotools import mask as mask_utils


# ============================================================
# PATHS
# ============================================================

ROOT = Path(
    "/shared/ssd_30T/zhuoyingyang/physact"
)

WORK = ROOT / "sam3_robowm"

PIPE = (
    WORK
    / "pipeline20_e2e_v1"
)

HYBRID_ROOT = (
    PIPE
    / "motion_hybrid"
)

PAI_ROOT = (
    ROOT
    / "pai_results"
    / "components_all"
)

OUT = (
    PIPE
    / "motion_hybrid_maskrefine_frozen_v4"
)

OUT.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# FROZEN PRIMARY9 PARAMETERS
#
# DO NOT CHANGE ON FRESH20.
# ============================================================

RAW_STATE_THRESHOLD = 0.03
LOOKBACK = 8
EPISODE_MAX_GAP = 2


# ============================================================
# FRESH20 HUMAN ONSET
#
# None = onset ABSTAIN, excluded from scorable evaluation.
#
# Intervals preserve annotation uncertainty.
# ============================================================

GT = {

    "Cosmos3_seed101_0004": (118, 118),
    "Cosmos3_seed101_0014": (72, 73),
    "Cosmos3_seed101_0015": (127, 129),
    "Cosmos3_seed101_0029": (128, 130),
    "Cosmos3_seed101_0034": (129, 130),

    "Cosmos3_seed101_0040": None,

    "Cosmos3_seed101_0042": (88, 88),

    "Cosmos3_seed102_0015": (165, 170),
    "Cosmos3_seed102_0019": (96, 97),
    "Cosmos3_seed102_0025": (105, 107),

    "Cosmos3_seed102_0027": None,

    "Cosmos3_seed102_0029": (105, 105),
    "Cosmos3_seed102_0036": (82, 87),
    "Cosmos3_seed102_0042": (106, 107),

    "Cosmos3_seed103_0003": (75, 76),
    "Cosmos3_seed103_0004": (55, 56),

    "Cosmos3_seed103_0026": None,

    "Cosmos3_seed103_0028": (149, 151),
    "Cosmos3_seed103_0035": (188, 190),
    "Cosmos3_seed103_0036": (90, 93),
}


# ============================================================
# KNOWN FROZEN HYBRID PREDICTIONS
#
# Used ONLY to verify that automatic file discovery found the
# same baseline traces used in the previous Fresh20 audit.
# ============================================================

EXPECTED_HYBRID = {

    "Cosmos3_seed101_0004": 41,
    "Cosmos3_seed101_0014": 45,
    "Cosmos3_seed101_0015": 129,
    "Cosmos3_seed101_0029": 130,
    "Cosmos3_seed101_0034": 130,

    "Cosmos3_seed101_0042": None,

    "Cosmos3_seed102_0015": 172,
    "Cosmos3_seed102_0019": 97,
    "Cosmos3_seed102_0025": 93,
    "Cosmos3_seed102_0029": 107,
    "Cosmos3_seed102_0036": 94,

    "Cosmos3_seed102_0042": None,

    "Cosmos3_seed103_0003": 75,
    "Cosmos3_seed103_0004": 55,
    "Cosmos3_seed103_0028": 144,
    "Cosmos3_seed103_0035": 127,

    "Cosmos3_seed103_0036": None,
}


# ============================================================
# UTILS
# ============================================================

def parse_case(case):

    m = re.match(
        r"Cosmos3_seed(\d+)_(\d+)",
        case,
    )

    if not m:
        raise RuntimeError(
            f"Cannot parse {case}"
        )

    return (
        m.group(1),
        m.group(2),
    )


def read_candidate_column(df, kind):

    if kind == "hybrid":

        choices = [
            "hybrid_candidate",
        ]

    else:

        choices = [
            "motion_candidate",
            "local_candidate",
        ]


    for c in choices:

        if c in df.columns:

            return (
                df[c]
                .fillna(0)
                .to_numpy(int)
                >
                0
            )


    raise RuntimeError(
        f"No {kind} candidate column. "
        f"columns={list(df.columns)}"
    )


# ============================================================
# FILE DISCOVERY
# ============================================================

def find_hybrid_trace(case):

    obvious = [
        HYBRID_ROOT
        / f"{case}_hybrid_trace.csv",

        HYBRID_ROOT
        / case
        / "hybrid_trace.csv",

        HYBRID_ROOT
        / case
        / f"{case}_hybrid_trace.csv",
    ]


    for p in obvious:

        if p.exists():

            return p


    candidates = []

    for p in HYBRID_ROOT.rglob(
        "*.csv"
    ):

        if case not in str(p):
            continue

        try:

            h = pd.read_csv(
                p,
                nrows=2,
            )

        except Exception:
            continue


        if "hybrid_candidate" in h.columns:

            candidates.append(
                p
            )


    if len(candidates) != 1:

        raise RuntimeError(
            f"{case}: expected exactly one "
            f"HYBRID trace, found:\n"
            +
            "\n".join(
                str(x)
                for x in candidates
            )
        )


    return candidates[0]


def find_local_trace(case):

    candidates = []


    for p in PIPE.rglob(
        "*.csv"
    ):

        ps = str(
            p
        ).lower()


        if case.lower() not in ps:
            continue


        if (
            "maskrefine_frozen_v4"
            in ps
        ):
            continue


        if "local" not in ps:
            continue


        try:

            h = pd.read_csv(
                p,
                nrows=2,
            )

        except Exception:
            continue


        if not (
            "motion_candidate" in h.columns
            or
            "local_candidate" in h.columns
        ):
            continue


        score = 0


        if (
            "motion_trace_localref.csv"
            in ps
        ):
            score += 20

        if "localref" in ps:
            score += 10

        if "local_reference" in ps:
            score += 8

        if "audit" in ps:
            score += 2

        if "refine" in ps:
            score -= 5


        candidates.append(
            (
                score,
                p,
            )
        )


    if not candidates:

        raise RuntimeError(
            f"{case}: no LOCAL trace found "
            f"under {PIPE}"
        )


    candidates.sort(
        key=lambda x: (
            -x[0],
            len(
                str(
                    x[1]
                )
            ),
        )
    )


    return candidates[0][1]


def find_object_tracks(case):

    candidates = []


    for p in PIPE.rglob(
        "cotracker_motion_tracks.npz"
    ):

        if case in str(p):

            candidates.append(
                p
            )


    if not candidates:

        # Narrow fallback outside PIPE.
        for p in WORK.rglob(
            "cotracker_motion_tracks.npz"
        ):

            ps = str(p)

            if (
                case in ps
                and
                "primary9"
                not in ps.lower()
            ):

                candidates.append(
                    p
                )


    if not candidates:

        raise RuntimeError(
            f"{case}: cannot find "
            "cotracker_motion_tracks.npz"
        )


    candidates.sort(
        key=lambda p: (
            0
            if "pipeline20_e2e_v1"
            in str(p)
            else 1,

            len(
                str(p)
            ),
        )
    )


    return candidates[0]


# ============================================================
# PAI MASK DECODING
# ============================================================

def get_objects(obj):

    if isinstance(
        obj,
        list,
    ):

        return [
            x
            for x in obj
            if isinstance(
                x,
                dict,
            )
            and
            "segmentation_mask_rle"
            in x
        ]


    if isinstance(
        obj,
        dict,
    ):

        if (
            "segmentation_mask_rle"
            in obj
        ):
            return [obj]


        for key in [
            "objects",
            "components",
            "results",
            "predictions",
        ]:

            if (
                key in obj
                and
                isinstance(
                    obj[key],
                    list,
                )
            ):

                return get_objects(
                    obj[key]
                )


    return []


def decode_masks(item):

    x = item[
        "segmentation_mask_rle"
    ]


    if not (
        isinstance(
            x,
            dict,
        )
        and
        "data" in x
        and
        "mask_shape" in x
    ):

        raise RuntimeError(
            "Unexpected PAI mask format"
        )


    shape = tuple(
        int(v)
        for v in x[
            "mask_shape"
        ]
    )


    flat = mask_utils.decode(
        x["data"]
    )


    flat = np.asarray(
        flat,
        dtype=np.uint8,
    ).reshape(-1)


    expected = int(
        np.prod(
            shape
        )
    )


    if flat.size != expected:

        raise RuntimeError(
            f"Mask size mismatch: "
            f"{flat.size} vs {expected}"
        )


    return (
        flat.reshape(
            shape,
            order="C",
        )
        >
        0
    )


# ============================================================
# TARGET OBJECT IDENTIFICATION
#
# Do NOT rely on hand-written Fresh20 phrases.
#
# Use the existing object CoTracker seed points:
# the correct PAI object mask should contain those points.
# ============================================================

def load_target_masks(
    case,
):

    track_path = (
        find_object_tracks(
            case
        )
    )


    z = np.load(
        track_path,
        allow_pickle=True,
    )


    seed_points = np.asarray(
        z[
            "seed_points"
        ],
        dtype=float,
    )


    if "seed_frame" in z:

        seed_frame = int(
            np.asarray(
                z[
                    "seed_frame"
                ]
            ).reshape(-1)[0]
        )

    else:

        seed_frame = 0


    seed, idx = parse_case(
        case
    )


    pkl_path = (
        PAI_ROOT
        / f"cosmos3_seed{seed}"
        / "sam"
        / f"{idx}.pkl"
    )


    if not pkl_path.exists():

        raise RuntimeError(
            f"Missing PAI file: "
            f"{pkl_path}"
        )


    with open(
        pkl_path,
        "rb",
    ) as f:

        obj = pickle.load(
            f
        )


    items = get_objects(
        obj
    )


    if not items:

        raise RuntimeError(
            f"{case}: no PAI objects"
        )


    best = None


    for j, item in enumerate(
        items
    ):

        masks = decode_masks(
            item
        )


        sf = int(
            np.clip(
                seed_frame,
                0,
                masks.shape[0] - 1,
            )
        )


        m = masks[
            sf
        ]


        H, W = m.shape


        xs = np.rint(
            seed_points[:, 0]
        ).astype(int)

        ys = np.rint(
            seed_points[:, 1]
        ).astype(int)


        valid = (
            (xs >= 0)
            &
            (xs < W)
            &
            (ys >= 0)
            &
            (ys < H)
        )


        if valid.sum() == 0:

            overlap = 0.0

        else:

            overlap = float(
                np.mean(
                    m[
                        ys[valid],
                        xs[valid],
                    ]
                )
            )


        phrase = str(
            item.get(
                "phrase",
                "",
            )
        )


        print(
            f"    PAI candidate {j:02d}: "
            f"phrase={phrase!r} "
            f"seed_overlap={overlap:.3f}"
        )


        if (
            best is None
            or
            overlap > best[0]
        ):

            best = (
                overlap,
                phrase,
                masks,
            )

        else:

            del masks
            gc.collect()


    if best is None:

        raise RuntimeError(
            f"{case}: target selection failed"
        )


    overlap, phrase, masks = best


    print(
        "    SELECTED TARGET:",
        repr(
            phrase
        ),
        "overlap=",
        f"{overlap:.3f}",
    )


    # Seed points were deliberately sampled inside the target.
    # Low overlap means target association/perception is dubious.
    if overlap < 0.50:

        raise RuntimeError(
            f"{case}: target-mask association "
            f"unreliable, overlap={overlap:.3f}"
        )


    return (
        masks,
        phrase,
        overlap,
        track_path,
        pkl_path,
    )


# ============================================================
# MASK STATE
# ============================================================

def iou(
    a,
    b,
):

    inter = int(
        np.count_nonzero(
            a & b
        )
    )

    union = int(
        np.count_nonzero(
            a | b
        )
    )


    if union == 0:

        return np.nan


    return float(
        inter / union
    )


def persistent_raw_state(
    masks,
    t,
):

    T = masks.shape[0]


    if (
        t < 1
        or
        t + 12 >= T
    ):

        return np.nan


    ref = masks[
        t - 1
    ]

    m8 = masks[
        t + 8
    ]

    m12 = masks[
        t + 12
    ]


    if (
        ref.sum() == 0
        or
        m8.sum() == 0
        or
        m12.sum() == 0
    ):

        return np.nan


    raw8 = (
        1.0
        -
        iou(
            ref,
            m8,
        )
    )

    raw12 = (
        1.0
        -
        iou(
            ref,
            m12,
        )
    )


    return float(
        min(
            raw8,
            raw12,
        )
    )


# ============================================================
# EPISODE LOGIC — FROZEN FROM PRIMARY9
# ============================================================

def make_episodes(
    frames,
    candidate,
):

    pos = [
        int(f)
        for f, c
        in zip(
            frames,
            candidate,
        )
        if c
    ]


    if not pos:

        return []


    episodes = []

    current = [
        pos[0]
    ]


    for f in pos[1:]:

        gap = (
            f
            -
            current[-1]
            -
            1
        )


        if (
            gap
            <=
            EPISODE_MAX_GAP
        ):

            current.append(
                f
            )

        else:

            episodes.append(
                current
            )

            current = [
                f
            ]


    episodes.append(
        current
    )


    return episodes


def episode_anchor(
    frames,
    candidate,
    episode,
):

    frame_to_i = {
        int(f): i
        for i, f
        in enumerate(
            frames
        )
    }


    start_i = frame_to_i[
        episode[0]
    ]

    end_i = frame_to_i[
        episode[-1]
    ]


    for i in range(
        start_i,
        min(
            end_i + 1,
            len(
                candidate
            )
            -
            2,
        )
        +
        1,
    ):

        win = candidate[
            i:i+3
        ]


        inds = np.where(
            win
        )[0]


        if len(inds) >= 2:

            return int(
                frames[
                    i
                    +
                    inds[0]
                ]
            )


    return None


def frozen_hybrid_onset(
    frames,
    candidate,
):

    for i in range(
        len(candidate)
        -
        2
    ):

        inds = np.where(
            candidate[
                i:i+3
            ]
        )[0]


        if len(inds) >= 2:

            return int(
                frames[
                    i
                    +
                    inds[0]
                ]
            )


    return None


# ============================================================
# DETECTOR
# ============================================================

def detect_frozen_v4(
    masks,
    hframes,
    hcand,
    lframes,
    lcand,
):

    episodes = make_episodes(
        hframes,
        hcand,
    )


    accepted_anchor = None

    episode_rows = []


    # HYBRID mask veto
    for eid, ep in enumerate(
        episodes,
        1,
    ):

        anchor = episode_anchor(
            hframes,
            hcand,
            ep,
        )


        if anchor is None:
            continue


        score = persistent_raw_state(
            masks,
            anchor,
        )


        accepted = (
            np.isfinite(
                score
            )
            and
            score
            >=
            RAW_STATE_THRESHOLD
        )


        episode_rows.append({
            "episode_id":
                eid,

            "episode_start":
                ep[0],

            "episode_last_positive":
                ep[-1],

            "anchor":
                anchor,

            "persistent_raw":
                score,

            "accepted":
                int(
                    accepted
                ),
        })


        if accepted:

            accepted_anchor = (
                anchor
            )

            break


    if accepted_anchor is None:

        return (
            None,
            None,
            episode_rows,
            [],
        )


    # Verified LOCAL precursor
    lo = max(
        1,
        accepted_anchor
        -
        LOOKBACK,
    )

    hi = accepted_anchor


    precursor_rows = []

    accepted_precursors = []


    for f, c in zip(
        lframes,
        lcand,
    ):

        f = int(
            f
        )


        if (
            f < lo
            or
            f > hi
            or
            not c
        ):

            continue


        score = persistent_raw_state(
            masks,
            f,
        )


        accepted = (
            np.isfinite(
                score
            )
            and
            score
            >=
            RAW_STATE_THRESHOLD
        )


        precursor_rows.append({
            "frame":
                f,

            "persistent_raw":
                score,

            "accepted":
                int(
                    accepted
                ),
        })


        if accepted:

            accepted_precursors.append(
                f
            )


    if accepted_precursors:

        final = min(
            accepted_precursors
        )

    else:

        final = (
            accepted_anchor
        )


    return (
        final,
        accepted_anchor,
        episode_rows,
        precursor_rows,
    )


# ============================================================
# EVALUATION AGAINST INTERVAL
# ============================================================

def classify(
    pred,
    interval,
):

    if interval is None:

        return (
            None,
            "ABSTAIN",
        )


    if pred is None:

        return (
            None,
            "MISSED",
        )


    lo, hi = interval


    if pred < lo:

        err = pred - lo

    elif pred > hi:

        err = pred - hi

    else:

        err = 0


    if abs(err) <= 2:

        return (
            err,
            "WITHIN_2",
        )


    if err < -2:

        return (
            err,
            "EARLY",
        )


    return (
        err,
        "LATE",
    )


# ============================================================
# RUN
# ============================================================

results = []


print("=" * 120)
print("FRESH20 — FROZEN HYBRID + MASK VETO + LOCAL REFINE V4")
print("=" * 120)

print(
    "RAW_STATE_THRESHOLD =",
    RAW_STATE_THRESHOLD,
)

print(
    "LOOKBACK =",
    LOOKBACK,
)

print(
    "EPISODE_MAX_GAP =",
    EPISODE_MAX_GAP,
)

print(
    "ALL PARAMETERS FROZEN FROM PRIMARY9."
)

print(
    "NO FRESH20 TUNING."
)


for i, (
    case,
    interval,
) in enumerate(
    GT.items(),
    1,
):

    print()
    print("#" * 120)

    print(
        f"[{i:02d}/20] {case}"
    )

    print(
        "HUMAN GT =",
        interval
        if interval is not None
        else "ABSTAIN",
    )


    hybrid_path = (
        find_hybrid_trace(
            case
        )
    )

    local_path = (
        find_local_trace(
            case
        )
    )


    print(
        "HYBRID TRACE =",
        hybrid_path,
    )

    print(
        "LOCAL TRACE  =",
        local_path,
    )


    H = pd.read_csv(
        hybrid_path
    )

    L = pd.read_csv(
        local_path
    )


    hframes = (
        H["frame"]
        .to_numpy(int)
    )

    lframes = (
        L["frame"]
        .to_numpy(int)
    )


    hcand = read_candidate_column(
        H,
        "hybrid",
    )

    lcand = read_candidate_column(
        L,
        "local",
    )


    baseline = frozen_hybrid_onset(
        hframes,
        hcand,
    )


    print(
        "FROZEN HYBRID =",
        baseline,
    )


    # For the three human-onset ABSTAIN cases, we still compute
    # predictions for provenance, but exclude them from scoring.
    try:

        (
            masks,
            phrase,
            overlap,
            track_path,
            pkl_path,
        ) = load_target_masks(
            case
        )


        (
            pred,
            anchor,
            episode_rows,
            precursor_rows,
        ) = detect_frozen_v4(
            masks,
            hframes,
            hcand,
            lframes,
            lcand,
        )


        perception_status = (
            "OK"
        )


    except Exception as e:

        print(
            "MASK PERCEPTION FAILURE:",
            repr(e),
        )


        pred = None
        anchor = None
        phrase = None
        overlap = np.nan
        track_path = None
        pkl_path = None
        episode_rows = []
        precursor_rows = []

        perception_status = (
            "MASK_ABSTAIN"
        )


    base_err, base_status = classify(
        baseline,
        interval,
    )

    v4_err, v4_status = classify(
        pred,
        interval,
    )


    print(
        "MASKED HYB ANCH =",
        anchor,
    )

    print(
        "FROZEN V4 FINAL =",
        pred,
        v4_err,
        v4_status,
    )


    print()
    print(
        "HYBRID EPISODES:"
    )


    if episode_rows:

        for x in episode_rows:

            print(
                f"  EP{x['episode_id']:02d} "
                f"{x['episode_start']:3d}-"
                f"{x['episode_last_positive']:3d} "
                f"anchor={x['anchor']:3d} "
                f"raw={x['persistent_raw']:.4f} "
                f"accept={x['accepted']}"
            )

    else:

        print(
            "  none"
        )


    print()
    print(
        "LOCAL PRECURSORS:"
    )


    if precursor_rows:

        for x in precursor_rows:

            print(
                f"  frame={x['frame']:3d} "
                f"raw={x['persistent_raw']:.4f} "
                f"accept={x['accepted']}"
            )

    else:

        print(
            "  none"
        )


    results.append({
        "case":
            case,

        "gt_lo":
            (
                interval[0]
                if interval is not None
                else np.nan
            ),

        "gt_hi":
            (
                interval[1]
                if interval is not None
                else np.nan
            ),

        "human_onset_status":
            (
                "SCORABLE"
                if interval is not None
                else "ABSTAIN"
            ),

        "perception_status":
            perception_status,

        "target_phrase":
            phrase,

        "target_seed_overlap":
            overlap,

        "hybrid_pred":
            baseline,

        "hybrid_error":
            base_err,

        "hybrid_status":
            base_status,

        "accepted_hybrid_anchor":
            anchor,

        "v4_pred":
            pred,

        "v4_error":
            v4_err,

        "v4_status":
            v4_status,

        "hybrid_trace":
            str(
                hybrid_path
            ),

        "local_trace":
            str(
                local_path
            ),

        "track_path":
            (
                str(track_path)
                if track_path is not None
                else None
            ),

        "pkl_path":
            (
                str(pkl_path)
                if pkl_path is not None
                else None
            ),
    })


    del H
    del L

    if (
        "masks"
        in locals()
    ):

        del masks

    gc.collect()


# ============================================================
# VERIFY BASELINE PROVENANCE
# ============================================================

print()
print("=" * 120)
print("VERIFYING FROZEN HYBRID BASELINE")
print("=" * 120)


bad = []


for r in results:

    case = r[
        "case"
    ]


    if case not in EXPECTED_HYBRID:

        continue


    got = r[
        "hybrid_pred"
    ]

    expected = EXPECTED_HYBRID[
        case
    ]


    same = (
        got == expected
        or
        (
            got is None
            and
            expected is None
        )
    )


    print(
        f"{case:27s} "
        f"expected={str(expected):>4s} "
        f"got={str(got):>4s} "
        f"{'OK' if same else 'MISMATCH'}"
    )


    if not same:

        bad.append(
            (
                case,
                expected,
                got,
            )
        )


if bad:

    raise RuntimeError(
        "STOP: discovered files do not reproduce "
        "the known Frozen HYBRID baseline. "
        f"Mismatches={bad}"
    )


print()
print(
    "Baseline provenance check PASSED."
)


# ============================================================
# SUMMARY
# ============================================================

scorable = [
    r
    for r in results
    if r[
        "human_onset_status"
    ]
    ==
    "SCORABLE"
]


def summarize(
    key,
):

    counts = {
        "WITHIN_2": 0,
        "EARLY": 0,
        "LATE": 0,
        "MISSED": 0,
    }


    errors = []


    for r in scorable:

        pred = r[
            key
        ]


        interval = (
            int(
                r[
                    "gt_lo"
                ]
            ),
            int(
                r[
                    "gt_hi"
                ]
            ),
        )


        err, status = classify(
            pred,
            interval,
        )


        counts[
            status
        ] += 1


        if err is not None:

            errors.append(
                abs(
                    err
                )
            )


    return (
        counts,
        errors,
    )


BH, BE = summarize(
    "hybrid_pred"
)

V4, V4E = summarize(
    "v4_pred"
)


print()
print("=" * 120)
print("FRESH20 FROZEN V4 SUMMARY")
print("=" * 120)

print(
    "TOTAL          = 20"
)

print(
    "ONSET ABSTAIN  = 3"
)

print(
    "SCORABLE       =",
    len(
        scorable
    ),
)


for name, c, e in [
    (
        "FROZEN HYBRID",
        BH,
        BE,
    ),
    (
        "FROZEN MASK V4",
        V4,
        V4E,
    ),
]:

    print()
    print(name)

    print(
        "  WITHIN ±2 =",
        c[
            "WITHIN_2"
        ],
        "/17",
    )

    print(
        "  EARLY     =",
        c[
            "EARLY"
        ],
    )

    print(
        "  LATE      =",
        c[
            "LATE"
        ],
    )

    print(
        "  MISSED    =",
        c[
            "MISSED"
        ],
    )


    if e:

        print(
            "  median abs error =",
            f"{np.median(e):.2f}",
        )

        print(
            "  mean abs error   =",
            f"{np.mean(e):.2f}",
        )

        print(
            "  max abs error    =",
            f"{np.max(e):.2f}",
        )


print()
print("=" * 120)
print("PER SCORABLE CASE")
print("=" * 120)


for r in scorable:

    gt_text = (
        f"{int(r['gt_lo'])}"
        if (
            int(
                r[
                    "gt_lo"
                ]
            )
            ==
            int(
                r[
                    "gt_hi"
                ]
            )
        )
        else
        (
            f"{int(r['gt_lo'])}-"
            f"{int(r['gt_hi'])}"
        )
    )


    print(
        f"{r['case']:27s} "
        f"GT={gt_text:7s} "
        f"HYB={str(r['hybrid_pred']):>4s} "
        f"H={r['hybrid_status']:8s} "
        f"V4={str(r['v4_pred']):>4s} "
        f"V4ERR={str(r['v4_error']):>4s} "
        f"{r['v4_status']}"
    )


print()
print("=" * 120)
print("ABSTAIN CASES — NOT SCORED")
print("=" * 120)


for r in results:

    if (
        r[
            "human_onset_status"
        ]
        ==
        "ABSTAIN"
    ):

        print(
            f"{r['case']:27s} "
            f"HYB={str(r['hybrid_pred']):>4s} "
            f"V4={str(r['v4_pred']):>4s} "
            f"perception={r['perception_status']}"
        )


out_csv = (
    OUT
    /
    "FRESH20_FROZEN_MASK_V4_SUMMARY.csv"
)


pd.DataFrame(
    results
).to_csv(
    out_csv,
    index=False,
)


print()
print("=" * 120)
print("SAVED")
print("=" * 120)

print(
    out_csv
)

print()
print(
    "NO PARAMETERS WERE TUNED ON FRESH20."
)
