from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(
    "/shared/ssd_30T/zhuoyingyang/physact/sam3_robowm"
)

CASE_CSV = (
    ROOT / "vlm_contact_oracle9/"
    "backward36_multi_episode_v1/"
    "BACKWARD36_ALL38_CASES.csv"
)

OUT_DIR = (
    ROOT / "vlm_contact_oracle9/"
    "local_causal_pair_v1"
)

WINDOW = 12

# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------

def first_col(df, names):
    for c in names:
        if c in df.columns:
            return c
    return None


def pairwise_auc(labels, scores):
    y = np.asarray(labels, dtype=int)
    s = np.asarray(scores, dtype=float)

    pos = s[y == 1]
    neg = s[y == 0]

    if len(pos) == 0 or len(neg) == 0:
        return np.nan

    wins = 0.0

    for p in pos:
        for n in neg:
            if p > n:
                wins += 1.0
            elif p == n:
                wins += 0.5

    return wins / (len(pos) * len(neg))


def load_strength(trace_path):
    df = pd.read_csv(trace_path)

    frame_col = first_col(
        df,
        ["frame", "frame_id", "t"]
    )

    needed = [
        "relative_rigid",
        "relative_q75",
        "relative_rigid_thr",
        "relative_point_thr",
    ]

    if (
        frame_col is None
        or any(c not in df.columns for c in needed)
    ):
        return None

    frame = pd.to_numeric(
        df[frame_col],
        errors="coerce",
    )

    valid = frame.notna()

    df = df.loc[valid].copy()
    df["frame"] = frame[valid].astype(int)

    for c in needed:
        df[c] = pd.to_numeric(
            df[c],
            errors="coerce",
        )

    with np.errstate(
        divide="ignore",
        invalid="ignore",
    ):
        rigid_ratio = (
            df["relative_rigid"]
            / df["relative_rigid_thr"]
        )

        point_ratio = (
            df["relative_q75"]
            / df["relative_point_thr"]
        )

    df["strength"] = np.maximum(
        rigid_ratio,
        point_ratio,
    )

    return {
        int(r["frame"]): float(r["strength"])
        for _, r in df.iterrows()
        if np.isfinite(r["strength"])
    }


def persistent_strength(strength, lo, hi):
    """
    Robust local motion strength.

    In every consecutive 3-frame window:
      take the 2nd-largest motion strength.

    Then take the maximum across the region.

    Therefore a single-frame spike cannot dominate;
    at least 2/3 frames must support the motion.
    """

    if hi < lo:
        return 0.0

    vals = np.array(
        [
            strength.get(f, np.nan)
            for f in range(lo, hi + 1)
        ],
        dtype=float,
    )

    if len(vals) < 2:
        return 0.0

    best = 0.0

    # Normal 3-frame windows.
    if len(vals) >= 3:
        for i in range(len(vals) - 2):

            w = vals[i:i+3]
            w = w[np.isfinite(w)]

            if len(w) >= 2:
                # second largest = 2-of-3 support
                v = np.sort(w)[-2]
                best = max(best, float(v))

    return best


# ------------------------------------------------------------
# Load case table and corrected GT
# ------------------------------------------------------------

cases = pd.read_csv(CASE_CSV)

cases.loc[
    cases["case"] == "Cosmos3_seed101_0035",
    ["label", "gt"]
] = [0, "NO_CONTACT"]

cases.loc[
    cases["case"] == "Cosmos3_seed103_0037",
    ["label", "gt"]
] = [1, "CONTACT"]


# ------------------------------------------------------------
# Find all Contact Event V2 candidate CSVs.
# ------------------------------------------------------------

candidate_files = list(
    (
        ROOT / "vlm_contact_oracle9"
    ).rglob(
        "CONTACT_EVENT_CANDIDATES_V2.csv"
    )
)

print("=" * 110)
print("CANDIDATE FILES")
print("=" * 110)

for p in candidate_files:
    print(p)

all_candidates = []

for p in candidate_files:

    d = pd.read_csv(p)

    case_col = first_col(
        d,
        [
            "case",
            "video",
            "video_id",
        ]
    )

    frame_col = first_col(
        d,
        [
            "frame",
            "candidate_frame",
        ]
    )

    event_col = first_col(
        d,
        [
            "contact_event_score",
            "event_score",
        ]
    )

    contact_col = first_col(
        d,
        [
            "contact_score",
            "raw_contact_score",
        ]
    )

    if (
        case_col is None
        or frame_col is None
        or event_col is None
    ):
        continue

    tmp = pd.DataFrame({
        "case":
            d[case_col].astype(str),

        "frame":
            pd.to_numeric(
                d[frame_col],
                errors="coerce",
            ),

        "event_score":
            pd.to_numeric(
                d[event_col],
                errors="coerce",
            ),
    })

    if contact_col is not None:
        tmp["contact_score"] = pd.to_numeric(
            d[contact_col],
            errors="coerce",
        )
    else:
        tmp["contact_score"] = np.nan

    tmp["candidate_source"] = str(p)

    tmp = tmp[
        tmp["frame"].notna()
        & tmp["event_score"].notna()
    ].copy()

    tmp["frame"] = (
        tmp["frame"]
        .astype(int)
    )

    all_candidates.append(tmp)


if not all_candidates:
    raise RuntimeError(
        "No candidate CSVs could be loaded."
    )

cand = pd.concat(
    all_candidates,
    ignore_index=True,
)

# Remove exact duplicated copies.
cand = cand.drop_duplicates(
    subset=[
        "case",
        "frame",
        "event_score",
        "contact_score",
    ]
).copy()

print()
print(
    "Loaded candidate rows:",
    len(cand)
)


# ------------------------------------------------------------
# Candidate-centered causal scoring
# ------------------------------------------------------------

candidate_rows = []
video_rows = []

for _, case_row in cases.iterrows():

    case = str(case_row["case"])

    trace_path = Path(
        str(case_row["hybrid_trace"])
    )

    c = cand[
        cand["case"] == case
    ].copy()

    if (
        not trace_path.exists()
        or len(c) == 0
    ):
        continue

    strength = load_strength(
        trace_path
    )

    if strength is None:
        continue

    scored = []

    for _, r in c.iterrows():

        t = int(r["frame"])

        # Everything strictly before contact.
        pre = persistent_strength(
            strength,
            t - WINDOW,
            t - 1,
        )

        # Everything strictly after contact.
        post = persistent_strength(
            strength,
            t + 1,
            t + WINDOW,
        )

        # ------------------------------------------------
        # Causal support
        #
        # Require meaningful post-contact response:
        # post >= 1 means it reaches the frozen normalized
        # motion threshold.
        #
        # Then ask how much of POST is genuinely new,
        # rather than motion already present before contact.
        # ------------------------------------------------

        if post < 1.0:
            causal_weight = 0.0
        else:
            causal_weight = np.clip(
                (post - pre)
                / max(post, 1e-8),
                0.0,
                1.0,
            )

        causal_event_score = (
            float(r["event_score"])
            * causal_weight
        )

        scored.append({
            "case": case,
            "frame": t,
            "event_score":
                float(r["event_score"]),
            "contact_score":
                r["contact_score"],
            "pre_strength":
                pre,
            "post_strength":
                post,
            "causal_weight":
                causal_weight,
            "causal_event_score":
                causal_event_score,
        })

    if not scored:
        continue

    sdf = pd.DataFrame(scored)

    candidate_rows.extend(
        sdf.to_dict("records")
    )

    best_idx = (
        sdf["causal_event_score"]
        .idxmax()
    )

    best = sdf.loc[
        best_idx
    ]

    video_rows.append({
        "case":
            case,

        "gt":
            case_row["gt"],

        "label":
            case_row["label"],

        "future_pair_score_36":
            case_row[
                "future_pair_score_36"
            ],

        "causal_best_frame":
            int(best["frame"]),

        "causal_pre_strength":
            float(
                best["pre_strength"]
            ),

        "causal_post_strength":
            float(
                best["post_strength"]
            ),

        "causal_weight":
            float(
                best["causal_weight"]
            ),

        "causal_event_score":
            float(
                best["causal_event_score"]
            ),
    })


candidate_df = pd.DataFrame(
    candidate_rows
)

video_df = pd.DataFrame(
    video_rows
)

candidate_out = (
    OUT_DIR /
    "LOCAL_CAUSAL_CANDIDATES.csv"
)

video_out = (
    OUT_DIR /
    "LOCAL_CAUSAL_VIDEO_SCORES.csv"
)

candidate_df.to_csv(
    candidate_out,
    index=False,
)

video_df.to_csv(
    video_out,
    index=False,
)


# ------------------------------------------------------------
# Evaluation
# ------------------------------------------------------------

binary = video_df[
    video_df["label"].notna()
].copy()

print()
print("=" * 110)
print("LOCAL CAUSAL SCORE")
print("=" * 110)

print(
    "coverage:",
    len(binary),
    "/",
    int(cases["label"].notna().sum()),
)

print(
    "CONTACT:",
    int(
        (binary["label"] == 1).sum()
    ),
)

print(
    "NO_CONTACT:",
    int(
        (binary["label"] == 0).sum()
    ),
)

print(
    "AUROC:",
    f"{pairwise_auc(binary['label'], binary['causal_event_score']):.4f}",
)


# Future on exactly same cases.
future = binary[
    binary[
        "future_pair_score_36"
    ].notna()
].copy()

print()
print(
    "FuturePair36 same support:",
    f"{pairwise_auc(future['label'], future['future_pair_score_36']):.4f}",
)


# ------------------------------------------------------------
# Key cases
# ------------------------------------------------------------

for key in [
    "Cosmos3_seed101_0004",
    "Cosmos2.5_0006",
]:

    print()
    print("=" * 110)
    print(key)
    print("=" * 110)

    x = candidate_df[
        candidate_df["case"] == key
    ].copy()

    if len(x) == 0:
        print("No candidates")
        continue

    print(
        x[
            [
                "frame",
                "event_score",
                "pre_strength",
                "post_strength",
                "causal_weight",
                "causal_event_score",
            ]
        ]
        .sort_values(
            "causal_event_score",
            ascending=False,
        )
        .head(15)
        .to_string(index=False)
    )


print()
print("Saved:")
print(candidate_out)
print(video_out)
