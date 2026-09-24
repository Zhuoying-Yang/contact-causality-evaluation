
from pathlib import Path
import numpy as np
import pandas as pd


ROOT = Path(
    "/shared/ssd_30T/zhuoyingyang/physact/sam3_robowm"
)

OUTDIR = (
    ROOT
    / "vlm_contact_oracle9"
    / "backward36_multi_episode_v1"
)

OUTDIR.mkdir(
    parents=True,
    exist_ok=True,
)

OUT_ALL = (
    OUTDIR
    / "BACKWARD36_ALL38_CASES.csv"
)

OUT_BINARY = (
    OUTDIR
    / "BACKWARD36_BINARY32_EVAL.csv"
)


# ============================================================
# Existing frozen Contact Event V2 outputs.
# NO Qwen rerun.
# ============================================================

OLD_CAND = (
    ROOT
    / "vlm_contact_oracle9"
    / "whole_video_contact_event_v2"
    / "CONTACT_EVENT_CANDIDATES_V2.csv"
)

OLD_VIDEO = (
    ROOT
    / "vlm_contact_oracle9"
    / "whole_video_contact_event_v2"
    / "CONTACT_EVENT_VIDEO_SCORES_V2.csv"
)


PILOT_CAND = (
    ROOT
    / "vlm_contact_oracle9"
    / "table_pilot8_v1"
    / "whole_video_contact_event_v2"
    / "CONTACT_EVENT_CANDIDATES_V2.csv"
)

PILOT_VIDEO = (
    ROOT
    / "vlm_contact_oracle9"
    / "table_pilot8_v1"
    / "whole_video_contact_event_v2"
    / "CONTACT_EVENT_VIDEO_SCORES_V2.csv"
)


NEW_CAND = (
    ROOT
    / "vlm_contact_oracle9"
    / "cosmos25_fourcase_v1"
    / "whole_video_contact_event_v2"
    / "CONTACT_EVENT_CANDIDATES_V2.csv"
)

NEW_VIDEO = (
    ROOT
    / "vlm_contact_oracle9"
    / "cosmos25_fourcase_v1"
    / "whole_video_contact_event_v2"
    / "CONTACT_EVENT_VIDEO_SCORES_V2.csv"
)


# ============================================================
# Exact five extra CONTACT cases.
# The two SPECIAL target-identity cases are intentionally NOT
# part of this requested five-case contact set.
# ============================================================

PILOT5 = {
    "Cosmos3_seed101_0035",
    "Cosmos3_seed103_0023",
    "Cosmos3_seed102_0018",
    "Cosmos3_seed103_0008",
    "Cosmos3_seed102_0020",
}


# ============================================================
# Current four.
#
# All four will appear in the final 38-case table.
# 0013 remains unresolved and is NOT forced into AUROC.
# ============================================================

NEW4_LABEL = {
    "Cosmos2.5_0006": 0.0,
    "Cosmos2.5_0013": np.nan,
    "Cosmos2.5_0024": 0.0,
    "Cosmos2.5_0028": 0.0,
}


# ============================================================
# Helpers
# ============================================================

def first_existing_column(
    df,
    candidates,
    required=True,
):
    for c in candidates:
        if c in df.columns:
            return c

    if required:
        raise RuntimeError(
            "Could not find any of columns: "
            + str(candidates)
            + "\nAvailable:\n"
            + str(df.columns.tolist())
        )

    return None


def to_bool_array(series):
    out = []

    for x in series:

        if pd.isna(x):
            out.append(False)
            continue

        if isinstance(
            x,
            (
                bool,
                np.bool_,
            ),
        ):
            out.append(
                bool(x)
            )
            continue

        if isinstance(
            x,
            (
                int,
                float,
                np.integer,
                np.floating,
            ),
        ):
            out.append(
                float(x) != 0.0
            )
            continue

        s = str(x).strip().lower()

        out.append(
            s in {
                "1",
                "true",
                "t",
                "yes",
                "y",
            }
        )

    return np.asarray(
        out,
        dtype=bool,
    )


def resolve_path(x):

    if x is None:
        return None

    if pd.isna(x):
        return None

    p = Path(
        str(x)
    )

    if p.exists():
        return p

    if not p.is_absolute():
        q = ROOT / p
        if q.exists():
            return q

    return None


def fallback_trace(
    case,
    source_name,
):

    roots = []

    if source_name == "OLD29":
        roots = [
            ROOT
            / "pipeline20_e2e_v1"
            / "motion_hybrid",

            ROOT
            / "vlm_contact_oracle9"
            / "motion_primary9_hybrid_veto_v1",
        ]

    elif source_name == "PILOT5":
        roots = [
            ROOT
            / "vlm_contact_oracle9"
            / "table_pilot8_v1"
            / "motion_hybrid",
        ]

    elif source_name == "COSMOS25_4":
        roots = [
            ROOT
            / "vlm_contact_oracle9"
            / "cosmos25_fourcase_v1"
            / "motion_hybrid",
        ]

    direct_names = [
        f"{case}_hybrid_trace.csv",
        f"{case}_motion_trace_hybrid.csv",
    ]

    for r in roots:

        if not r.exists():
            continue

        for name in direct_names:
            p = r / name
            if p.exists():
                return p

        matches = sorted(
            r.rglob(
                f"{case}*hybrid*trace*.csv"
            )
        )

        if matches:
            return matches[0]

    return None


# ============================================================
# Frozen HYBRID persistence:
#
# "2 candidates in a 3-frame window."
#
# We return the FIRST positive frame belonging to the first
# 3-frame window that contains >=2 positives.
#
# Examples:
#   [5,6] -> onset 5
#   [70,71] -> onset 70
#
# This reconstructs the frozen reported onset rather than
# simply taking an arbitrary later motion frame.
# ============================================================


def all_persistent_response_onsets(
    trace_path,
    preferred_column=None,
):
    trace = pd.read_csv(trace_path)

    frame_col = first_existing_column(
        trace,
        ["frame", "frame_id", "t"],
        required=False,
    )

    if (
        preferred_column is not None
        and preferred_column in trace.columns
    ):
        motion_col = preferred_column
    else:
        motion_col = first_existing_column(
            trace,
            [
                "hybrid_candidate",
                "motion_candidate",
                "candidate",
            ],
        )

    flags = to_bool_array(trace[motion_col])

    if frame_col is None:
        frames = np.arange(len(trace), dtype=int)
    else:
        frames = (
            pd.to_numeric(
                trace[frame_col],
                errors="coerce",
            )
            .fillna(-1)
            .astype(int)
            .to_numpy()
        )

    valid = frames >= 0
    frames = frames[valid]
    flags = flags[valid]

    if len(frames) == 0:
        return [], motion_col, 0

    max_frame = int(frames.max())
    arr = np.zeros(max_frame + 1, dtype=bool)

    for f, x in zip(frames, flags):
        if 0 <= f < len(arr):
            arr[f] = bool(x)

    positive_count = int(arr.sum())

    if positive_count == 0:
        return [], motion_col, 0

    positive_frames = np.flatnonzero(arr).astype(int).tolist()

    # Gap <= 2 means the points can still satisfy
    # the frozen 2-of-3 persistence rule together.
    clusters = []
    current = [positive_frames[0]]

    for f in positive_frames[1:]:
        if f - current[-1] <= 2:
            current.append(f)
        else:
            clusters.append(current)
            current = [f]

    clusters.append(current)

    # Each cluster needs >=2 positives to satisfy
    # the frozen 2-of-3 persistence requirement.
    onsets = [
        int(cluster[0])
        for cluster in clusters
        if len(cluster) >= 2
    ]

    return onsets, motion_col, positive_count


def pairwise_auc(
    labels,
    scores,
):

    y = np.asarray(
        labels,
        dtype=int,
    )

    s = np.asarray(
        scores,
        dtype=float,
    )

    pos = s[y == 1]
    neg = s[y == 0]

    if (
        len(pos) == 0
        or len(neg) == 0
    ):
        return np.nan

    wins = 0.0

    for p in pos:
        for n in neg:

            if p > n:
                wins += 1.0

            elif p == n:
                wins += 0.5

    return wins / (
        len(pos)
        * len(neg)
    )


def label_old(
    audit,
):

    if str(audit) == "CONTACT":
        return 1.0

    if str(audit) == "NO_CONTACT":
        return 0.0

    return np.nan


# ============================================================
# One dataset
# ============================================================

def process_source(
    source_name,
    cand_path,
    video_path,
    selected_cases=None,
):

    if not cand_path.exists():
        raise FileNotFoundError(
            cand_path
        )

    if not video_path.exists():
        raise FileNotFoundError(
            video_path
        )

    cand = pd.read_csv(
        cand_path
    )

    video = pd.read_csv(
        video_path
    )

    case_col_c = first_existing_column(
        cand,
        [
            "case",
            "name",
        ],
    )

    case_col_v = first_existing_column(
        video,
        [
            "case",
            "name",
        ],
    )

    frame_col = first_existing_column(
        cand,
        [
            "candidate_frame",
            "frame",
            "t",
        ],
    )

    score_col = first_existing_column(
        cand,
        [
            "contact_event_score",
            "event_score",
            "contact_score",
        ],
    )

    trace_col = first_existing_column(
        cand,
        [
            "motion_trace",
            "hybrid_trace",
            "motion_trace_path",
        ],
        required=False,
    )

    motion_col_field = first_existing_column(
        cand,
        [
            "motion_column",
            "hybrid_column",
        ],
        required=False,
    )

    print()
    print("=" * 110)
    print(source_name)
    print("=" * 110)

    print(
        "candidate rows:",
        len(cand),
    )

    print(
        "candidate score column:",
        score_col,
    )

    print(
        "candidate frame column:",
        frame_col,
    )

    print(
        "trace column:",
        trace_col,
    )

    print(
        "motion-column field:",
        motion_col_field,
    )

    if selected_cases is None:
        cases = sorted(
            video[
                case_col_v
            ]
            .astype(str)
            .unique()
            .tolist()
        )
    else:
        cases = sorted(
            selected_cases
        )

    results = []

    for case in cases:

        crows = cand[
            cand[
                case_col_c
            ].astype(str)
            == case
        ].copy()

        vrows = video[
            video[
                case_col_v
            ].astype(str)
            == case
        ].copy()

        if len(vrows) == 0:
            raise RuntimeError(
                f"{source_name}: "
                f"video-level row missing: {case}"
            )

        vrow = vrows.iloc[0]

        # -----------------------------------------------
        # Ground truth / audit
        # -----------------------------------------------

        audit = (
            vrow["audit"]
            if "audit" in vrow.index
            else ""
        )

        split = (
            vrow["split"]
            if "split" in vrow.index
            else source_name
        )

        if source_name == "OLD29":

            label = label_old(
                audit
            )

            group = str(
                split
            )

        elif source_name == "PILOT5":

            label = 1.0
            audit = "CONTACT"
            group = "PILOT5_CONTACT"

        elif source_name == "COSMOS25_4":

            label = NEW4_LABEL[
                case
            ]

            group = "COSMOS25_SELECTED4"

        else:
            raise RuntimeError(
                source_name
            )

        # -----------------------------------------------
        # Existing FuturePair36 baseline
        # -----------------------------------------------

        future36 = np.nan

        if (
            "future_pair_score_36"
            in vrow.index
        ):
            future36 = pd.to_numeric(
                pd.Series(
                    [
                        vrow[
                            "future_pair_score_36"
                        ]
                    ]
                ),
                errors="coerce",
            ).iloc[0]

        max_event = np.nan

        if (
            "max_event_score"
            in vrow.index
        ):
            max_event = pd.to_numeric(
                pd.Series(
                    [
                        vrow[
                            "max_event_score"
                        ]
                    ]
                ),
                errors="coerce",
            ).iloc[0]

        # -----------------------------------------------
        # Find HYBRID trace.
        # -----------------------------------------------

        trace_path = None
        preferred_motion_col = None

        if len(crows) > 0:

            if trace_col is not None:

                vals = (
                    crows[
                        trace_col
                    ]
                    .dropna()
                    .tolist()
                )

                for x in vals:
                    p = resolve_path(
                        x
                    )

                    if p is not None:
                        trace_path = p
                        break

            if motion_col_field is not None:

                vals = (
                    crows[
                        motion_col_field
                    ]
                    .dropna()
                    .astype(str)
                    .tolist()
                )

                if vals:
                    preferred_motion_col = (
                        vals[0]
                    )

        if trace_path is None:

            trace_path = fallback_trace(
                case,
                source_name,
            )

        if trace_path is None:
            raise RuntimeError(
                f"{source_name}: "
                f"cannot find HYBRID trace "
                f"for {case}"
            )

        (
            response_onsets,
            used_motion_col,
            hybrid_positive_frames,
        ) = all_persistent_response_onsets(
            trace_path,
            preferred_motion_col,
        )

        # Retain first onset only for old diagnostic columns.
        # Pairing below uses ALL episode onsets.
        response_onset = (
            response_onsets[0]
            if response_onsets
            else None
        )

        # -----------------------------------------------
        # BACKWARD36
        #
        # Eligible:
        #       response_onset - 36 <= candidate <= onset
        #
        # Never use a later contact to explain earlier response.
        # -----------------------------------------------

        backward_score = np.nan
        backward_frame = np.nan
        eligible_count = 0
        eligible_frames = ""

        if not response_onsets:

            status = (
                "ABSTAIN_NO_RELIABLE_RESPONSE"
            )

        elif len(crows) == 0:

            status = (
                "ABSTAIN_NO_CONTACT_CANDIDATES"
            )

        else:

            crows[
                "_frame"
            ] = pd.to_numeric(
                crows[
                    frame_col
                ],
                errors="coerce",
            )

            crows[
                "_score"
            ] = pd.to_numeric(
                crows[
                    score_col
                ],
                errors="coerce",
            )

            valid_candidates = (
                crows["_frame"].notna()
                &
                crows["_score"].notna()
            )

            eligible_mask = pd.Series(
                False,
                index=crows.index,
            )

            for r in response_onsets:
                lo = int(r) - 36
                hi = int(r)

                eligible_mask |= (
                    valid_candidates
                    &
                    (crows["_frame"] >= lo)
                    &
                    (crows["_frame"] <= hi)
                )

            eligible = crows[
                eligible_mask
            ].copy()

            eligible_count = len(
                eligible
            )

            eligible_frames = ",".join(
                str(int(x))
                for x in sorted(
                    eligible[
                        "_frame"
                    ]
                    .astype(int)
                    .tolist()
                )
            )

            if len(eligible) == 0:

                status = (
                    "ABSTAIN_NO_PRE_RESPONSE_CANDIDATE"
                )

            else:

                best_idx = (
                    eligible[
                        "_score"
                    ]
                    .idxmax()
                )

                best = eligible.loc[
                    best_idx
                ]

                backward_score = float(
                    best[
                        "_score"
                    ]
                )

                backward_frame = int(
                    best[
                        "_frame"
                    ]
                )

                status = "OK"

        if pd.isna(
            label
        ):
            gt_text = "UNRESOLVED"
        elif int(
            label
        ) == 1:
            gt_text = "CONTACT"
        else:
            gt_text = "NO_CONTACT"

        results.append({
            "source":
                source_name,

            "group":
                group,

            "split":
                split,

            "case":
                case,

            "audit":
                audit,

            "gt":
                gt_text,

            "label":
                label,

            "hybrid_trace":
                str(
                    trace_path
                ),

            "hybrid_column":
                used_motion_col,

            "hybrid_positive_frames":
                hybrid_positive_frames,

            "response_episode_count":
                len(response_onsets),

            "response_episode_onsets":
                ",".join(
                    str(int(x))
                    for x in response_onsets
                ),

            "first_response_frame":
                (
                    np.nan
                    if response_onset is None
                    else int(
                        response_onset
                    )
                ),

            "backward36_window_start":
                (
                    np.nan
                    if response_onset is None
                    else int(
                        response_onset
                    )
                    - 36
                ),

            "eligible_candidate_count":
                eligible_count,

            "eligible_candidate_frames":
                eligible_frames,

            "backward36_score":
                backward_score,

            "backward36_best_frame":
                backward_frame,

            "future_pair_score_36":
                future36,

            "max_event_score":
                max_event,

            "status":
                status,
        })

        print(
            f"{case:28s} "
            f"GT={gt_text:10s} "
            f"r0={str(response_onset):>4s} "
            f"eligible={eligible_count:2d} "
            f"B36={backward_score!s:>10s} "
            f"F36={future36!s:>10s} "
            f"{status}"
        )

    return pd.DataFrame(
        results
    )


# ============================================================
# Run all requested groups.
# ============================================================

old = process_source(
    "OLD29",
    OLD_CAND,
    OLD_VIDEO,
)

pilot = process_source(
    "PILOT5",
    PILOT_CAND,
    PILOT_VIDEO,
    selected_cases=PILOT5,
)

new = process_source(
    "COSMOS25_4",
    NEW_CAND,
    NEW_VIDEO,
    selected_cases=set(
        NEW4_LABEL.keys()
    ),
)


all38 = pd.concat(
    [
        old,
        pilot,
        new,
    ],
    ignore_index=True,
)


# ============================================================
# Structural sanity checks.
# ============================================================

if len(old) != 29:
    raise RuntimeError(
        f"Expected OLD29=29, got {len(old)}"
    )

if len(pilot) != 5:
    raise RuntimeError(
        f"Expected PILOT5=5, got {len(pilot)}"
    )

if len(new) != 4:
    raise RuntimeError(
        f"Expected COSMOS25_4=4, got {len(new)}"
    )

if len(all38) != 38:
    raise RuntimeError(
        f"Expected total=38, got {len(all38)}"
    )


binary = all38[
    all38[
        "label"
    ].notna()
].copy()

binary[
    "label"
] = binary[
    "label"
].astype(int)


if len(binary) != 32:
    raise RuntimeError(
        "Expected 32 resolved binary cases "
        f"(24 old + 5 Pilot + 3 new), "
        f"got {len(binary)}"
    )


all38.to_csv(
    OUT_ALL,
    index=False,
)

binary.to_csv(
    OUT_BINARY,
    index=False,
)


# ============================================================
# Evaluation
# ============================================================

def report_auc(
    title,
    df,
):

    print()
    print("=" * 110)
    print(title)
    print("=" * 110)

    print(
        "resolved binary:",
        len(df),
    )

    print(
        "CONTACT:",
        int(
            (
                df["label"]
                == 1
            ).sum()
        ),
    )

    print(
        "NO_CONTACT:",
        int(
            (
                df["label"]
                == 0
            ).sum()
        ),
    )

    for score_col in [
        "future_pair_score_36",
        "backward36_score",
    ]:

        q = df[
            df[
                score_col
            ].notna()
        ].copy()

        auc = pairwise_auc(
            q[
                "label"
            ].to_numpy(),
            q[
                score_col
            ].to_numpy(),
        )

        pos = int(
            (
                q["label"]
                == 1
            ).sum()
        )

        neg = int(
            (
                q["label"]
                == 0
            ).sum()
        )

        print(
            f"{score_col:24s} "
            f"coverage={len(q)}/{len(df)} "
            f"CONTACT={pos} "
            f"NO_CONTACT={neg} "
            f"AUROC={auc:.4f}"
        )


# Primary9
report_auc(
    "PRIMARY9",
    binary[
        (
            binary["source"]
            == "OLD29"
        )
        &
        (
            binary["split"]
            .astype(str)
            == "PRIMARY9"
        )
    ],
)

# Fresh20 clear binary
report_auc(
    "FRESH20 CLEAR BINARY",
    binary[
        (
            binary["source"]
            == "OLD29"
        )
        &
        (
            binary["split"]
            .astype(str)
            == "FRESH20"
        )
    ],
)

# Original 9+20 binary
report_auc(
    "ORIGINAL 9+20 CLEAR BINARY",
    binary[
        binary[
            "source"
        ]
        == "OLD29"
    ],
)

# Five extra contacts + three resolved new negatives
report_auc(
    "EXTRA 5 CONTACT + 3 RESOLVED COSMOS2.5 NEGATIVES",
    binary[
        binary[
            "source"
        ].isin(
            [
                "PILOT5",
                "COSMOS25_4",
            ]
        )
    ],
)

# Entire requested pool
report_auc(
    "ALL REQUESTED: 9 + 20 + 5 CONTACT + CURRENT 4",
    binary,
)


print()
print("=" * 110)
print("ALL 38 CASES — BACKWARD36 DETAILS")
print("=" * 110)

show_cols = [
    "source",
    "split",
    "case",
    "gt",
    "first_response_frame",
    "eligible_candidate_frames",
    "backward36_best_frame",
    "backward36_score",
    "future_pair_score_36",
    "status",
]

print(
    all38[
        show_cols
    ]
    .sort_values(
        [
            "source",
            "case",
        ]
    )
    .to_string(
        index=False
    )
)


print()
print("=" * 110)
print("BACKWARD36 RANKING — RESOLVED BINARY CASES")
print("=" * 110)

print(
    binary[
        [
            "source",
            "case",
            "gt",
            "first_response_frame",
            "backward36_best_frame",
            "backward36_score",
            "future_pair_score_36",
        ]
    ]
    .sort_values(
        "backward36_score",
        ascending=False,
        na_position="last",
    )
    .to_string(
        index=False
    )
)


print()
print("Saved all38 :", OUT_ALL)
print("Saved binary:", OUT_BINARY)

print()
print("Definition:")
print(
    "Backward36 considers ONLY geometry-selected "
    "Contact Event V2 candidates satisfying:"
)
print(
    "    first_response - 36 <= candidate_frame "
    "<= first_response"
)
print(
    "Later contact can NEVER retroactively explain "
    "an earlier response."
)
print(
    "No reliable response or no eligible candidate "
    "=> ABSTAIN / NaN, never score 0."
)
