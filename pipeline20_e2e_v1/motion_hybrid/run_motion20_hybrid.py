from pathlib import Path
import csv
import numpy as np


ROOT = Path(
    "/shared/ssd_30T/zhuoyingyang/physact/sam3_robowm"
)

ABS_ROOT = (
    ROOT
    / "pipeline20_e2e_v1/motion_abs"
)

LOCAL_ROOT = (
    ROOT
    / "pipeline20_e2e_v1/motion_local"
)

OUT = (
    ROOT
    / "pipeline20_e2e_v1/motion_hybrid"
)

OUT.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# HUMAN GT
# Evaluation only.
# Never used to form hybrid candidates.
# ============================================================

CASES = [('Cosmos3_seed101_0004', 123, 124), ('Cosmos3_seed101_0014', 123, 124), ('Cosmos3_seed101_0015', 123, 124), ('Cosmos3_seed101_0029', 123, 124), ('Cosmos3_seed101_0034', 123, 124), ('Cosmos3_seed101_0040', 123, 124), ('Cosmos3_seed101_0042', 123, 124), ('Cosmos3_seed102_0015', 123, 124), ('Cosmos3_seed102_0019', 123, 124), ('Cosmos3_seed102_0025', 123, 124), ('Cosmos3_seed102_0027', 123, 124), ('Cosmos3_seed102_0029', 123, 124), ('Cosmos3_seed102_0036', 123, 124), ('Cosmos3_seed102_0042', 123, 124), ('Cosmos3_seed103_0003', 123, 124), ('Cosmos3_seed103_0004', 123, 124), ('Cosmos3_seed103_0026', 123, 124), ('Cosmos3_seed103_0028', 123, 124), ('Cosmos3_seed103_0035', 123, 124), ('Cosmos3_seed103_0036', 123, 124)]


# ============================================================
# FROZEN persistence rule
#
# EXACT SAME persistence idea as before.
# No new threshold introduced here.
# ============================================================

PERSIST_WINDOW = 3

PERSIST_REQUIRED = 2


# ============================================================
# CSV HELPERS
# ============================================================

def read_csv(path):

    with open(
        path,
        newline=""
    ) as f:

        return list(
            csv.DictReader(f)
        )


def to_float(x):

    try:
        return float(x)

    except Exception:
        return np.nan


def to_int(x):

    try:
        return int(
            float(x)
        )

    except Exception:
        return 0


# ============================================================
# DETECTION
# ============================================================

def detect_with_persistence(
    frames,
    candidate
):

    frames = np.asarray(
        frames,
        dtype=int
    )

    candidate = np.asarray(
        candidate,
        dtype=bool
    )

    if len(frames) == 0:

        return None

    frame_to_index = {
        int(f): i
        for i, f in enumerate(frames)
    }

    first = int(
        frames.min()
    )

    last = int(
        frames.max()
    )

    for t in range(
        first,
        last + 1
    ):

        if t not in frame_to_index:
            continue

        i = frame_to_index[t]

        # Same logic as original:
        # onset itself must be candidate.
        if not candidate[i]:
            continue

        count = 0

        for tt in range(
            t,
            t + PERSIST_WINDOW
        ):

            if tt not in frame_to_index:
                continue

            j = frame_to_index[tt]

            count += int(
                candidate[j]
            )

        if count >= PERSIST_REQUIRED:

            return t

    return None


# ============================================================
# LOAD PREVIOUS SUMMARY ONSETS
# ============================================================

abs_summary = {}

for r in read_csv(
    ABS_ROOT
    / "MOTION_PRIMARY9_AUDIT.csv"
):

    x = r[
        "auto_onset"
    ]

    abs_summary[
        r["case"]
    ] = (
        int(x)
        if x != ""
        else None
    )


local_summary = {}

for r in read_csv(
    LOCAL_ROOT
    / "MOTION_PRIMARY9_LOCALREF_AUDIT.csv"
):

    x = r[
        "localref_onset"
    ]

    local_summary[
        r["case"]
    ] = (
        int(x)
        if x != ""
        else None
    )


# ============================================================
# RUN
# ============================================================

summary = []


print("=" * 125)
print("PIPELINE20 HYBRID CAMERA-MOTION VETO")
print("=" * 125)

print()
print(
    "Hybrid candidate = "
    "ABSOLUTE candidate AND "
    "LOCAL-RELATIVE candidate"
)

print(
    "No new motion threshold."
)

print(
    "Persistence remains 2-of-3."
)


for (
    case,
    human_start,
    human_end,
) in CASES:

    print()
    print("=" * 110)
    print(case)
    print("=" * 110)

    abs_path = (
        ABS_ROOT
        / case
        / "motion_trace.csv"
    )

    local_path = (
        LOCAL_ROOT
        / case
        / "motion_trace_localref.csv"
    )

    abs_rows = read_csv(
        abs_path
    )

    local_rows = read_csv(
        local_path
    )


    # --------------------------------------------------------
    # Index local-reference trace by frame
    # --------------------------------------------------------

    local_by_frame = {

        int(r["frame"]): r

        for r in local_rows
    }


    combined = []


    for a in abs_rows:

        t = int(
            a["frame"]
        )

        if t not in local_by_frame:

            continue

        l = (
            local_by_frame[t]
        )

        abs_candidate = (
            to_int(
                a[
                    "motion_candidate"
                ]
            )
        )

        local_candidate = (
            to_int(
                l[
                    "motion_candidate"
                ]
            )
        )

        # ====================================================
        # CAMERA VETO
        #
        # Keep absolute candidate ONLY if the
        # local-relative detector also supports it.
        #
        # No new numeric threshold.
        # ====================================================

        hybrid_candidate = int(
            abs_candidate == 1
            and
            local_candidate == 1
        )


        combined.append({

            "frame":
                t,

            "abs_candidate":
                abs_candidate,

            "local_candidate":
                local_candidate,

            "hybrid_candidate":
                hybrid_candidate,

            # Original absolute signals
            "abs_rigid":
                to_float(
                    a[
                        "rigid_speed_norm"
                    ]
                ),

            "abs_q75":
                to_float(
                    a[
                        "point_q75_norm"
                    ]
                ),

            "abs_rigid_thr":
                to_float(
                    a[
                        "rigid_threshold"
                    ]
                ),

            "abs_point_thr":
                to_float(
                    a[
                        "point_threshold"
                    ]
                ),

            # Camera/local relative signals
            "camera_ref_speed":
                to_float(
                    l[
                        "ref_speed_norm"
                    ]
                ),

            "relative_rigid":
                to_float(
                    l[
                        "relative_rigid_norm"
                    ]
                ),

            "relative_q75":
                to_float(
                    l[
                        "relative_q75_norm"
                    ]
                ),

            "relative_rigid_thr":
                to_float(
                    l[
                        "rigid_threshold"
                    ]
                ),

            "relative_point_thr":
                to_float(
                    l[
                        "point_threshold"
                    ]
                ),
        })


    frames = [
        r["frame"]
        for r in combined
    ]

    hybrid_candidates = [
        r[
            "hybrid_candidate"
        ]
        for r in combined
    ]


    hybrid_onset = (
        detect_with_persistence(
            frames,
            hybrid_candidates
        )
    )


    # --------------------------------------------------------
    # Save combined trace
    # --------------------------------------------------------

    trace_path = (
        OUT
        / f"{case}_hybrid_trace.csv"
    )

    fields = [

        "frame",

        "abs_candidate",

        "local_candidate",

        "hybrid_candidate",

        "abs_rigid",

        "abs_q75",

        "abs_rigid_thr",

        "abs_point_thr",

        "camera_ref_speed",

        "relative_rigid",

        "relative_q75",

        "relative_rigid_thr",

        "relative_point_thr",
    ]

    with open(
        trace_path,
        "w",
        newline=""
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fields
        )

        writer.writeheader()

        writer.writerows(
            combined
        )


    abs_onset = (
        abs_summary[
            case
        ]
    )

    local_onset = (
        local_summary[
            case
        ]
    )


    if hybrid_onset is None:

        err = None

        inside = False

        within2 = False

    else:

        err = (
            hybrid_onset
            -
            human_start
        )

        inside = (
            human_start
            <= hybrid_onset
            <= human_end
        )

        within2 = (
            abs(err)
            <= 2
        )


    print(
        "HUMAN:",
        f"{human_start}-{human_end}"
    )

    print(
        "ABSOLUTE:",
        abs_onset
    )

    print(
        "LOCAL-REL:",
        local_onset
    )

    print(
        "HYBRID:",
        hybrid_onset
    )

    print(
        "ERROR:",
        err
    )

    print(
        "IN HUMAN:",
        inside
    )

    print(
        "WITHIN ±2:",
        within2
    )


    # --------------------------------------------------------
    # Print diagnostics around:
    #
    # 1. old absolute onset
    # 2. human onset
    # --------------------------------------------------------

    centers = []

    if abs_onset is not None:

        centers.append(
            (
                "ABS",
                abs_onset
            )
        )

    centers.append(
        (
            "HUMAN",
            human_start
        )
    )


    by_frame = {
        r["frame"]: r
        for r in combined
    }


    printed = set()

    for tag, center in centers:

        if center in printed:
            continue

        printed.add(
            center
        )

        print()

        print(
            f"--- {tag} CENTER "
            f"{center} ---"
        )

        for t in range(
            max(
                min(frames),
                center - 2
            ),
            min(
                max(frames),
                center + 2
            )
            + 1
        ):

            if t not in by_frame:

                continue

            r = (
                by_frame[t]
            )

            print(

                f"f={t:3d}  "

                f"ABS={r['abs_candidate']}  "

                f"LOCAL={r['local_candidate']}  "

                f"HYB={r['hybrid_candidate']}  "

                f"absRigid="
                f"{r['abs_rigid']:.5f}  "

                f"absQ75="
                f"{r['abs_q75']:.5f}  "

                f"cam="
                f"{r['camera_ref_speed']:.5f}  "

                f"relRigid="
                f"{r['relative_rigid']:.5f}  "

                f"relQ75="
                f"{r['relative_q75']:.5f}"
            )


    summary.append({

        "case":
            case,

        "human_start":
            human_start,

        "human_end":
            human_end,

        "absolute_onset":
            (
                abs_onset
                if abs_onset is not None
                else ""
            ),

        "local_relative_onset":
            (
                local_onset
                if local_onset is not None
                else ""
            ),

        "hybrid_onset":
            (
                hybrid_onset
                if hybrid_onset is not None
                else ""
            ),

        "error_vs_human_start":
            (
                err
                if err is not None
                else ""
            ),

        "inside_human_interval":
            int(
                inside
            ),

        "within_2_frames":
            int(
                within2
            ),
    })


# ============================================================
# SAVE SUMMARY
# ============================================================

summary_path = (
    OUT
    / "MOTION_PRIMARY9_HYBRID_VETO_AUDIT.csv"
)

fields = [

    "case",

    "human_start",

    "human_end",

    "absolute_onset",

    "local_relative_onset",

    "hybrid_onset",

    "error_vs_human_start",

    "inside_human_interval",

    "within_2_frames",
]

with open(
    summary_path,
    "w",
    newline=""
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=fields
    )

    writer.writeheader()

    writer.writerows(
        summary
    )


# ============================================================
# FINAL TABLE
# ============================================================

print()
print("=" * 125)

print(
    "PRIMARY9 HYBRID CAMERA-VETO AUDIT"
)

print("=" * 125)

print(

    f"{'CASE':29s} "

    f"{'HUMAN':>9s} "

    f"{'ABS':>5s} "

    f"{'LOCAL':>6s} "

    f"{'HYB':>5s} "

    f"{'ERR':>5s} "

    f"{'IN_GT':>6s} "

    f"{'±2':>4s}"
)


for r in summary:

    human = (
        f"{r['human_start']}-"
        f"{r['human_end']}"
    )

    print(

        f"{r['case']:29s} "

        f"{human:>9s} "

        f"{str(r['absolute_onset']):>5s} "

        f"{str(r['local_relative_onset']):>6s} "

        f"{str(r['hybrid_onset']):>5s} "

        f"{str(r['error_vs_human_start']):>5s} "

        f"{r['inside_human_interval']:6d} "

        f"{r['within_2_frames']:4d}"
    )


detected = [

    r

    for r in summary

    if r[
        "hybrid_onset"
    ] != ""
]


inside_n = sum(

    r[
        "inside_human_interval"
    ]

    for r in summary
)


near_n = sum(

    r[
        "within_2_frames"
    ]

    for r in summary
)


errors = [

    abs(
        int(
            r[
                "error_vs_human_start"
            ]
        )
    )

    for r in detected
]


print()
print("=" * 125)
print("SUMMARY")
print("=" * 125)


print(
    "Detected:",
    f"{len(detected)}/{len(CASES)}"
)

print(
    "Inside human interval:",
    f"{inside_n}/{len(CASES)}"
)

print(
    "Within ±2:",
    f"{near_n}/{len(CASES)}"
)


if errors:

    print(
        "Median |error|:",
        float(
            np.median(
                errors
            )
        )
    )

    print(
        "Max |error|:",
        int(
            np.max(
                errors
            )
        )
    )


print()
print(
    "Definition:"
)

print(
    "HYBRID(t) = "
    "ABS_CANDIDATE(t) AND "
    "LOCAL_RELATIVE_CANDIDATE(t)"
)

print()
print(
    "No new numeric threshold."
)

print(
    "Original absolute thresholds unchanged."
)

print(
    "Original local-relative thresholds unchanged."
)

print(
    "Persistence unchanged: "
    "2 candidates in 3-frame window."
)

print()
print("Saved:")
print(
    summary_path
)
