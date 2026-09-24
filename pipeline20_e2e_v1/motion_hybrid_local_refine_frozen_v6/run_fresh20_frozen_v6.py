#!/usr/bin/env python3

from pathlib import Path
import csv
import sys

import numpy as np
import pandas as pd


ROOT = Path(
    "/shared/ssd_30T/zhuoyingyang/physact/sam3_robowm"
)

PIPE = ROOT / "pipeline20_e2e_v1"

OUT = (
    PIPE
    / "motion_hybrid_local_refine_frozen_v6"
)

OUT.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# FROZEN V6
#
# IMPORTANT:
# DO NOT tune any of these using Fresh20.
#
# Stage 1:
#   frozen HYBRID 2-of-3 -> robust anchor
#
# Stage 2:
#   earliest LOCAL candidate only within
#   [anchor - 8, anchor]
#
# ============================================================

LOOKBACK = 8


# ============================================================
# HUMAN-VERIFIED FRESH20 RESPONSE ONSET
#
# Only 17 scorable cases.
#
# ABSTAIN:
#   Cosmos3_seed101_0040
#   Cosmos3_seed102_0027
#   Cosmos3_seed103_0026
#
# Intervals are preserved instead of converting them to a
# single frame.
# ============================================================

GT = {
    "Cosmos3_seed101_0004": (118, 118),
    "Cosmos3_seed101_0014": (72, 73),
    "Cosmos3_seed101_0015": (127, 129),
    "Cosmos3_seed101_0029": (128, 130),
    "Cosmos3_seed101_0034": (129, 130),
    "Cosmos3_seed101_0042": (88, 88),

    "Cosmos3_seed102_0015": (165, 170),
    "Cosmos3_seed102_0019": (96, 97),
    "Cosmos3_seed102_0025": (105, 107),
    "Cosmos3_seed102_0029": (105, 105),
    "Cosmos3_seed102_0036": (82, 87),
    "Cosmos3_seed102_0042": (106, 107),

    "Cosmos3_seed103_0003": (75, 76),
    "Cosmos3_seed103_0004": (55, 56),
    "Cosmos3_seed103_0028": (149, 151),
    "Cosmos3_seed103_0035": (188, 190),
    "Cosmos3_seed103_0036": (90, 93),
}


# Previous frozen-HYBRID results from the existing audit.
# Diagnostic only.
#
# We will ALSO reconstruct the anchor directly from trace files
# and verify consistency.
OLD_AUDIT_PRED = {
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


def first_2_of_3(
    frames,
    candidate,
):

    frames = np.asarray(
        frames,
        dtype=int,
    )

    candidate = np.asarray(
        candidate,
        dtype=bool,
    )

    for i in range(
        len(candidate) - 2
    ):

        w = candidate[
            i:i + 3
        ]

        inds = np.where(
            w
        )[0]

        if len(inds) >= 2:

            return int(
                frames[
                    i + inds[0]
                ]
            )

    return None


def interval_error(
    pred,
    lo,
    hi,
):

    """
    Signed distance to human onset interval.

    pred < lo -> negative
    pred inside [lo, hi] -> 0
    pred > hi -> positive
    """

    if pred is None:
        return None

    if pred < lo:
        return int(
            pred - lo
        )

    if pred > hi:
        return int(
            pred - hi
        )

    return 0


def classify(
    pred,
    lo,
    hi,
):

    if pred is None:

        return (
            None,
            "MISSED_RESPONSE",
        )

    err = interval_error(
        pred,
        lo,
        hi,
    )

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


def discover_search_roots():

    roots = [
        PIPE,
    ]

    # Add top-level directories that look Fresh20-related,
    # without recursively scanning the entire project.
    for p in ROOT.iterdir():

        if not p.is_dir():
            continue

        n = p.name.lower()

        if (
            "fresh20" in n
            or
            "pipeline20" in n
        ):

            if p not in roots:
                roots.append(p)

    return roots


SEARCH_ROOTS = discover_search_roots()


def discover_trace(case):

    """
    Find a CSV containing BOTH:
      hybrid_candidate
      local_candidate

    Preference:
      file/path containing 'hybrid'
    """

    short = case.replace(
        "Cosmos3_",
        "",
    )

    matches = []

    for base in SEARCH_ROOTS:

        if not base.exists():
            continue

        for p in base.rglob(
            "*.csv"
        ):

            sp = str(p)

            if (
                case not in sp
                and
                short not in sp
            ):
                continue

            # Avoid our current output dir.
            if OUT in p.parents:
                continue

            matches.append(
                p
            )


    # Sort hybrid-looking files first.
    matches = sorted(
        set(matches),
        key=lambda p: (
            "hybrid" not in str(p).lower(),
            len(str(p)),
            str(p),
        ),
    )


    usable = []

    for p in matches:

        try:

            cols = pd.read_csv(
                p,
                nrows=2,
            ).columns.tolist()

        except Exception:

            continue


        if (
            "frame" in cols
            and
            "hybrid_candidate" in cols
            and
            "local_candidate" in cols
        ):

            usable.append(
                p
            )


    if not usable:

        print()
        print(
            "!!! NO HYBRID TRACE FOUND FOR",
            case,
        )

        print(
            "Matching CSV candidates:"
        )

        for p in matches[:30]:
            print(" ", p)

        return None


    if len(usable) > 1:

        print()
        print(
            f"[DISCOVERY] {case}: "
            f"{len(usable)} usable traces; "
            f"using:"
        )

        for p in usable:
            print(" ", p)

        print(
            "SELECTED:",
            usable[0],
        )


    return usable[0]


def run_case(
    case,
    gt_interval,
):

    lo_gt, hi_gt = (
        gt_interval
    )

    path = discover_trace(
        case
    )

    if path is None:

        return None


    df = pd.read_csv(
        path
    )


    frames = (
        df["frame"]
        .to_numpy(int)
    )

    hybrid = (
        df["hybrid_candidate"]
        .fillna(0)
        .to_numpy(int)
        > 0
    )

    local = (
        df["local_candidate"]
        .fillna(0)
        .to_numpy(int)
        > 0
    )


    # ========================================================
    # FROZEN HYBRID ANCHOR
    # ========================================================

    anchor = first_2_of_3(
        frames,
        hybrid,
    )


    # ========================================================
    # FROZEN V6 LOCAL REFINEMENT
    # ========================================================

    if anchor is None:

        local_refine = None
        final = None
        shift = None
        search_start = None

    else:

        search_start = max(
            int(frames.min()),
            anchor - LOOKBACK,
        )

        refine_window = (
            (frames >= search_start)
            &
            (frames <= anchor)
        )

        local_hits = frames[
            refine_window
            &
            local
        ]


        if len(local_hits):

            local_refine = int(
                local_hits[0]
            )

        else:

            local_refine = None


        if (
            local_refine is not None
            and
            local_refine < anchor
            and
            anchor - local_refine
            <= LOOKBACK
        ):

            final = (
                local_refine
            )

        else:

            final = anchor


        shift = (
            final - anchor
        )


    anchor_err, anchor_status = (
        classify(
            anchor,
            lo_gt,
            hi_gt,
        )
    )

    final_err, final_status = (
        classify(
            final,
            lo_gt,
            hi_gt,
        )
    )


    old_pred = (
        OLD_AUDIT_PRED.get(
            case
        )
    )


    old_err, old_status = (
        classify(
            old_pred,
            lo_gt,
            hi_gt,
        )
    )


    # Did our reconstructed HYBRID anchor match
    # the previous Fresh20 audit?
    if (
        anchor is None
        and
        old_pred is None
    ):

        audit_match = True

    else:

        audit_match = (
            anchor == old_pred
        )


    # Diagnostic around anchor.
    out_case = (
        OUT / case
    )

    out_case.mkdir(
        parents=True,
        exist_ok=True,
    )


    dd = df.copy()

    dd[
        "frozen_v6_anchor"
    ] = (
        anchor
        if anchor is not None
        else np.nan
    )

    dd[
        "frozen_v6_search_start"
    ] = (
        search_start
        if search_start is not None
        else np.nan
    )

    dd[
        "frozen_v6_local_refine"
    ] = (
        local_refine
        if local_refine is not None
        else np.nan
    )

    dd[
        "frozen_v6_final"
    ] = (
        final
        if final is not None
        else np.nan
    )

    dd.to_csv(
        out_case
        / "fresh20_frozen_v6_trace.csv",
        index=False,
    )


    return {
        "case": case,

        "gt_lo": lo_gt,
        "gt_hi": hi_gt,

        "trace_path": str(path),

        "old_audit_pred":
            old_pred,

        "reconstructed_anchor":
            anchor,

        "audit_anchor_match":
            audit_match,

        "anchor_error":
            anchor_err,

        "anchor_status":
            anchor_status,

        "local_refine":
            local_refine,

        "final_pred":
            final,

        "shift":
            shift,

        "final_error":
            final_err,

        "final_status":
            final_status,
    }


# ============================================================
# RUN
# ============================================================

print("=" * 110)
print("FRESH20 — FROZEN HYBRID-ANCHORED LOCAL REFINEMENT V6")
print("=" * 110)

print()
print(
    "FROZEN LOOKBACK =",
    LOOKBACK,
)

print(
    "SCORABLE CASES  =",
    len(GT),
)

print(
    "ABSTAIN CASES   = 3"
)

print()
print("Search roots:")

for p in SEARCH_ROOTS:
    print(" ", p)


results = []


for i, (
    case,
    gt_interval,
) in enumerate(
    GT.items(),
    1,
):

    print()
    print("#" * 110)

    print(
        f"[{i:02d}/{len(GT):02d}] "
        f"{case}"
    )

    print(
        "HUMAN GT =",
        gt_interval,
    )


    r = run_case(
        case,
        gt_interval,
    )


    if r is None:

        print(
            "FAILED TRACE DISCOVERY"
        )

        continue


    results.append(
        r
    )


    print(
        "OLD AUDIT PRED =",
        r[
            "old_audit_pred"
        ],
    )

    print(
        "HYBRID anchor  =",
        r[
            "reconstructed_anchor"
        ],
    )

    print(
        "audit matches? =",
        r[
            "audit_anchor_match"
        ],
    )

    print(
        "LOCAL refine   =",
        r[
            "local_refine"
        ],
    )

    print(
        "FINAL onset    =",
        r[
            "final_pred"
        ],
    )

    print(
        "shift          =",
        r[
            "shift"
        ],
    )

    print(
        "FINAL error    =",
        r[
            "final_error"
        ],
    )

    print(
        "STATUS         =",
        r[
            "final_status"
        ],
    )


# ============================================================
# REQUIRE COMPLETE DISCOVERY
# ============================================================

if len(results) != len(GT):

    print()
    print("=" * 110)
    print(
        "STOP: ONLY FOUND",
        len(results),
        "/",
        len(GT),
        "SCORABLE CASES"
    )
    print("=" * 110)

    print(
        "Do NOT interpret the aggregate result yet."
    )

    print(
        "Send me the FAILED TRACE DISCOVERY output."
    )

    sys.exit(2)


# ============================================================
# VERIFY BASELINE RECONSTRUCTION
# ============================================================

mismatches = [
    r
    for r in results
    if not r[
        "audit_anchor_match"
    ]
]


print()
print("=" * 110)
print("BASELINE TRACE RECONSTRUCTION CHECK")
print("=" * 110)

print(
    "Anchor matches previous audit:",
    f"{len(results) - len(mismatches)}/{len(results)}"
)


if mismatches:

    print()
    print(
        "WARNING: baseline mismatch cases:"
    )

    for r in mismatches:

        print(
            f"{r['case']:27s} "
            f"OLD={str(r['old_audit_pred']):>4s} "
            f"TRACE={str(r['reconstructed_anchor']):>4s}"
        )


# ============================================================
# SUMMARY FUNCTION
# ============================================================

def summarize(
    results,
    status_col,
    error_col,
):

    counts = {
        "WITHIN_2": 0,
        "EARLY": 0,
        "LATE": 0,
        "MISSED_RESPONSE": 0,
    }

    errors = []

    for r in results:

        status = r[
            status_col
        ]

        counts[
            status
        ] += 1

        e = r[
            error_col
        ]

        if e is not None:
            errors.append(
                abs(e)
            )


    if errors:

        median_abs = float(
            np.median(errors)
        )

        mean_abs = float(
            np.mean(errors)
        )

        max_abs = float(
            np.max(errors)
        )

    else:

        median_abs = np.nan
        mean_abs = np.nan
        max_abs = np.nan


    return (
        counts,
        median_abs,
        mean_abs,
        max_abs,
    )


old_counts, old_med, old_mean, old_max = (
    summarize(
        results,
        "anchor_status",
        "anchor_error",
    )
)

new_counts, new_med, new_mean, new_max = (
    summarize(
        results,
        "final_status",
        "final_error",
    )
)


print()
print("=" * 110)
print("FRESH20 FROZEN V6 RESULT")
print("=" * 110)


print()
print("FROZEN HYBRID ANCHOR:")
print(
    f"  WITHIN ±2 = "
    f"{old_counts['WITHIN_2']}/17"
)
print(
    f"  EARLY     = "
    f"{old_counts['EARLY']}/17"
)
print(
    f"  LATE      = "
    f"{old_counts['LATE']}/17"
)
print(
    f"  MISSED    = "
    f"{old_counts['MISSED_RESPONSE']}/17"
)
print(
    f"  median abs error = "
    f"{old_med:.2f}"
)
print(
    f"  mean abs error   = "
    f"{old_mean:.2f}"
)
print(
    f"  max abs error    = "
    f"{old_max:.2f}"
)


print()
print("AFTER FROZEN V6 LOCAL REFINEMENT:")
print(
    f"  WITHIN ±2 = "
    f"{new_counts['WITHIN_2']}/17"
)
print(
    f"  EARLY     = "
    f"{new_counts['EARLY']}/17"
)
print(
    f"  LATE      = "
    f"{new_counts['LATE']}/17"
)
print(
    f"  MISSED    = "
    f"{new_counts['MISSED_RESPONSE']}/17"
)
print(
    f"  median abs error = "
    f"{new_med:.2f}"
)
print(
    f"  mean abs error   = "
    f"{new_mean:.2f}"
)
print(
    f"  max abs error    = "
    f"{new_max:.2f}"
)


print()
print("=" * 110)
print("PER CASE")
print("=" * 110)


for r in results:

    print(
        f"{r['case']:27s} "
        f"GT={r['gt_lo']:3d}-{r['gt_hi']:<3d} "
        f"ANCHOR={str(r['reconstructed_anchor']):>4s} "
        f"LOCAL={str(r['local_refine']):>4s} "
        f"FINAL={str(r['final_pred']):>4s} "
        f"SHIFT={str(r['shift']):>4s} "
        f"ERR={str(r['final_error']):>4s} "
        f"{r['final_status']}"
    )


print()
print("=" * 110)
print("CASES CHANGED BY V6")
print("=" * 110)


changed = [
    r
    for r in results
    if (
        r["reconstructed_anchor"]
        !=
        r["final_pred"]
    )
]


if not changed:

    print(
        "No cases changed."
    )

else:

    for r in changed:

        old_abs = (
            None
            if r["anchor_error"] is None
            else abs(
                r["anchor_error"]
            )
        )

        new_abs = (
            None
            if r["final_error"] is None
            else abs(
                r["final_error"]
            )
        )


        if (
            old_abs is not None
            and
            new_abs is not None
        ):

            if new_abs < old_abs:
                verdict = "IMPROVED"
            elif new_abs > old_abs:
                verdict = "WORSENED"
            else:
                verdict = "SAME_ERROR"

        else:
            verdict = "NA"


        print(
            f"{r['case']:27s} "
            f"{str(r['reconstructed_anchor']):>4s}"
            f" -> "
            f"{str(r['final_pred']):>4s} "
            f"{verdict}"
        )


# ============================================================
# SAVE
# ============================================================

summary_path = (
    OUT
    / "FRESH20_FROZEN_V6_SUMMARY.csv"
)


with open(
    summary_path,
    "w",
    newline="",
) as f:

    fields = list(
        results[0].keys()
    )

    w = csv.DictWriter(
        f,
        fieldnames=fields,
    )

    w.writeheader()
    w.writerows(
        results
    )


print()
print("Saved:")
print(summary_path)
