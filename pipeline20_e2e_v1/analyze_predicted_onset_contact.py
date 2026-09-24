#!/usr/bin/env python3

from pathlib import Path
import ast
import csv
import cv2
import inspect
import pickle
import re
import types

import numpy as np


# ============================================================
# PATHS
# ============================================================

BASE = Path(
    "/shared/ssd_30T/zhuoyingyang/physact/"
    "sam3_robowm"
)

ROOT = BASE / "pipeline20_e2e_v1"

OLD_ANALYZER = (
    BASE
    / "contact_depth_primary9_v1"
    / "analyze_depth_contact_primary9.py"
)

ONSET_CSV = (
    ROOT
    / "PREDICTED_MOTION_ONSETS.csv"
)

MANIFEST_CSV = (
    ROOT
    / "PIPELINE20_MANIFEST.csv"
)

PREFLIGHT_CSV = (
    ROOT
    / "TARGET_MASK_PREFLIGHT.csv"
)

ROBOTSEG_ROOT = (
    ROOT
    / "robotseg_ee"
)

DEPTH_ROOT = (
    ROOT
    / "contact_depth"
    / "depth"
)

PAI_ROOT = Path(
    "/shared/ssd_30T/zhuoyingyang/physact/"
    "pai_results/components_all"
)

OUT_CSV = (
    ROOT
    / "PREDICTED_ONSET_CONTACT_RAW.csv"
)


# ============================================================
# LOAD OLD ANALYZER DEFINITIONS ONLY
#
# Important:
#   - imports old helper functions
#   - imports old numeric constants
#   - DOES NOT execute old Primary9 main loop
# ============================================================

def load_old_analyzer_library(path):
    source = path.read_text()
    tree = ast.parse(
        source,
        filename=str(path),
    )

    keep = []

    for node in tree.body:

        if isinstance(
            node,
            (
                ast.Import,
                ast.ImportFrom,
                ast.FunctionDef,
                ast.AsyncFunctionDef,
                ast.ClassDef,
            ),
        ):
            keep.append(node)
            continue

        if isinstance(
            node,
            ast.Assign,
        ):
            names = []

            for t in node.targets:
                if isinstance(
                    t,
                    ast.Name,
                ):
                    names.append(t.id)

            # Keep numeric/config constants and CASES.
            if (
                names
                and
                all(
                    n.isupper()
                    or
                    n == "CASES"
                    for n in names
                )
            ):
                keep.append(node)

        elif isinstance(
            node,
            ast.AnnAssign,
        ):
            if isinstance(
                node.target,
                ast.Name,
            ):
                n = node.target.id

                if (
                    n.isupper()
                    or
                    n == "CASES"
                ):
                    keep.append(node)

    lib_tree = ast.Module(
        body=keep,
        type_ignores=[],
    )

    ast.fix_missing_locations(
        lib_tree
    )

    ns = {
        "__name__":
            "old_contact_library",
        "__file__":
            str(path),
    }

    exec(
        compile(
            lib_tree,
            str(path),
            "exec",
        ),
        ns,
    )

    return types.SimpleNamespace(
        **ns
    )


OLD = load_old_analyzer_library(
    OLD_ANALYZER
)

required = [
    "load_object_masks",
    "analyze_frame",
    "contact_side_boundary",
    "nearest_pairs",
    "erode_mask",
    "local_depth",
]

print(
    "========================================"
)
print(
    "OLD CONTACT ANALYZER FUNCTIONS"
)
print(
    "========================================"
)

for name in required:
    if not hasattr(
        OLD,
        name,
    ):
        raise RuntimeError(
            f"Old analyzer missing function: "
            f"{name}"
        )

    fn = getattr(
        OLD,
        name,
    )

    print(
        f"{name:25s}",
        inspect.signature(fn),
    )


# ============================================================
# CSV HELPERS
# ============================================================

def read_csv(path):
    if not path.exists():
        return []

    with open(
        path,
        newline="",
    ) as f:
        return list(
            csv.DictReader(f)
        )


onset_rows = read_csv(
    ONSET_CSV
)

manifest_rows = read_csv(
    MANIFEST_CSV
)

preflight_rows = read_csv(
    PREFLIGHT_CSV
)


if len(onset_rows) != 20:
    raise RuntimeError(
        f"Expected 20 onset rows, "
        f"got {len(onset_rows)}"
    )


# ============================================================
# CASE HELPERS
# ============================================================

CASE_RE = re.compile(
    r"^Cosmos3_(seed\d{3})_(\d{4})$"
)


def case_parts(case):
    m = CASE_RE.match(case)

    if not m:
        raise RuntimeError(
            f"Bad case name: {case}"
        )

    return (
        m.group(1),
        m.group(2),
    )


def pkl_path_for_case(case):
    seed, vid = case_parts(case)

    return (
        PAI_ROOT
        / f"cosmos3_{seed}"
        / "sam"
        / f"{vid}.pkl"
    )


def available_phrases(pkl_path):
    with open(
        pkl_path,
        "rb",
    ) as f:
        x = pickle.load(f)

    out = []

    for item in x:
        if not isinstance(
            item,
            dict,
        ):
            continue

        phrase = item.get(
            "phrase"
        )

        if phrase is not None:
            out.append(
                str(phrase).strip()
            )

    return out


# ============================================================
# TARGET PHRASE RESOLUTION
#
# First use our frozen preflight / manifest.
# Only if those don't expose phrase directly,
# use the unique non-environment PAI phrase.
# ============================================================

ENVIRONMENT_PHRASES = {
    "robot",
    "robot arm",
    "robot arm grip",
    "robot grip",
    "gripper",
    "arm",
    "table",
    "trash bin",
    "cabinet",
    "drawer",
    "background",
    "floor",
}


def row_matches_case(
    row,
    case,
    seed,
    vid,
):
    values = {
        str(v).strip()
        for v in row.values()
        if v is not None
    }

    if case in values:
        return True

    has_seed = seed in values

    has_vid = (
        vid in values
        or
        str(int(vid)) in values
    )

    return (
        has_seed
        and
        has_vid
    )


def resolve_target_phrase(
    case,
    pkl_path,
):
    seed, vid = case_parts(
        case
    )

    phrases = available_phrases(
        pkl_path
    )

    # --------------------------------------------------------
    # 1. Search frozen preflight first, then manifest.
    # If any cell exactly equals one of the PAI phrases,
    # that's the safest match.
    # --------------------------------------------------------

    for source_rows in (
        preflight_rows,
        manifest_rows,
    ):

        for row in source_rows:

            if not row_matches_case(
                row,
                case,
                seed,
                vid,
            ):
                continue

            candidates = []

            for value in row.values():
                if value is None:
                    continue

                value = (
                    str(value)
                    .strip()
                )

                if value in phrases:
                    candidates.append(
                        value
                    )

            candidates = list(
                dict.fromkeys(
                    candidates
                )
            )

            if len(
                candidates
            ) == 1:
                return (
                    candidates[0],
                    "FROZEN_PREFLIGHT",
                )

    # --------------------------------------------------------
    # 2. Conservative fallback:
    # require exactly ONE non-robot/environment phrase.
    # --------------------------------------------------------

    candidates = [
        p
        for p in phrases
        if p.lower()
        not in ENVIRONMENT_PHRASES
    ]

    if len(
        candidates
    ) == 1:
        return (
            candidates[0],
            "UNIQUE_NON_ENV",
        )

    raise RuntimeError(
        f"{case}: cannot resolve unique target phrase.\n"
        f"Available phrases: {phrases}\n"
        f"Remaining candidates: {candidates}"
    )


# ============================================================
# FLEXIBLE CALLER
#
# We use the OLD functions unchanged.
# This only maps their parameter names to our fresh inputs.
# ============================================================

def call_load_object_masks(
    pkl_path,
    phrase,
    case,
):
    """
    Reuse frozen old analyzer exactly.

    Old signature:
        load_object_masks(seed, vid, phrase)

    Example:
        Cosmos3_seed101_0023
        -> seed101, 0023
    """
    seed, vid = case_parts(case)

    return OLD.load_object_masks(
        seed,
        vid,
        phrase,
    )


def normalize_object_sequence(x):

    # Common case: ndarray T,H,W
    if isinstance(
        x,
        np.ndarray,
    ):
        return x

    # Some loaders may return
    # (masks, metadata)
    if isinstance(
        x,
        tuple,
    ):
        for item in x:
            if isinstance(
                item,
                np.ndarray,
            ):
                if (
                    item.ndim
                    >= 3
                ):
                    return item

            if isinstance(
                item,
                list,
            ):
                if len(item):
                    return item

    if isinstance(
        x,
        list,
    ):
        return x

    if isinstance(
        x,
        dict,
    ):
        for key in (
            "masks",
            "mask",
            "object_masks",
            "segmentation_masks",
        ):
            if key in x:
                return x[key]

    raise RuntimeError(
        "Unknown object-mask return type: "
        f"{type(x)}"
    )


def object_mask_at(
    seq,
    frame_idx,
):

    if isinstance(
        seq,
        dict,
    ):
        for key in (
            frame_idx,
            str(frame_idx),
            f"{frame_idx:06d}",
        ):
            if key in seq:
                return np.asarray(
                    seq[key]
                ).astype(bool)

        raise KeyError(
            f"No frame {frame_idx} "
            "in object mask dictionary."
        )

    m = np.asarray(
        seq[frame_idx]
    )

    if (
        m.ndim == 3
        and
        m.shape[0] == 1
    ):
        m = m[0]

    return (
        m > 0
    )


def load_ee_mask(
    case,
    frame_idx,
):

    p = (
        ROBOTSEG_ROOT
        / case
        / "ee_masks"
        / f"{frame_idx:06d}.png"
    )

    if not p.exists():
        raise FileNotFoundError(
            f"Missing EE mask: {p}"
        )

    m = cv2.imread(
        str(p),
        cv2.IMREAD_GRAYSCALE,
    )

    if m is None:
        raise RuntimeError(
            f"Cannot read EE mask: {p}"
        )

    return (
        m > 0
    )


def load_depth_frame(
    case,
    frame_idx,
):

    p = (
        DEPTH_ROOT
        / case
        / "depth_raw.npy"
    )

    if not p.exists():
        raise FileNotFoundError(
            f"Missing depth: {p}"
        )

    d = np.load(
        p,
        mmap_mode="r",
    )

    if not (
        0
        <= frame_idx
        < d.shape[0]
    ):
        raise RuntimeError(
            f"{case}: frame {frame_idx} "
            f"outside depth shape {d.shape}"
        )

    return np.asarray(
        d[frame_idx]
    )


def call_analyze_frame(
    case,
    frame_idx,
    ee_mask,
    object_mask,
    depth_frame,
):

    fn = OLD.analyze_frame

    sig = inspect.signature(
        fn
    )

    kwargs = {}

    for name, param in (
        sig.parameters.items()
    ):

        n = name.lower()

        found = True

        # DEPTH
        if "depth" in n:
            value = depth_frame

        # OBJECT MASK
        elif (
            (
                "object" in n
                or
                n.startswith("obj")
            )
            and
            "mask" in n
        ):
            value = object_mask

        # EE / ROBOT MASK
        elif (
            (
                "ee" in n
                or
                "gripper" in n
                or
                "robot" in n
            )
            and
            "mask" in n
        ):
            value = ee_mask

        # FRAME INDEX
        elif n in {
            "frame_idx",
            "frame_index",
            "frame_id",
            "frame",
            "t",
            "idx",
        }:
            value = frame_idx

        # CASE
        elif n in {
            "case",
            "case_name",
            "name",
        }:
            value = case

        else:
            found = False

        if found:
            kwargs[name] = value

        elif (
            param.default
            is inspect.Parameter.empty
        ):
            raise RuntimeError(
                "Cannot map required parameter "
                f"for analyze_frame: {name}\n"
                f"signature={sig}"
            )

    result = fn(
        **kwargs
    )

    if result is None:
        raise RuntimeError(
            f"{case} frame {frame_idx}: "
            "old analyze_frame returned None"
        )

    if not isinstance(
        result,
        dict,
    ):
        raise RuntimeError(
            "Expected analyze_frame to "
            "return dict; got "
            f"{type(result)}"
        )

    return result


# ============================================================
# ONE-CASE FUNCTION
# ============================================================

def analyze_case(
    case,
    frame_idx,
):

    pkl_path = pkl_path_for_case(
        case
    )

    if not pkl_path.exists():
        raise FileNotFoundError(
            pkl_path
        )

    phrase, phrase_source = (
        resolve_target_phrase(
            case,
            pkl_path,
        )
    )

    object_seq = (
        call_load_object_masks(
            pkl_path,
            phrase,
            case,
        )
    )

    object_seq = (
        normalize_object_sequence(
            object_seq
        )
    )

    object_mask = (
        object_mask_at(
            object_seq,
            frame_idx,
        )
    )

    ee_mask = (
        load_ee_mask(
            case,
            frame_idx,
        )
    )

    depth_frame = (
        load_depth_frame(
            case,
            frame_idx,
        )
    )

    if (
        object_mask.shape
        != ee_mask.shape
    ):
        raise RuntimeError(
            f"{case}: object/EE shape mismatch "
            f"{object_mask.shape} vs "
            f"{ee_mask.shape}"
        )

    result = call_analyze_frame(
        case,
        frame_idx,
        ee_mask,
        object_mask,
        depth_frame,
    )

    result = dict(
        result
    )

    result[
        "target_phrase"
    ] = phrase

    result[
        "phrase_source"
    ] = phrase_source

    return result


# ============================================================
# SELF-CHECK ON ORIGINAL DEV CASE
#
# This verifies that reusing the old functions reproduces
# the old frozen result before touching fresh20.
# ============================================================

def run_self_check():

    case = (
        "Cosmos3_seed101_0023"
    )

    frame_idx = 123

    old_ee = (
        BASE
        / "robotseg_primary9_ee_v1"
        / case
        / "ee_masks"
        / f"{frame_idx:06d}.png"
    )

    old_depth = (
        BASE
        / "contact_depth_primary9_v1"
        / "depth"
        / case
        / "depth_raw.npy"
    )

    old_pkl = (
        PAI_ROOT
        / "cosmos3_seed101"
        / "sam"
        / "0023.pkl"
    )

    if not (
        old_ee.exists()
        and
        old_depth.exists()
        and
        old_pkl.exists()
    ):
        print()
        print(
            "SELF-CHECK skipped: "
            "old DEV files incomplete."
        )
        return

    obj_seq = (
        call_load_object_masks(
            old_pkl,
            "yellow cube",
            case,
        )
    )

    obj_seq = (
        normalize_object_sequence(
            obj_seq
        )
    )

    obj = object_mask_at(
        obj_seq,
        frame_idx,
    )

    ee = cv2.imread(
        str(old_ee),
        cv2.IMREAD_GRAYSCALE,
    )

    ee = (
        ee > 0
    )

    d = np.load(
        old_depth,
        mmap_mode="r",
    )

    result = call_analyze_frame(
        case,
        frame_idx,
        ee,
        obj,
        np.asarray(
            d[frame_idx]
        ),
    )

    print()
    print(
        "========================================"
    )
    print(
        "DEV SELF-CHECK seed101_0023 @ 123"
    )
    print(
        "========================================"
    )

    print(
        "d2_q25:",
        result.get(
            "d2_q25"
        ),
        " expected ~0.1974",
    )

    print(
        "dz_q25:",
        result.get(
            "dz_q25"
        ),
        " expected ~0.0633",
    )

    d2 = float(
        result["d2_q25"]
    )

    dz = float(
        result["dz_q25"]
    )

    if abs(
        d2 - 0.1974
    ) > 0.005:
        raise RuntimeError(
            "SELF-CHECK FAILED: "
            f"d2_q25={d2:.6f}, "
            "expected ~0.1974"
        )

    if abs(
        dz - 0.0633
    ) > 0.010:
        raise RuntimeError(
            "SELF-CHECK FAILED: "
            f"dz_q25={dz:.6f}, "
            "expected ~0.0633"
        )

    print(
        "SELF-CHECK PASSED."
    )


run_self_check()


# ============================================================
# FRESH20
# ============================================================

rows_out = []


print()
print(
    "========================================"
)
print(
    "FRESH20 PREDICTED-ONSET CONTACT"
)
print(
    "========================================"
)


for i, row in enumerate(
    onset_rows,
    1,
):

    case = (
        row["case"]
        .strip()
    )

    onset_raw = (
        row.get(
            "predicted_motion_onset",
            "",
        )
        .strip()
    )

    print()
    print(
        f"[{i:02d}/20] {case}"
    )

    # --------------------------------------------------------
    # NO RESPONSE
    # --------------------------------------------------------

    if onset_raw in {
        "",
        "None",
        "nan",
        "NaN",
    }:

        out = {
            "case":
                case,
            "predicted_motion_onset":
                "",
            "motion_detected":
                0,
            "status":
                "NO_RESPONSE_DETECTED",
            "target_phrase":
                "",
            "phrase_source":
                "",
            "d2_min":
                "",
            "d2_q25":
                "",
            "dz_q25":
                "",
            "dz_med":
                "",
        }

        rows_out.append(
            out
        )

        print(
            "STATUS: NO_RESPONSE_DETECTED"
        )

        continue

    frame_idx = int(
        float(
            onset_raw
        )
    )

    # --------------------------------------------------------
    # ANALYZE THE FROZEN HYBRID ONSET
    # --------------------------------------------------------

    try:
        result = analyze_case(
            case,
            frame_idx,
        )

        out = {
            "case":
                case,
            "predicted_motion_onset":
                frame_idx,
            "motion_detected":
                1,
            "status":
                "OK",
        }

        # Preserve every scalar returned
        # by the old analyzer.
        for k, v in result.items():

            if (
                np.isscalar(v)
                or
                isinstance(
                    v,
                    (
                        str,
                        bool,
                    ),
                )
            ):
                out[k] = v

        rows_out.append(
            out
        )

        print(
            "target:",
            result.get(
                "target_phrase"
            ),
        )

        print(
            "d2_min =",
            result.get(
                "d2_min"
            ),
        )

        print(
            "d2_q25 =",
            result.get(
                "d2_q25"
            ),
        )

        print(
            "dz_q25 =",
            result.get(
                "dz_q25"
            ),
        )

        print(
            "dz_med =",
            result.get(
                "dz_med"
            ),
        )

    except Exception as e:

        out = {
            "case":
                case,
            "predicted_motion_onset":
                frame_idx,
            "motion_detected":
                1,
            "status":
                "ANALYSIS_FAILED",
            "error":
                repr(e),
        }

        rows_out.append(
            out
        )

        print(
            "FAILED:",
            repr(e),
        )


# ============================================================
# SAVE
# ============================================================

preferred = [
    "case",
    "predicted_motion_onset",
    "motion_detected",
    "status",
    "target_phrase",
    "phrase_source",
    "d2_min",
    "d2_q25",
    "dz_q25",
    "dz_med",
]

all_keys = set()

for r in rows_out:
    all_keys.update(
        r.keys()
    )

fieldnames = [
    x
    for x in preferred
    if x in all_keys
]

fieldnames += sorted(
    all_keys
    - set(fieldnames)
)


with open(
    OUT_CSV,
    "w",
    newline="",
) as f:

    w = csv.DictWriter(
        f,
        fieldnames=fieldnames,
    )

    w.writeheader()

    for r in rows_out:
        w.writerow(r)


# ============================================================
# SUMMARY
# ============================================================

ok = [
    r
    for r in rows_out
    if r.get(
        "status"
    ) == "OK"
]

no_response = [
    r
    for r in rows_out
    if r.get(
        "status"
    ) == "NO_RESPONSE_DETECTED"
]

failed = [
    r
    for r in rows_out
    if r.get(
        "status"
    ) == "ANALYSIS_FAILED"
]


print()
print(
    "========================================"
)
print(
    "SUMMARY"
)
print(
    "========================================"
)

print(
    "OK:",
    len(ok),
)

print(
    "NO_RESPONSE_DETECTED:",
    len(no_response),
)

print(
    "ANALYSIS_FAILED:",
    len(failed),
)

print()
print(
    "Saved:",
    OUT_CSV,
)


if failed:

    print()
    print(
        "FAILED CASES:"
    )

    for r in failed:
        print(
            r["case"],
            r.get(
                "error",
                "",
            ),
        )
