#!/usr/bin/env python3

from pathlib import Path
import ast
import csv
import pickle
import random
import re

PHYS = Path("/shared/ssd_30T/zhuoyingyang/physact")
ROOT = PHYS / "sam3_robowm"
WORK = ROOT / "pipeline20_e2e_v1"

VIDEO_ROOT = (
    PHYS
    / "cosmos3"
    / "export_robowm68_3seeds"
)

PAI_ROOT = (
    PHYS
    / "pai_results"
    / "components_all"
)

RNG = random.Random(20260823)


# ============================================================
# VIDEO-ID -> TARGET SEMANTIC
#
# Free-object tasks only:
# discard_trash / pick / pull / push / put_on_plate
#
# We exclude:
# 0000-0002 drawer
# 0043-0047 button
# 0048-0057 put_in_drawer
# ============================================================

TARGETS = {
    "0003": "yellow cube",
    "0004": "yellow cube",
    "0005": "blue cube",
    "0006": "brown paper cup",
    "0007": "white cube",
    "0008": "brown paper cup",
    "0009": "brown paper cup",
    "0010": "white cup",
    "0011": "white cup",
    "0012": "white cup",

    "0013": "white cup",
    "0014": "white cup",
    "0015": "white cup",
    "0016": "white cup",
    "0017": "white cube",
    "0018": "white cup",
    "0019": "banana",
    "0020": "banana",
    "0021": "brown paper cup",
    "0022": "brown paper cup",

    "0023": "yellow cube",
    "0024": "yellow cube",
    "0025": "yellow cube",
    "0026": "white tape roll",
    "0027": "white tape roll",
    "0028": "white tape roll",
    "0029": "brown cube",
    "0030": "brown cube",
    "0031": "brown cube",
    "0032": "brown cube",

    "0033": "yellow cube",
    "0034": "yellow cube",
    "0035": "yellow cube",
    "0036": "white tape roll",
    "0037": "white tape roll",
    "0038": "white tape roll",
    "0039": "brown cube",
    "0040": "brown cube",
    "0041": "brown cube",
    "0042": "brown cube",

    "0058": "white cup",
    "0059": "white cup",
    "0060": "white cup",
    "0061": "white cup",
    "0062": "brown paper cup",
    "0063": "brown paper cup",
    "0064": "white cube",
    "0065": "brown paper cup",
    "0066": "green cup",
    "0067": "green cup",
}


# ============================================================
# ALIASES
#
# We do NOT use fuzzy geometry here.
# Aliases only handle wording differences in PAI phrase labels.
# ============================================================

ALIASES = {
    "yellow cube": {
        "yellow cube",
    },

    "blue cube": {
        "blue cube",
    },

    "white cube": {
        "white cube",
        "small white cube",
    },

    "brown cube": {
        "brown cube",
    },

    "white cup": {
        "white cup",
    },

    "brown paper cup": {
        "brown paper cup",
        "brown cup",
    },

    "banana": {
        "banana",
        "yellow banana",
    },

    "white tape roll": {
        "white tape roll",
        "white tape",
        "white roll of tape",
        "tape roll",
    },

    "green cup": {
        "green cup",
    },
}


# ============================================================
# PRIMARY9 EXCLUSION
# ============================================================

PRIMARY9 = {
    ("seed101", "0023"),
    ("seed102", "0022"),
    ("seed101", "0025"),
    ("seed102", "0014"),
    ("seed103", "0037"),

    ("seed101", "0006"),
    ("seed101", "0022"),
    ("seed103", "0011"),
    ("seed103", "0022"),
}


# Known unusable semantic ambiguity / prompt mismatch.
MANUAL_EXCLUDE = {
    ("seed101", "0007"),
    ("seed102", "0007"),
    ("seed103", "0007"),
}


# ============================================================
# HELPERS
# ============================================================

def norm_phrase(x):
    return " ".join(
        str(x)
        .strip()
        .lower()
        .split()
    )


def pai_dir(seed):
    return (
        PAI_ROOT
        / f"cosmos3_{seed}"
        / "sam"
    )


def find_pkl(seed, vid):
    root = pai_dir(seed)

    direct = root / f"{vid}.pkl"

    if direct.exists():
        return direct

    hits = sorted(
        root.glob(f"*{vid}*.pkl")
    )

    if len(hits) == 1:
        return hits[0]

    return None


def load_phrases(pkl_path):
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    phrases = []

    # Expected PAI format:
    # list[dict] with phrase + segmentation_mask_rle
    if isinstance(data, list):
        for item in data:
            if isinstance(item, dict):
                if "phrase" in item:
                    phrases.append(
                        str(item["phrase"])
                    )

    elif isinstance(data, dict):
        # Defensive fallback.
        if "phrase" in data:
            phrases.append(
                str(data["phrase"])
            )

        for v in data.values():
            if isinstance(v, list):
                for item in v:
                    if (
                        isinstance(item, dict)
                        and
                        "phrase" in item
                    ):
                        phrases.append(
                            str(item["phrase"])
                        )

    return phrases


def unique_target_phrase(seed, vid, target):
    pkl = find_pkl(seed, vid)

    if pkl is None:
        return {
            "ok": False,
            "reason": "NO_PKL",
            "pkl": "",
            "phrase": "",
            "available": [],
        }

    try:
        phrases = load_phrases(pkl)
    except Exception as e:
        return {
            "ok": False,
            "reason": f"PKL_ERROR:{e}",
            "pkl": str(pkl),
            "phrase": "",
            "available": [],
        }

    aliases = {
        norm_phrase(x)
        for x in ALIASES[target]
    }

    matches = [
        p
        for p in phrases
        if norm_phrase(p) in aliases
    ]

    if len(matches) != 1:
        return {
            "ok": False,
            "reason": (
                f"MATCHES={len(matches)}"
            ),
            "pkl": str(pkl),
            "phrase": "",
            "available": phrases,
        }

    return {
        "ok": True,
        "reason": "OK",
        "pkl": str(pkl),
        # IMPORTANT:
        # use exact PAI phrase spelling
        "phrase": matches[0],
        "available": phrases,
    }


def replace_assignment(
    text,
    variable_name,
    replacement,
):
    tree = ast.parse(text)

    node = None

    for x in tree.body:
        if isinstance(x, ast.Assign):
            for target in x.targets:
                if (
                    isinstance(target, ast.Name)
                    and
                    target.id == variable_name
                ):
                    node = x
                    break

        elif isinstance(x, ast.AnnAssign):
            if (
                isinstance(x.target, ast.Name)
                and
                x.target.id == variable_name
            ):
                node = x

        if node is not None:
            break

    if node is None:
        raise RuntimeError(
            f"Cannot find assignment {variable_name}"
        )

    lines = text.splitlines(
        keepends=True
    )

    start = node.lineno - 1
    end = node.end_lineno - 1

    prefix = (
        lines[start][:node.col_offset]
    )

    suffix = (
        lines[end][node.end_col_offset:]
    )

    return (
        "".join(lines[:start])
        +
        prefix
        +
        replacement
        +
        suffix
        +
        "".join(lines[end + 1:])
    )


# ============================================================
# PREFLIGHT ALL CANDIDATES
# ============================================================

eligible_by_seed = {
    "seed101": [],
    "seed102": [],
    "seed103": [],
}

audit_rows = []


for seed in [
    "seed101",
    "seed102",
    "seed103",
]:

    for vid, target in TARGETS.items():

        case = (
            f"Cosmos3_{seed}_{vid}"
        )

        if (seed, vid) in PRIMARY9:
            audit_rows.append({
                "case": case,
                "target": target,
                "status": "EXCLUDE_PRIMARY9",
                "phrase": "",
                "pkl": "",
                "available": "",
            })
            continue

        if (seed, vid) in MANUAL_EXCLUDE:
            audit_rows.append({
                "case": case,
                "target": target,
                "status": "EXCLUDE_MANUAL",
                "phrase": "",
                "pkl": "",
                "available": "",
            })
            continue

        video = (
            VIDEO_ROOT
            / seed
            / f"{vid}.mp4"
        )

        if not video.exists():
            audit_rows.append({
                "case": case,
                "target": target,
                "status": "NO_VIDEO",
                "phrase": "",
                "pkl": "",
                "available": "",
            })
            continue

        result = unique_target_phrase(
            seed,
            vid,
            target,
        )

        if not result["ok"]:
            audit_rows.append({
                "case": case,
                "target": target,
                "status": result["reason"],
                "phrase": "",
                "pkl": result["pkl"],
                "available": " | ".join(
                    result["available"]
                ),
            })
            continue

        row = {
            "seed": seed,
            "vid": vid,
            "case": case,
            "target": target,
            "phrase": result["phrase"],
            "pkl": result["pkl"],
            "video": str(video),
        }

        eligible_by_seed[seed].append(
            row
        )

        audit_rows.append({
            "case": case,
            "target": target,
            "status": "ELIGIBLE",
            "phrase": result["phrase"],
            "pkl": result["pkl"],
            "available": " | ".join(
                result["available"]
            ),
        })


# ============================================================
# PRINT PREFLIGHT COUNTS
# ============================================================

print("=" * 100)
print("PAI TARGET-MASK PREFLIGHT")
print("=" * 100)

for seed in eligible_by_seed:
    print(
        seed,
        "eligible =",
        len(
            eligible_by_seed[seed]
        )
    )


# ============================================================
# RANDOM 7 / 7 / 6
# ============================================================

QUOTA = {
    "seed101": 7,
    "seed102": 7,
    "seed103": 6,
}

selected = []

for seed, n in QUOTA.items():

    pool = list(
        eligible_by_seed[seed]
    )

    if len(pool) < n:
        raise RuntimeError(
            f"{seed}: only "
            f"{len(pool)} eligible, "
            f"need {n}"
        )

    picked = RNG.sample(
        pool,
        n,
    )

    picked.sort(
        key=lambda x: x["vid"]
    )

    selected.extend(
        picked
    )


print()
print("=" * 100)
print("NEW RANDOM CLEAN20")
print("=" * 100)

for i, r in enumerate(
    selected,
    start=1,
):
    print(
        f"{i:02d}  "
        f"{r['case']:28s}  "
        f"{r['phrase']}"
    )


# ============================================================
# SAVE MANIFEST + PREFLIGHT AUDIT
# ============================================================

manifest_path = (
    WORK
    / "PIPELINE20_MANIFEST.csv"
)

with open(
    manifest_path,
    "w",
    newline="",
) as f:

    fields = [
        "case",
        "seed",
        "vid",
        "target",
        "phrase",
        "video",
        "pkl",
    ]

    w = csv.DictWriter(
        f,
        fieldnames=fields,
    )

    w.writeheader()
    w.writerows(selected)


audit_path = (
    WORK
    / "TARGET_MASK_PREFLIGHT.csv"
)

with open(
    audit_path,
    "w",
    newline="",
) as f:

    fields = [
        "case",
        "target",
        "status",
        "phrase",
        "pkl",
        "available",
    ]

    w = csv.DictWriter(
        f,
        fieldnames=fields,
    )

    w.writeheader()
    w.writerows(audit_rows)


# ============================================================
# PATCH ABS
#
# human_start/end/contact_gt are DUMMY AUDIT VALUES ONLY.
# We confirmed they are NOT used by the detector.
# ============================================================

ABS = (
    WORK
    / "motion_abs"
    / "run_motion20_abs.py"
)

text = ABS.read_text()

abs_cases = []

for r in selected:
    abs_cases.append({
        "seed": r["seed"],
        "vid": r["vid"],

        # Dummy audit fields.
        "contact_gt":
            "UNLABELED_FRESH",

        "human_start":
            123,

        "human_end":
            124,

        # Exact PAI target phrase.
        "phrase":
            r["phrase"],
    })

text = replace_assignment(
    text,
    "CASES",
    "CASES = " + repr(abs_cases),
)

ABS.write_text(text)


# ============================================================
# PATCH LOCAL
# ============================================================

LOCAL = (
    WORK
    / "motion_local"
    / "run_motion20_local.py"
)

text = LOCAL.read_text()

local_cases = []

for r in selected:
    local_cases.append((
        r["seed"],
        r["vid"],

        "UNLABELED_FRESH",

        # Dummy audit GT only.
        123,
        124,

        r["phrase"],
    ))

text = replace_assignment(
    text,
    "CASES",
    "CASES = " + repr(local_cases),
)

LOCAL.write_text(text)


# ============================================================
# ROBOTSEG
# ============================================================

ROBOTSEG_SRC = (
    PHYS
    / "RobotSeg"
    / "test"
    / "run_unified_ee_dev9.py"
)

ROBOTSEG_DST = (
    PHYS
    / "RobotSeg"
    / "test"
    / "run_unified_ee_pipeline20.py"
)

text = ROBOTSEG_SRC.read_text()

entries = []

for r in selected:

    entries.append(
        "    "
        + repr(r["case"])
        + ": Path("
        + repr(r["video"])
        + ")"
    )

robotseg_cases = (
    "CASES = {\n"
    +
    ",\n".join(entries)
    +
    "\n}"
)

text = replace_assignment(
    text,
    "CASES",
    robotseg_cases,
)

text = text.replace(
    "robotseg_unified_ee_dev9",
    "pipeline20_e2e_v1/robotseg_ee",
)

text = text.replace(
    "robotseg_primary9_ee_v1",
    "pipeline20_e2e_v1/robotseg_ee",
)

ROBOTSEG_DST.write_text(text)


# ============================================================
# VDA
# ============================================================

VDA_SRC = (
    ROOT
    / "contact_depth_primary9_v1"
    / "run_vda_primary9.py"
)

VDA_DST = (
    ROOT
    / "contact_depth_primary9_v1"
    / "run_vda_pipeline20.py"
)

text = VDA_SRC.read_text()

vda_cases = [
    (
        r["seed"],
        r["vid"],
    )
    for r in selected
]

text = replace_assignment(
    text,
    "CASES",
    "CASES = "
    + repr(vda_cases),
)

text = text.replace(
    "contact_depth_primary9_v1",
    "pipeline20_e2e_v1/contact_depth",
)

VDA_DST.write_text(text)


# ============================================================
# COMPILE CHECK
# ============================================================

for p in [
    ABS,
    LOCAL,
    ROBOTSEG_DST,
    VDA_DST,
]:
    compile(
        p.read_text(),
        str(p),
        "exec",
    )

print()
print("=" * 100)
print("READY")
print("=" * 100)

print("Manifest:")
print(manifest_path)

print()
print("Preflight audit:")
print(audit_path)

print()
print("ABS:")
print(ABS)

print()
print("LOCAL:")
print(LOCAL)

print()
print("RobotSeg:")
print(ROBOTSEG_DST)

print()
print("VDA:")
print(VDA_DST)

print()
print("All scripts compile.")
