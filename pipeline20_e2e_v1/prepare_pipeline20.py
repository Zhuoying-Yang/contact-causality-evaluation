#!/usr/bin/env python3

from pathlib import Path
import ast
import csv
import re
import sys

PHYS = Path("/shared/ssd_30T/zhuoyingyang/physact")
ROOT = PHYS / "sam3_robowm"
WORK = ROOT / "pipeline20_e2e_v1"
WORK.mkdir(parents=True, exist_ok=True)

CASES = [
    {"seed":"seed101","vid":"0007","target":"white cube"},
    {"seed":"seed101","vid":"0012","target":"white cup"},
    {"seed":"seed101","vid":"0024","target":"yellow cube"},
    {"seed":"seed101","vid":"0026","target":"white tape roll"},
    {"seed":"seed101","vid":"0038","target":"white tape roll"},
    {"seed":"seed101","vid":"0061","target":"white cup"},
    {"seed":"seed101","vid":"0062","target":"brown paper cup"},

    {"seed":"seed102","vid":"0004","target":"yellow cube"},
    {"seed":"seed102","vid":"0012","target":"white cup"},
    {"seed":"seed102","vid":"0021","target":"brown paper cup"},
    {"seed":"seed102","vid":"0024","target":"yellow cube"},
    {"seed":"seed102","vid":"0029","target":"brown cube"},
    {"seed":"seed102","vid":"0038","target":"white tape roll"},
    {"seed":"seed102","vid":"0059","target":"white cup"},

    {"seed":"seed103","vid":"0003","target":"yellow cube"},
    {"seed":"seed103","vid":"0025","target":"yellow cube"},
    {"seed":"seed103","vid":"0033","target":"yellow cube"},
    {"seed":"seed103","vid":"0059","target":"white cup"},
    {"seed":"seed103","vid":"0062","target":"brown paper cup"},
    {"seed":"seed103","vid":"0063","target":"brown paper cup"},
]

# ------------------------------------------------------------
# manifest
# ------------------------------------------------------------

with open(WORK / "PIPELINE20_MANIFEST.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["case", "seed", "vid", "target"])
    w.writeheader()

    for c in CASES:
        w.writerow({
            "case": f"Cosmos3_{c['seed']}_{c['vid']}",
            **c,
        })


KNOWN_TARGETS = {
    "yellow cube",
    "blue cube",
    "white cube",
    "brown cube",
    "white cup",
    "brown paper cup",
    "white tape roll",
    "yellow banana",
    "banana",
}


def find_case_assignment(tree, text):
    """
    Find a literal list such as CASES / PRIMARY9 / DEV9_CASES
    that contains seed/video specs.
    """
    candidates = []

    for node in ast.walk(tree):
        value = None
        name = None

        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            if isinstance(node.targets[0], ast.Name):
                name = node.targets[0].id
                value = node.value

        elif isinstance(node, ast.AnnAssign):
            if isinstance(node.target, ast.Name):
                name = node.target.id
                value = node.value

        if name is None or value is None:
            continue

        if not isinstance(value, (ast.List, ast.Tuple)):
            continue

        try:
            obj = ast.literal_eval(value)
        except Exception:
            continue

        s = repr(obj)

        # Must look like a Cosmos/seed case list.
        if not (
            re.search(r"seed10[123]", s)
            and re.search(r"\b00[0-6]\d\b", s)
        ):
            continue

        score = 0
        uname = name.upper()

        if "CASE" in uname:
            score += 10
        if uname == "CASES":
            score += 20
        if "PRIMARY" in uname or "DEV9" in uname:
            score += 5

        candidates.append((score, node, name, obj))

    if not candidates:
        raise RuntimeError("Could not find literal case-list assignment.")

    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0]


def find_seed_vid(obj):
    seed = None
    vid = None

    def rec(x):
        nonlocal seed, vid

        if isinstance(x, str):
            if re.fullmatch(r"seed10[123]", x):
                seed = x

            if re.fullmatch(r"00[0-6]\d", x):
                vid = x

            m = re.search(r"(seed10[123]).*?(00[0-6]\d)", x)
            if m:
                seed = seed or m.group(1)
                vid = vid or m.group(2)

        elif isinstance(x, dict):
            for k, v in x.items():
                rec(k)
                rec(v)

        elif isinstance(x, (list, tuple)):
            for v in x:
                rec(v)

    rec(obj)
    return seed, vid


def adapt_template(obj, new_case, old_seed, old_vid):
    """
    Preserve every old audit/config field except:
      seed
      video id
      case-name/path strings
      object semantic phrase

    IMPORTANT:
    old GT fields are intentionally left untouched because these
    scripts may require them structurally. We will IGNORE all GT/error
    columns from this fresh run. The detector itself must not depend
    on human GT.
    """

    if isinstance(obj, str):
        s = obj

        if s == old_seed:
            return new_case["seed"]

        if s == old_vid:
            return new_case["vid"]

        if s.lower() in KNOWN_TARGETS:
            return new_case["target"]

        # Replace embedded old case strings / paths.
        if old_seed:
            s = s.replace(old_seed, new_case["seed"])

        if old_vid:
            s = s.replace(old_vid, new_case["vid"])

        # If a semantic target occurs as one complete config string.
        if obj.lower() in KNOWN_TARGETS:
            s = new_case["target"]

        return s

    if isinstance(obj, tuple):
        return tuple(
            adapt_template(v, new_case, old_seed, old_vid)
            for v in obj
        )

    if isinstance(obj, list):
        return [
            adapt_template(v, new_case, old_seed, old_vid)
            for v in obj
        ]

    if isinstance(obj, dict):
        out = {}

        for k, v in obj.items():
            nk = adapt_template(k, new_case, old_seed, old_vid)
            nv = adapt_template(v, new_case, old_seed, old_vid)

            # Dict schemas often make role detection easier.
            kl = str(k).lower()

            if "seed" in kl:
                nv = new_case["seed"]

            elif (
                kl in {"vid", "video_id", "video", "id"}
                or "video_id" in kl
            ):
                if isinstance(v, str):
                    nv = new_case["vid"]

            elif (
                "phrase" in kl
                or "target" in kl
                or "object" in kl
            ):
                if isinstance(v, str):
                    nv = new_case["target"]

            out[nk] = nv

        return out

    return obj


def replace_node_source(text, node, replacement):
    lines = text.splitlines(keepends=True)

    # ast lines are 1-indexed.
    start_line = node.lineno - 1
    end_line = node.end_lineno - 1

    prefix = lines[start_line][:node.col_offset]
    suffix = lines[end_line][node.end_col_offset:]

    new_block = prefix + replacement + suffix

    return (
        "".join(lines[:start_line])
        + new_block
        + "".join(lines[end_line + 1:])
    )


def patch_script(src, dst, replacements):
    src = Path(src)
    dst = Path(dst)

    if not src.exists():
        raise FileNotFoundError(src)

    text = src.read_text()
    tree = ast.parse(text)

    score, node, name, original_cases = find_case_assignment(tree, text)

    if not original_cases:
        raise RuntimeError(f"{src}: empty case list")

    template = original_cases[0]

    old_seed, old_vid = find_seed_vid(template)

    if old_seed is None or old_vid is None:
        # Search all original entries until one exposes both.
        for t in original_cases:
            old_seed, old_vid = find_seed_vid(t)
            if old_seed and old_vid:
                template = t
                break

    if old_seed is None or old_vid is None:
        raise RuntimeError(
            f"{src}: could not infer seed/id from template:\n{template}"
        )

    new_cases = [
        adapt_template(
            template,
            c,
            old_seed=old_seed,
            old_vid=old_vid,
        )
        for c in CASES
    ]

    replacement_literal = (
        f"{name} = "
        + repr(new_cases)
    )

    # Replace the full assignment, not only list value.
    assignment_text = replacement_literal

    # Our helper currently replaces whole node while preserving indentation.
    text = replace_node_source(
        text,
        node,
        assignment_text,
    )

    # Paths / project names.
    for old, new in replacements.items():
        text = text.replace(old, new)

    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(text)

    print()
    print("=" * 80)
    print("PATCHED")
    print("source :", src)
    print("dest   :", dst)
    print("list   :", name)
    print("template:", template)
    print("cases  :", len(new_cases))


# ============================================================
# SOURCE SCRIPTS
# ============================================================

ABS_DIR = ROOT / "motion_primary9_audit_v1"

abs_candidates = []

for p in ABS_DIR.glob("*.py"):
    try:
        txt = p.read_text(errors="ignore")
    except Exception:
        continue

    if "CoTrackerPredictor" in txt and "CASE" in txt:
        abs_candidates.append(p)

if not abs_candidates:
    raise RuntimeError(
        f"Cannot identify absolute-motion script in {ABS_DIR}"
    )

ABS_SRC = sorted(abs_candidates)[0]

LOCAL_SRC = (
    ROOT
    / "motion_primary9_localref_audit_v1"
    / "run_motion_primary9_localref_audit.py"
)

HYBRID_SRC = (
    ROOT
    / "motion_primary9_hybrid_veto_v1"
    / "run_hybrid_veto_primary9.py"
)

ROBOTSEG_SRC = (
    PHYS
    / "RobotSeg"
    / "test"
    / "run_unified_ee_dev9.py"
)

VDA_SRC = (
    ROOT
    / "contact_depth_primary9_v1"
    / "run_vda_primary9.py"
)


# ============================================================
# PATCH MOTION SCRIPTS
# ============================================================

patch_script(
    ABS_SRC,
    WORK / "motion_abs" / "run_motion20_abs.py",
    {
        "motion_primary9_audit_v1":
            "pipeline20_e2e_v1/motion_abs",
    },
)

patch_script(
    LOCAL_SRC,
    WORK / "motion_local" / "run_motion20_local.py",
    {
        "motion_primary9_audit_v1":
            "pipeline20_e2e_v1/motion_abs",

        "motion_primary9_localref_audit_v1":
            "pipeline20_e2e_v1/motion_local",
    },
)

# ============================================================
# PATCH HYBRID
#
# The hybrid script does NOT contain its own CASES list.
# It simply combines the ABS and LOCAL detector outputs.
# Therefore only redirect its input/output paths.
# ============================================================

HYBRID_DST = (
    WORK
    / "motion_hybrid"
    / "run_motion20_hybrid.py"
)

HYBRID_DST.parent.mkdir(
    parents=True,
    exist_ok=True,
)

hybrid_text = HYBRID_SRC.read_text()

hybrid_text = hybrid_text.replace(
    "motion_primary9_audit_v1",
    "pipeline20_e2e_v1/motion_abs",
)

hybrid_text = hybrid_text.replace(
    "motion_primary9_localref_audit_v1",
    "pipeline20_e2e_v1/motion_local",
)

hybrid_text = hybrid_text.replace(
    "motion_primary9_hybrid_veto_v1",
    "pipeline20_e2e_v1/motion_hybrid",
)

HYBRID_DST.write_text(hybrid_text)

print()
print("=" * 80)
print("PATCHED HYBRID PATHS ONLY")
print("source :", HYBRID_SRC)
print("dest   :", HYBRID_DST)
print("cases  : inherited from ABS/LOCAL outputs")

# ============================================================
# PATCH ROBOTSEG
#
# Write next to original script so any __file__-relative imports
# continue to work.
# ============================================================

patch_script(
    ROBOTSEG_SRC,
    PHYS / "RobotSeg" / "test" / "run_unified_ee_pipeline20.py",
    {
        "robotseg_unified_ee_dev9":
            "pipeline20_e2e_v1/robotseg_ee",

        "robotseg_primary9_ee_v1":
            "pipeline20_e2e_v1/robotseg_ee",
    },
)


# ============================================================
# PATCH VDA
# ============================================================

patch_script(
    VDA_SRC,
    ROOT
    / "contact_depth_primary9_v1"
    / "run_vda_pipeline20.py",
    {
        "contact_depth_primary9_v1":
            "pipeline20_e2e_v1/contact_depth",
    },
)


print()
print("=" * 80)
print("PIPELINE20 PREPARATION COMPLETE")
print("=" * 80)

print("Manifest:")
print(WORK / "PIPELINE20_MANIFEST.csv")

print()
print("Absolute motion:")
print(WORK / "motion_abs" / "run_motion20_abs.py")

print()
print("Local-reference motion:")
print(WORK / "motion_local" / "run_motion20_local.py")

print()
print("Hybrid motion:")
print(WORK / "motion_hybrid" / "run_motion20_hybrid.py")

print()
print("RobotSeg:")
print(PHYS / "RobotSeg" / "test" / "run_unified_ee_pipeline20.py")

print()
print("VDA:")
print(
    ROOT
    / "contact_depth_primary9_v1"
    / "run_vda_pipeline20.py"
)
