#!/usr/bin/env python3

from pathlib import Path
import ast

PHYS = Path("/shared/ssd_30T/zhuoyingyang/physact")
ROOT = PHYS / "sam3_robowm"
WORK = ROOT / "pipeline20_e2e_v1"

CASES = [
    ("seed101","0007","white cube"),
    ("seed101","0012","white cup"),
    ("seed101","0024","yellow cube"),
    ("seed101","0026","white tape roll"),
    ("seed101","0038","white tape roll"),
    ("seed101","0061","white cup"),
    ("seed101","0062","brown paper cup"),

    ("seed102","0004","yellow cube"),
    ("seed102","0012","white cup"),
    ("seed102","0021","brown paper cup"),
    ("seed102","0024","yellow cube"),
    ("seed102","0029","brown cube"),
    ("seed102","0038","white tape roll"),
    ("seed102","0059","white cup"),

    ("seed103","0003","yellow cube"),
    ("seed103","0025","yellow cube"),
    ("seed103","0033","yellow cube"),
    ("seed103","0059","white cup"),
    ("seed103","0062","brown paper cup"),
    ("seed103","0063","brown paper cup"),
]

VIDEO_ROOT = (
    PHYS
    / "cosmos3"
    / "export_robowm68_3seeds"
)


def replace_assignment(text, variable_name, replacement):
    tree = ast.parse(text)

    target_node = None

    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if (
                    isinstance(target, ast.Name)
                    and target.id == variable_name
                ):
                    target_node = node
                    break

        elif isinstance(node, ast.AnnAssign):
            if (
                isinstance(node.target, ast.Name)
                and node.target.id == variable_name
            ):
                target_node = node

        if target_node is not None:
            break

    if target_node is None:
        raise RuntimeError(
            f"Could not find assignment: {variable_name}"
        )

    lines = text.splitlines(keepends=True)

    start = target_node.lineno - 1
    end = target_node.end_lineno - 1

    prefix = lines[start][:target_node.col_offset]
    suffix = lines[end][target_node.end_col_offset:]

    new_block = (
        prefix
        + replacement
        + suffix
    )

    return (
        "".join(lines[:start])
        + new_block
        + "".join(lines[end + 1:])
    )


# ============================================================
# ROBOTSEG
# ============================================================

robotseg_src = (
    PHYS
    / "RobotSeg"
    / "test"
    / "run_unified_ee_dev9.py"
)

robotseg_dst = (
    PHYS
    / "RobotSeg"
    / "test"
    / "run_unified_ee_pipeline20.py"
)

text = robotseg_src.read_text()

entries = []

for seed, vid, phrase in CASES:
    name = f"Cosmos3_{seed}_{vid}"

    video = (
        VIDEO_ROOT
        / seed
        / f"{vid}.mp4"
    )

    entries.append(
        f'    "{name}": Path("{video}")'
    )

cases_code = (
    "CASES = {\n"
    + ",\n".join(entries)
    + "\n}"
)

text = replace_assignment(
    text,
    "CASES",
    cases_code,
)

# Redirect output directory.
text = text.replace(
    'robotseg_unified_ee_dev9',
    'pipeline20_e2e_v1/robotseg_ee',
)

text = text.replace(
    'robotseg_primary9_ee_v1',
    'pipeline20_e2e_v1/robotseg_ee',
)

robotseg_dst.write_text(text)

print("=" * 80)
print("ROBOTSEG PATCHED")
print("source:", robotseg_src)
print("dest  :", robotseg_dst)
print("cases :", len(CASES))


# ============================================================
# VDA
# ============================================================

vda_src = (
    ROOT
    / "contact_depth_primary9_v1"
    / "run_vda_primary9.py"
)

vda_dst = (
    ROOT
    / "contact_depth_primary9_v1"
    / "run_vda_pipeline20.py"
)

text = vda_src.read_text()

vda_cases = [
    (seed, vid)
    for seed, vid, _ in CASES
]

cases_code = (
    "CASES = "
    + repr(vda_cases)
)

text = replace_assignment(
    text,
    "CASES",
    cases_code,
)

text = text.replace(
    'contact_depth_primary9_v1',
    'pipeline20_e2e_v1/contact_depth',
)

vda_dst.write_text(text)

print()
print("=" * 80)
print("VDA PATCHED")
print("source:", vda_src)
print("dest  :", vda_dst)
print("cases :", len(CASES))


# ============================================================
# SANITY CHECK PATHS
# ============================================================

print()
print("=" * 80)
print("VIDEO EXISTENCE CHECK")
print("=" * 80)

missing = []

for seed, vid, phrase in CASES:
    video = (
        VIDEO_ROOT
        / seed
        / f"{vid}.mp4"
    )

    ok = video.exists()

    print(
        f"{seed}_{vid:4s} "
        f"{phrase:18s} "
        f"{'OK' if ok else 'MISSING'}"
    )

    if not ok:
        missing.append(str(video))


if missing:
    print()
    print("WARNING: missing videos:")
    for p in missing:
        print(p)
else:
    print()
    print("All 20 videos exist.")

print()
print("Done.")
