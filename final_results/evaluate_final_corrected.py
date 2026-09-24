from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

FILES = {
    "Original Backward":
        ROOT/"vlm_contact_oracle9/backward36_all38_v1/BACKWARD36_ALL38_CASES.csv",
    "MultiEpisode 1.0x":
        ROOT/"vlm_contact_oracle9/backward36_multi_episode_v1/BACKWARD36_ALL38_CASES.csv",
    "Strong 1.5x":
        ROOT/"vlm_contact_oracle9/backward36_strong15_v1/BACKWARD36_ALL38_CASES.csv",
    "Strong 2.0x":
        ROOT/"vlm_contact_oracle9/backward36_strong20_v1/BACKWARD36_ALL38_CASES.csv",
}

CORRECTIONS = {
    "Cosmos3_seed101_0035": (0, "NO_CONTACT"),
    "Cosmos3_seed103_0037": (1, "CONTACT"),
}

def fix_gt(df):
    df = df.copy()
    for case, (label, gt) in CORRECTIONS.items():
        df.loc[df["case"] == case, "label"] = label
        df.loc[df["case"] == case, "gt"] = gt
    return df

def auc(df, score_col):
    x = df[
        df["label"].notna() &
        df[score_col].notna()
    ].copy()

    y = x["label"].astype(int).to_numpy()
    s = x[score_col].astype(float).to_numpy()

    pos = s[y == 1]
    neg = s[y == 0]

    wins = 0.0
    total = 0

    for p in pos:
        for n in neg:
            total += 1
            if p > n:
                wins += 1
            elif p == n:
                wins += 0.5

    return len(x), len(pos), len(neg), wins / total

dfs = {
    name: fix_gt(pd.read_csv(path))
    for name, path in FILES.items()
}

master = dfs["Strong 1.5x"].copy()

resolved = master[master["label"].notna()].copy()

rows = []

# ------------------------------------------------------------
# FuturePair36 — important evaluation subsets
# ------------------------------------------------------------

subsets = {
    "FuturePair36 — Primary9":
        resolved[
            (resolved["source"] == "OLD29") &
            (resolved["split"] == "PRIMARY9")
        ],

    "FuturePair36 — Primary9 + Fresh20":
        resolved[
            (resolved["source"] == "OLD29") &
            (resolved["split"].isin(["PRIMARY9", "FRESH20"]))
        ],

    "FuturePair36 — Primary9 + Expanded":
        resolved[
            (
                (resolved["source"] == "OLD29") &
                (resolved["split"] == "PRIMARY9")
            )
            |
            resolved["source"].isin(["PILOT5", "COSMOS25_4"])
        ],

    "FuturePair36 — Expanded only":
        resolved[
            resolved["source"].isin(["PILOT5", "COSMOS25_4"])
        ],

    "FuturePair36 — Full expanded resolved":
        resolved,
}

for name, d in subsets.items():
    n, pos, neg, score = auc(d, "future_pair_score_36")
    rows.append({
        "evaluation": name,
        "coverage": n,
        "contact": pos,
        "no_contact": neg,
        "auroc": score,
    })

# ------------------------------------------------------------
# Backward methods — own coverage
# ------------------------------------------------------------

for name, d in dfs.items():
    d = d[d["label"].notna()].copy()
    n, pos, neg, score = auc(d, "backward36_score")

    rows.append({
        "evaluation": name + " — own coverage",
        "coverage": n,
        "contact": pos,
        "no_contact": neg,
        "auroc": score,
    })

# ------------------------------------------------------------
# Common support of all backward methods
# ------------------------------------------------------------

common = None

for d in dfs.values():
    scored = set(
        d[
            d["label"].notna() &
            d["backward36_score"].notna()
        ]["case"]
    )
    common = scored if common is None else common & scored

for name, d in dfs.items():
    q = d[
        d["case"].isin(common) &
        d["label"].notna()
    ].copy()

    n, pos, neg, score = auc(q, "backward36_score")

    rows.append({
        "evaluation": name + " — common25",
        "coverage": n,
        "contact": pos,
        "no_contact": neg,
        "auroc": score,
    })

q = master[
    master["case"].isin(common) &
    master["label"].notna()
].copy()

n, pos, neg, score = auc(q, "future_pair_score_36")

rows.append({
    "evaluation": "FuturePair36 — common25",
    "coverage": n,
    "contact": pos,
    "no_contact": neg,
    "auroc": score,
})

# ------------------------------------------------------------
# Local causal pre/post ablation
# ------------------------------------------------------------

local_path = (
    ROOT /
    "vlm_contact_oracle9/local_causal_pair_v1/"
    "LOCAL_CAUSAL_VIDEO_SCORES.csv"
)

local = pd.read_csv(local_path)

for case, (label, gt) in CORRECTIONS.items():
    local.loc[local["case"] == case, "label"] = label
    local.loc[local["case"] == case, "gt"] = gt

n, pos, neg, score = auc(local, "causal_event_score")

rows.append({
    "evaluation": "Local causal pre/post — full resolved",
    "coverage": n,
    "contact": pos,
    "no_contact": neg,
    "auroc": score,
})

# ------------------------------------------------------------
# Save
# ------------------------------------------------------------

out = pd.DataFrame(rows)

out.to_csv(
    ROOT/"final_results/CONTACT_FINAL_METRICS.csv",
    index=False
)

pd.DataFrame([
    {
        "case": case,
        "corrected_label": label,
        "corrected_gt": gt,
    }
    for case, (label, gt) in CORRECTIONS.items()
]).to_csv(
    ROOT/"final_results/FINAL_GT_CORRECTIONS.csv",
    index=False
)

print(out.to_string(index=False))
