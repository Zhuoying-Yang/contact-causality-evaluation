from pathlib import Path
import pandas as pd

ROOT = Path(
    "/shared/ssd_30T/zhuoyingyang/physact/sam3_robowm"
)

CAND = (
    ROOT
    / "vlm_contact_oracle9"
    / "whole_video_contact_event_v2"
    / "CONTACT_EVENT_CANDIDATES_V2.csv"
)

ORACLE = (
    ROOT
    / "vlm_contact_oracle9"
    / "fresh20_contact_score_frozen_v6"
    / "FRESH20_CONTACT_SCORE_FROZEN_V6.csv"
)

AUDIT = (
    ROOT
    / "pipeline20_e2e_v1"
    / "FRESH20_HUMAN_ONSET_AUDIT_V2.csv"
)

cand = pd.read_csv(CAND)
oracle = pd.read_csv(ORACLE)
audit = pd.read_csv(AUDIT)

cand = cand[cand["split"] == "FRESH20"].copy()

oracle_map = dict(
    zip(
        oracle["case"],
        oracle["score"],
    )
)

rows = []

for _, a in audit.iterrows():

    if not bool(a["human_onset_scorable"]):
        continue

    case = a["case"]

    if pd.isna(a["human_response_lo"]):
        continue

    gt = int(a["human_response_lo"])

    q = cand[cand["case"] == case].copy()

    if len(q) == 0:
        continue

    q["dist_to_human"] = (
        q["frame"].astype(int) - gt
    ).abs()

    near = q.sort_values(
        ["dist_to_human", "frame"]
    ).iloc[0]

    best = q.loc[
        q["contact_event_score"].idxmax()
    ]

    rows.append({
        "case": case,
        "human_onset": gt,

        "nearest_candidate":
            int(near["frame"]),

        "nearest_delta":
            int(near["frame"]) - gt,

        "nearest_event_score":
            float(
                near["contact_event_score"]
            ),

        "best_event_frame":
            int(best["frame"]),

        "best_event_score":
            float(
                best["contact_event_score"]
            ),

        "oracle_v6_score":
            oracle_map.get(
                case,
                float("nan"),
            ),
    })

out = pd.DataFrame(rows)

print("=" * 110)
print("CONTACT-EVENT V2 DIAGNOSTIC")
print("=" * 110)

print(
    out.to_string(
        index=False,
        float_format=lambda x: f"{x:.2f}",
    )
)

print()
print(
    "Nearest candidate within ±2:",
    int((out["nearest_delta"].abs() <= 2).sum()),
    "/",
    len(out),
)

print(
    "Nearest candidate within ±5:",
    int((out["nearest_delta"].abs() <= 5).sum()),
    "/",
    len(out),
)

out.to_csv(
    ROOT
    / "vlm_contact_oracle9"
    / "whole_video_contact_event_v2"
    / "CONTACT_EVENT_V2_DIAGNOSTIC.csv",
    index=False,
)
