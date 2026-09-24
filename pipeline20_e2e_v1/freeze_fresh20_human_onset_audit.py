from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(
    "/shared/ssd_30T/zhuoyingyang/physact/"
    "sam3_robowm/pipeline20_e2e_v1"
)

RAW = ROOT / "PREDICTED_ONSET_CONTACT_RAW.csv"
OUT = ROOT / "FRESH20_HUMAN_ONSET_AUDIT_V2.csv"

# ------------------------------------------------------------
# Human annotations from visual review.
#
# lo/hi = uncertainty interval for FIRST sustained object
# response. Translation / rotation / deformation all count.
#
# score=False means human onset itself is too ambiguous to use
# in onset accuracy.
# ------------------------------------------------------------

ANN = {

    "Cosmos3_seed101_0004": {
        "lo": 118,
        "hi": 118,
        "score": True,
        "human_note":
            "Normal contact; confirmed first meaningful response/contact-related "
            "motion at frame 118. Earlier 52-58 estimate ignored.",
    },

    "Cosmos3_seed101_0014": {
        "lo": 72,
        "hi": 73,
        "score": True,
        "human_note":
            "Penetration; response around 72-73.",
    },

    "Cosmos3_seed101_0015": {
        "lo": 127,
        "hi": 129,
        "score": True,
        "human_note":
            "Normal contact; deformation; response 127-129.",
    },

    "Cosmos3_seed101_0029": {
        "lo": 128,
        "hi": 130,
        "score": True,
        "human_note":
            "Normal contact; deformation; rigid body breaks into parts; "
            "response 128-130.",
    },

    "Cosmos3_seed101_0034": {
        "lo": 129,
        "hi": 130,
        "score": True,
        "human_note":
            "Normal contact; deformation; response 129-130.",
    },

    "Cosmos3_seed101_0040": {
        "lo": np.nan,
        "hi": np.nan,
        "score": False,
        "human_note":
            "Normal contact; almost no visible object motion.",
    },

    "Cosmos3_seed101_0042": {
        "lo": 88,
        "hi": 88,
        "score": True,
        "human_note":
            "Normal contact; response around 88; rigid body breaks; "
            "upper part moves while lower part stays.",
    },

    "Cosmos3_seed102_0015": {
        # Definition includes deformation.
        # Human noted deformation begins at 165,
        # clearer motion around 169-170.
        "lo": 165,
        "hi": 170,
        "score": True,
        "human_note":
            "Normal contact; first deformation about 165; "
            "clear response 169-170.",
    },

    "Cosmos3_seed102_0019": {
        "lo": 96,
        "hi": 97,
        "score": True,
        "human_note":
            "Normal contact; deformation; response 96-97.",
    },

    "Cosmos3_seed102_0025": {
        "lo": 105,
        "hi": 107,
        "score": True,
        "human_note":
            "Normal contact; response 105-107.",
    },

    "Cosmos3_seed102_0027": {
        "lo": np.nan,
        "hi": np.nan,
        "score": False,
        "human_note":
            "Duplicate target/object; original one almost does not move. "
            "ABSTAIN for onset evaluation.",
    },

    # IMPORTANT:
    # User wrote seed102_0009 = no-contact motion around 105.
    # Fresh20 contains seed102_0029, not seed102_0009.
    # Therefore do NOT silently transfer that annotation here.
    "Cosmos3_seed102_0029": {
        "lo": 105,
        "hi": 105,
        "score": True,
        "human_note":
            "NO_CONTACT_MOTION; first response around frame 105; "
            "rigid body breaks into parts.",
    },

    "Cosmos3_seed102_0036": {
        "lo": 82,
        "hi": 87,
        "score": True,
        "human_note":
            "NO-CONTACT motion begins 82-87; "
            "contact and coupled motion begin around 92-94.",
    },

    "Cosmos3_seed102_0042": {
        "lo": 106,
        "hi": 107,
        "score": True,
        "human_note":
            "Normal contact; rigid body breaks; "
            "upper part moves while lower part stays; response 106-107.",
    },

    "Cosmos3_seed103_0003": {
        "lo": 75,
        "hi": 76,
        "score": True,
        "human_note":
            "Normal contact or possibly <1-frame pre-contact motion; "
            "deformation; response 75-76.",
    },

    "Cosmos3_seed103_0004": {
        "lo": 55,
        "hi": 56,
        "score": True,
        "human_note":
            "Normal contact; slight deformation; response 55-56.",
    },

    "Cosmos3_seed103_0026": {
        "lo": 140,
        "hi": 140,
        "score": False,
        "human_note":
            "Normal contact; slight shake around 140; "
            "uncertain whether this should count as true response.",
    },

    "Cosmos3_seed103_0028": {
        "lo": 149,
        "hi": 151,
        "score": True,
        "human_note":
            "Normal contact; response 149-151; "
            "part of EE duplicates and drops.",
    },

    "Cosmos3_seed103_0035": {
        "lo": 188,
        "hi": 190,
        "score": True,
        "human_note":
            "Normal contact; response 188-190.",
    },

    "Cosmos3_seed103_0036": {
        "lo": 90,
        "hi": 93,
        "score": True,
        "human_note":
            "Normal contact; slight shake; response 90-93.",
    },
}


df = pd.read_csv(RAW)

rows = []

for _, r in df.iterrows():

    case = r["case"]

    if case not in ANN:
        raise RuntimeError(
            f"Missing human annotation: {case}"
        )

    a = ANN[case]

    pred = r["predicted_motion_onset"]

    lo = a["lo"]
    hi = a["hi"]

    score = bool(a["score"])

    if not score:

        onset_eval = "ABSTAIN"

        onset_error = np.nan

    elif pd.isna(pred):

        onset_eval = "MISSED_RESPONSE"

        onset_error = np.nan

    else:

        pred = float(pred)

        # Distance to human uncertainty interval.
        if pred < lo:
            onset_error = pred - lo

        elif pred > hi:
            onset_error = pred - hi

        else:
            onset_error = 0.0

        # Keep the same +/-2-frame interpretation
        # used previously. This is NOT newly tuned here.
        if pred < lo - 2:
            onset_eval = "EARLY"

        elif pred > hi + 2:
            onset_eval = "LATE"

        else:
            onset_eval = "WITHIN_2"

    row = dict(r)

    row.update({
        "human_response_lo": lo,
        "human_response_hi": hi,
        "human_onset_scorable": int(score),
        "onset_eval": onset_eval,
        "onset_error_to_interval": onset_error,
        "human_note": a["human_note"],
    })

    rows.append(row)


out = pd.DataFrame(rows)

out.to_csv(
    OUT,
    index=False,
)

cols = [
    "case",
    "predicted_motion_onset",
    "human_response_lo",
    "human_response_hi",
    "onset_eval",
    "onset_error_to_interval",
    "human_note",
]

print("=" * 120)
print("FRESH20 HUMAN ONSET AUDIT")
print("=" * 120)

print(
    out[cols].to_string(
        index=False,
        float_format=lambda x: f"{x:.1f}",
    )
)

print()
print("=" * 120)
print("SCORABLE CASE SUMMARY")
print("=" * 120)

sc = out[
    out["human_onset_scorable"] == 1
]

print(
    sc["onset_eval"]
    .value_counts()
    .to_string()
)

print()
print(
    "Scorable:",
    len(sc),
    "/",
    len(out),
)

print()
print("Saved:")
print(OUT)

print()
print("=" * 120)
print("UNMATCHED HUMAN NOTE")
print("=" * 120)
print(
    "seed102_0009: NO_CONTACT_MOTION around frame 105; "
    "rigid body breaks into parts."
)
print(
    "This was NOT assigned to seed102_0029 automatically."
)
