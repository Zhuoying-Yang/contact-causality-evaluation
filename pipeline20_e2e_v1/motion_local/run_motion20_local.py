from pathlib import Path
import pickle
import csv
import subprocess

import cv2
import numpy as np
import torch

from pycocotools import mask as mask_utils
from cotracker.predictor import CoTrackerPredictor


# ============================================================
# PATHS
# ============================================================

PHYS = Path(
    "/shared/ssd_30T/zhuoyingyang/physact"
)

ROOT = (
    PHYS
    / "sam3_robowm"
)

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

ROBOTSEG_ROOT = (
    ROOT
    / "pipeline20_e2e_v1/robotseg_ee"
)

OLD_ROOT = (
    ROOT
    / "pipeline20_e2e_v1/motion_abs"
)

OUT = (
    ROOT
    / "pipeline20_e2e_v1/motion_local"
)

OUT.mkdir(
    parents=True,
    exist_ok=True
)

CHECKPOINT = (
    PHYS
    / "ReVidgen"
    / "checkpoints"
    / "Cotracker"
    / "scaled_offline.pth"
)

FFMPEG = Path(
    "/data/home/zhuoyingyang/"
    "miniconda3/envs/cosmos25/bin/ffmpeg"
)


# ============================================================
# FROZEN HUMAN GT
#
# ONLY used for final audit.
# NEVER used to select seed / motion onset.
# ============================================================

CASES = [('seed101', '0004', 'UNLABELED_FRESH', 123, 124, 'yellow cube'), ('seed101', '0014', 'UNLABELED_FRESH', 123, 124, 'white cup'), ('seed101', '0015', 'UNLABELED_FRESH', 123, 124, 'white cup'), ('seed101', '0029', 'UNLABELED_FRESH', 123, 124, 'brown cube'), ('seed101', '0034', 'UNLABELED_FRESH', 123, 124, 'yellow cube'), ('seed101', '0040', 'UNLABELED_FRESH', 123, 124, 'brown cube'), ('seed101', '0042', 'UNLABELED_FRESH', 123, 124, 'brown cube'), ('seed102', '0015', 'UNLABELED_FRESH', 123, 124, 'white cup'), ('seed102', '0019', 'UNLABELED_FRESH', 123, 124, 'banana'), ('seed102', '0025', 'UNLABELED_FRESH', 123, 124, 'yellow cube'), ('seed102', '0027', 'UNLABELED_FRESH', 123, 124, 'white tape roll'), ('seed102', '0029', 'UNLABELED_FRESH', 123, 124, 'brown cube'), ('seed102', '0036', 'UNLABELED_FRESH', 123, 124, 'white tape roll'), ('seed102', '0042', 'UNLABELED_FRESH', 123, 124, 'brown cube'), ('seed103', '0003', 'UNLABELED_FRESH', 123, 124, 'yellow cube'), ('seed103', '0004', 'UNLABELED_FRESH', 123, 124, 'yellow cube'), ('seed103', '0026', 'UNLABELED_FRESH', 123, 124, 'white tape'), ('seed103', '0028', 'UNLABELED_FRESH', 123, 124, 'white tape roll'), ('seed103', '0035', 'UNLABELED_FRESH', 123, 124, 'yellow cube'), ('seed103', '0036', 'UNLABELED_FRESH', 123, 124, 'white tape roll')]


# ============================================================
# LOCAL REFERENCE PARAMETERS
#
# These are not fitted to the 9 outputs.
# ============================================================

LOCAL_K = 40

MIN_LOCAL_COMMON = 10

MIN_OBJECT_COMMON_FRACTION = 0.25

BASELINE_VALID_STEPS = 5

MAD_MULTIPLIER = 6.0

MIN_RIGID_SPEED_NORM = 0.002

MIN_POINT_RESPONSE_NORM = 0.002

POINT_RESPONSE_QUANTILE = 0.75

PERSIST_WINDOW = 3

PERSIST_REQUIRED = 2

# Avoid reference features directly on object / robot boundary.
EXCLUSION_DILATE_PX = 10

EPS = 1e-8


# ============================================================
# VIDEO
# ============================================================

def load_video(path):

    cap = cv2.VideoCapture(
        str(path)
    )

    if not cap.isOpened():

        raise RuntimeError(
            f"Cannot open {path}"
        )

    fps = float(
        cap.get(
            cv2.CAP_PROP_FPS
        )
    )

    frames = []

    while True:

        ok, frame = cap.read()

        if not ok:
            break

        frames.append(frame)

    cap.release()

    if not frames:

        raise RuntimeError(
            f"No frames: {path}"
        )

    return (
        np.stack(
            frames,
            axis=0
        ),
        fps,
    )


# ============================================================
# PAI OBJECT MASK
# ============================================================

def decode_3d_rle(container):

    T, H, W = [
        int(x)
        for x in container[
            "mask_shape"
        ]
    ]

    x = mask_utils.decode(
        container["data"]
    )

    if (
        x.ndim == 3
        and x.shape[-1] == 1
    ):

        x = x[..., 0]

    x = np.asarray(x)

    if x.shape == (T, H, W):

        pass

    elif x.shape == (T * H, W):

        x = x.reshape(
            T,
            H,
            W
        )

    elif x.shape == (H, T * W):

        x = (
            x.reshape(
                H,
                T,
                W
            )
            .transpose(
                1,
                0,
                2
            )
        )

    elif x.size == T * H * W:

        x = x.reshape(
            T,
            H,
            W
        )

    else:

        raise RuntimeError(
            f"Bad RLE shape: {x.shape}"
        )

    return x.astype(bool)


def load_object_masks(
    seed,
    vid,
    phrase,
):

    p = (
        PAI_ROOT
        / f"cosmos3_{seed}"
        / "sam"
        / f"{vid}.pkl"
    )

    with open(
        p,
        "rb"
    ) as f:

        items = pickle.load(f)

    matches = [
        x
        for x in items
        if x.get("phrase")
        == phrase
    ]

    if len(matches) != 1:

        raise RuntimeError(
            f"{seed}_{vid}: "
            f"{phrase}: "
            f"{len(matches)} matches"
        )

    return decode_3d_rle(
        matches[0][
            "segmentation_mask_rle"
        ]
    )


# ============================================================
# BASIC GEOMETRY
# ============================================================

def centroid(mask):

    ys, xs = np.nonzero(mask)

    if len(xs) == 0:

        return np.array(
            [np.nan, np.nan]
        )

    return np.array(
        [
            float(np.median(xs)),
            float(np.median(ys)),
        ]
    )


def bbox_diag(mask):

    ys, xs = np.nonzero(mask)

    if len(xs) == 0:

        return np.nan

    return float(
        np.hypot(
            xs.max() - xs.min() + 1,
            ys.max() - ys.min() + 1,
        )
    )


def dilate(mask, px):

    k = (
        2 * px + 1
    )

    kernel = (
        cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (k, k)
        )
    )

    return (
        cv2.dilate(
            mask.astype(np.uint8),
            kernel
        )
        > 0
    )


# ============================================================
# STATIC LOCAL REFERENCE POINTS
#
# 1. Exclude object
# 2. Exclude whole RobotSeg robot
# 3. Shi-Tomasi features
# 4. Sort by distance to object
# 5. Keep nearest 40
# ============================================================

def select_local_reference_points(
    frame,
    object_mask,
    robot_mask,
):

    H, W = object_mask.shape

    forbidden = (
        dilate(
            object_mask,
            EXCLUSION_DILATE_PX
        )
        |
        dilate(
            robot_mask,
            EXCLUSION_DILATE_PX
        )
    )

    allowed = (
        ~forbidden
    ).astype(np.uint8) * 255

    # Avoid borders.
    allowed[
        :10, :
    ] = 0

    allowed[
        -10:, :
    ] = 0

    allowed[
        :, :10
    ] = 0

    allowed[
        :, -10:
    ] = 0

    gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )

    corners = cv2.goodFeaturesToTrack(

        gray,

        maxCorners=800,

        qualityLevel=0.01,

        minDistance=8,

        mask=allowed,

        blockSize=7,

        useHarrisDetector=False,
    )

    if corners is None:

        raise RuntimeError(
            "No local reference features"
        )

    pts = (
        corners[:, 0, :]
        .astype(np.float32)
    )

    c = centroid(
        object_mask
    )

    dist = np.linalg.norm(
        pts - c[None, :],
        axis=1
    )

    order = np.argsort(
        dist
    )

    pts = pts[
        order
    ]

    pts = pts[
        :min(
            LOCAL_K,
            len(pts)
        )
    ]

    if len(pts) < MIN_LOCAL_COMMON:

        raise RuntimeError(
            f"Only {len(pts)} "
            f"reference points"
        )

    return pts


# ============================================================
# ROBUST MAD
# ============================================================

def robust_mad(x):

    x = np.asarray(
        x,
        dtype=float
    )

    med = float(
        np.median(x)
    )

    return float(
        np.median(
            np.abs(
                x - med
            )
        )
    )


# ============================================================
# SAME-ID RELATIVE MOTION
#
# object delta_i
# minus
# median local-reference delta
#
# This removes frame shake / camera translation.
# ============================================================

def compute_trace(
    object_tracks,
    object_vis,
    ref_tracks,
    ref_vis,
    seed_frame,
    object_diag,
):

    T = object_tracks.shape[0]

    N_obj = object_tracks.shape[1]

    rows = []

    for t in range(T):

        empty = {
            "frame": t,

            "object_common":
                0,

            "object_common_fraction":
                0.0,

            "reference_common":
                0,

            "ref_dx":
                np.nan,

            "ref_dy":
                np.nan,

            "ref_speed_norm":
                np.nan,

            "relative_rigid_norm":
                np.nan,

            "relative_q75_norm":
                np.nan,

            "relative_median_point_norm":
                np.nan,

            "relative_coherence":
                np.nan,
        }

        if t <= seed_frame:

            rows.append(
                empty
            )

            continue

        # ----------------------------------------------------
        # Object SAME IDs
        # ----------------------------------------------------

        obj_common = (
            object_vis[t - 1]
            &
            object_vis[t]
        )

        obj_idx = np.where(
            obj_common
        )[0]

        n_obj = len(
            obj_idx
        )

        obj_fraction = (
            n_obj
            /
            max(
                N_obj,
                1
            )
        )

        # ----------------------------------------------------
        # Reference SAME IDs
        # ----------------------------------------------------

        ref_common = (
            ref_vis[t - 1]
            &
            ref_vis[t]
        )

        ref_idx = np.where(
            ref_common
        )[0]

        n_ref = len(
            ref_idx
        )

        if (
            n_obj == 0
            or n_ref == 0
        ):

            empty[
                "object_common"
            ] = n_obj

            empty[
                "object_common_fraction"
            ] = obj_fraction

            empty[
                "reference_common"
            ] = n_ref

            rows.append(
                empty
            )

            continue

        object_delta = (
            object_tracks[
                t,
                obj_idx
            ]
            -
            object_tracks[
                t - 1,
                obj_idx
            ]
        )

        ref_delta = (
            ref_tracks[
                t,
                ref_idx
            ]
            -
            ref_tracks[
                t - 1,
                ref_idx
            ]
        )

        # Robust local scene translation.
        ref_vector = np.array(
            [
                float(
                    np.median(
                        ref_delta[:, 0]
                    )
                ),
                float(
                    np.median(
                        ref_delta[:, 1]
                    )
                ),
            ],
            dtype=float
        )

        ref_speed = float(
            np.linalg.norm(
                ref_vector
            )
        )

        # ----------------------------------------------------
        # Remove local scene motion from EVERY object point.
        # ----------------------------------------------------

        relative_delta = (
            object_delta
            -
            ref_vector[
                None, :
            ]
        )

        rigid_vector = np.array(
            [
                float(
                    np.median(
                        relative_delta[:, 0]
                    )
                ),
                float(
                    np.median(
                        relative_delta[:, 1]
                    )
                ),
            ],
            dtype=float
        )

        rigid_speed = float(
            np.linalg.norm(
                rigid_vector
            )
        )

        point_speed = np.linalg.norm(
            relative_delta,
            axis=1
        )

        median_point = float(
            np.median(
                point_speed
            )
        )

        q75 = float(
            np.quantile(
                point_speed,
                POINT_RESPONSE_QUANTILE
            )
        )

        coherence = float(
            rigid_speed
            /
            max(
                median_point,
                EPS
            )
        )

        coherence = min(
            coherence,
            1.0
        )

        rows.append({

            "frame":
                t,

            "object_common":
                n_obj,

            "object_common_fraction":
                obj_fraction,

            "reference_common":
                n_ref,

            "ref_dx":
                float(
                    ref_vector[0]
                ),

            "ref_dy":
                float(
                    ref_vector[1]
                ),

            "ref_speed_norm":
                ref_speed
                / object_diag,

            "relative_rigid_norm":
                rigid_speed
                / object_diag,

            "relative_q75_norm":
                q75
                / object_diag,

            "relative_median_point_norm":
                median_point
                / object_diag,

            "relative_coherence":
                coherence,
        })

    return rows


# ============================================================
# DETECT ONSET
#
# SAME thresholds as previous conceptual rule.
# No tuning from PRIMARY9.
# ============================================================

def detect_onset(
    rows,
    seed_frame,
):

    valid = [
        r
        for r in rows
        if (
            r["frame"]
            > seed_frame

            and

            r[
                "object_common_fraction"
            ]
            >=
            MIN_OBJECT_COMMON_FRACTION

            and

            r[
                "reference_common"
            ]
            >=
            MIN_LOCAL_COMMON

            and

            np.isfinite(
                r[
                    "relative_rigid_norm"
                ]
            )

            and

            np.isfinite(
                r[
                    "relative_q75_norm"
                ]
            )
        )
    ]

    baseline = valid[
        :BASELINE_VALID_STEPS
    ]

    if len(baseline) < 3:

        return (
            None,
            np.nan,
            np.nan,
        )

    rb = np.asarray(
        [
            r[
                "relative_rigid_norm"
            ]
            for r in baseline
        ]
    )

    pb = np.asarray(
        [
            r[
                "relative_q75_norm"
            ]
            for r in baseline
        ]
    )

    rigid_thr = max(

        MIN_RIGID_SPEED_NORM,

        float(
            np.median(rb)
        )
        +
        MAD_MULTIPLIER
        *
        robust_mad(rb)
    )

    point_thr = max(

        MIN_POINT_RESPONSE_NORM,

        float(
            np.median(pb)
        )
        +
        MAD_MULTIPLIER
        *
        robust_mad(pb)
    )

    T = len(rows)

    candidate = np.zeros(
        T,
        dtype=bool
    )

    for r in rows:

        t = r["frame"]

        valid_step = (

            r[
                "object_common_fraction"
            ]
            >=
            MIN_OBJECT_COMMON_FRACTION

            and

            r[
                "reference_common"
            ]
            >=
            MIN_LOCAL_COMMON

            and

            np.isfinite(
                r[
                    "relative_rigid_norm"
                ]
            )

            and

            np.isfinite(
                r[
                    "relative_q75_norm"
                ]
            )
        )

        if not valid_step:

            continue

        rigid_trigger = (
            r[
                "relative_rigid_norm"
            ]
            >
            rigid_thr
        )

        point_trigger = (
            r[
                "relative_q75_norm"
            ]
            >
            point_thr
        )

        candidate[t] = (
            rigid_trigger
            or
            point_trigger
        )

    auto = None

    for t in range(
        seed_frame + 1,
        T
    ):

        if not candidate[t]:

            continue

        stop = min(
            T,
            t + PERSIST_WINDOW
        )

        if (
            candidate[
                t:stop
            ].sum()
            >=
            PERSIST_REQUIRED
        ):

            auto = t

            break

    for r in rows:

        r[
            "rigid_threshold"
        ] = rigid_thr

        r[
            "point_threshold"
        ] = point_thr

        r[
            "motion_candidate"
        ] = int(
            candidate[
                r["frame"]
            ]
        )

    return (
        auto,
        rigid_thr,
        point_thr,
    )


# ============================================================
# CSV
# ============================================================

def save_trace(
    path,
    rows,
):

    fields = [

        "frame",

        "object_common",

        "object_common_fraction",

        "reference_common",

        "ref_dx",
        "ref_dy",

        "ref_speed_norm",

        "relative_rigid_norm",

        "relative_q75_norm",

        "relative_median_point_norm",

        "relative_coherence",

        "rigid_threshold",

        "point_threshold",

        "motion_candidate",
    ]

    with open(
        path,
        "w",
        newline=""
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fields
        )

        writer.writeheader()

        writer.writerows(
            rows
        )


# ============================================================
# SEED REVIEW
# ============================================================

def save_seed_review(
    frame,
    object_mask,
    robot_mask,
    object_points,
    reference_points,
    path,
):

    out = frame.copy()

    overlay = out.copy()

    overlay[
        object_mask
    ] = (
        0,
        255,
        0
    )

    overlay[
        robot_mask
    ] = (
        255,
        0,
        255
    )

    out = cv2.addWeighted(
        out,
        0.75,
        overlay,
        0.25,
        0
    )

    # Green = object points
    for x, y in object_points:

        cv2.circle(
            out,
            (
                int(round(x)),
                int(round(y))
            ),
            3,
            (0, 255, 0),
            -1
        )

    # Cyan = static reference points
    for x, y in reference_points:

        cv2.circle(
            out,
            (
                int(round(x)),
                int(round(y))
            ),
            3,
            (255, 255, 0),
            -1
        )

    cv2.putText(

        out,

        (
            "GREEN=object  "
            "CYAN=local static reference"
        ),

        (20, 40),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.75,

        (255, 255, 255),

        2,

        cv2.LINE_AA
    )

    cv2.imwrite(
        str(path),
        out
    )


# ============================================================
# REVIEW VIDEO
# ============================================================

def make_review_video(
    frames,
    fps,
    obj_tracks,
    obj_vis,
    ref_tracks,
    ref_vis,
    rows,
    human_start,
    human_end,
    old_auto,
    new_auto,
    case_out,
):

    T, H, W, _ = (
        frames.shape
    )

    relevant = [
        human_start,
        human_end,
    ]

    if old_auto is not None:

        relevant.append(
            old_auto
        )

    if new_auto is not None:

        relevant.append(
            new_auto
        )

    start = max(
        0,
        min(relevant) - 12
    )

    end = min(
        T - 1,
        max(relevant) + 12
    )

    temp = (
        case_out
        / "review_mp4v.mp4"
    )

    final = (
        case_out
        / "review_h264.mp4"
    )

    writer = cv2.VideoWriter(

        str(temp),

        cv2.VideoWriter_fourcc(
            *"mp4v"
        ),

        fps,

        (W, H)
    )

    by_frame = {
        int(r["frame"]):
            r
        for r in rows
    }

    for t in range(
        start,
        end + 1
    ):

        frame = (
            frames[t]
            .copy()
        )

        # Object = green.
        for i in np.where(
            obj_vis[t]
        )[0]:

            x, y = (
                obj_tracks[
                    t, i
                ]
            )

            cv2.circle(
                frame,
                (
                    int(round(x)),
                    int(round(y))
                ),
                3,
                (0, 255, 0),
                -1
            )

        # Reference = cyan.
        for i in np.where(
            ref_vis[t]
        )[0]:

            x, y = (
                ref_tracks[
                    t, i
                ]
            )

            cv2.circle(
                frame,
                (
                    int(round(x)),
                    int(round(y))
                ),
                3,
                (255, 255, 0),
                -1
            )

        # Human GT = yellow outer border.
        if (
            human_start
            <= t
            <= human_end
        ):

            cv2.rectangle(
                frame,
                (2, 2),
                (W - 3, H - 3),
                (0, 255, 255),
                6
            )

        # Old absolute onset = blue.
        if (
            old_auto is not None
            and t == old_auto
        ):

            cv2.rectangle(
                frame,
                (10, 10),
                (W - 11, H - 11),
                (255, 0, 0),
                5
            )

        # New relative onset = red.
        if (
            new_auto is not None
            and t == new_auto
        ):

            cv2.rectangle(
                frame,
                (18, 18),
                (W - 19, H - 19),
                (0, 0, 255),
                5
            )

        r = by_frame[t]

        cv2.rectangle(
            frame,
            (10, 10),
            (1190, 190),
            (0, 0, 0),
            -1
        )

        texts = [

            f"FRAME={t}",

            (
                f"HUMAN="
                f"{human_start}-{human_end}  "
                f"OLD_ABS={old_auto}  "
                f"NEW_LOCALREF={new_auto}"
            ),

            (
                f"ref_speed="
                f"{r['ref_speed_norm']:.5f}"
            ),

            (
                f"relative_rigid="
                f"{r['relative_rigid_norm']:.5f}  "
                f"thr="
                f"{r['rigid_threshold']:.5f}"
            ),

            (
                f"relative_q75="
                f"{r['relative_q75_norm']:.5f}  "
                f"thr="
                f"{r['point_threshold']:.5f}"
            ),

            (
                f"obj_common="
                f"{r['object_common_fraction']:.2f}  "
                f"ref_common="
                f"{r['reference_common']}  "
                f"candidate="
                f"{r['motion_candidate']}"
            ),
        ]

        y = 35

        for text in texts:

            cv2.putText(
                frame,
                text,
                (20, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.62,
                (255, 255, 255),
                2,
                cv2.LINE_AA
            )

            y += 27

        writer.write(
            frame
        )

    writer.release()

    if FFMPEG.exists():

        cmd = [

            str(FFMPEG),

            "-y",

            "-i",
            str(temp),

            "-c:v",
            "libx264",

            "-pix_fmt",
            "yuv420p",

            "-movflags",
            "+faststart",

            "-an",

            str(final),
        ]

        p = subprocess.run(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )

        if (
            p.returncode == 0
            and final.exists()
        ):

            temp.unlink(
                missing_ok=True
            )

            return final

    return temp


# ============================================================
# OLD AUDIT
# ============================================================

old_auto_map = {}

old_csv = (
    OLD_ROOT
    / "MOTION_PRIMARY9_AUDIT.csv"
)

with open(
    old_csv,
    newline=""
) as f:

    for r in csv.DictReader(f):

        x = (
            r["auto_onset"]
        )

        old_auto_map[
            r["case"]
        ] = (
            int(x)
            if x != ""
            else None
        )


# ============================================================
# MODEL
# ============================================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print("=" * 110)
print("PRIMARY9 LOCAL-REFERENCE MOTION AUDIT")
print("=" * 110)

print(
    "Device:",
    device
)

print(
    "Loading CoTracker..."
)

model = CoTrackerPredictor(
    checkpoint=str(
        CHECKPOINT
    )
).to(device)

model.eval()

print(
    "CoTracker ready."
)


# ============================================================
# RUN
# ============================================================

summary = []


for idx, (
    seed,
    vid,
    contact_gt,
    human_start,
    human_end,
    phrase,
) in enumerate(
    CASES,
    start=1
):

    case = (
        f"Cosmos3_{seed}_{vid}"
    )

    print()
    print("=" * 110)

    print(
        f"[{idx}/{len(CASES)}]",
        case
    )

    print(
        "HUMAN:",
        human_start,
        "-",
        human_end
    )

    print("=" * 110)

    case_out = (
        OUT / case
    )

    case_out.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Existing object tracks
    # --------------------------------------------------------

    old_npz = np.load(

        OLD_ROOT
        / case
        / "cotracker_motion_tracks.npz"
    )

    obj_tracks = (
        old_npz["tracks"]
        .astype(np.float32)
    )

    obj_vis = (
        old_npz["visibility"]
        .astype(bool)
    )

    obj_points = (
        old_npz["seed_points"]
        .astype(np.float32)
    )

    seed_frame = int(
        old_npz[
            "seed_frame"
        ][0]
    )

    # --------------------------------------------------------
    # Video
    # --------------------------------------------------------

    video_path = (
        VIDEO_ROOT
        / seed
        / f"{vid}.mp4"
    )

    frames, fps = (
        load_video(
            video_path
        )
    )

    T, H, W, _ = (
        frames.shape
    )

    # --------------------------------------------------------
    # Object mask at same early seed
    # --------------------------------------------------------

    object_masks = (
        load_object_masks(
            seed,
            vid,
            phrase
        )
    )

    object_mask = (
        object_masks[
            seed_frame
        ]
    )

    object_diag = (
        bbox_diag(
            object_mask
        )
    )

    # --------------------------------------------------------
    # Whole robot mask at same seed
    # --------------------------------------------------------

    robot_path = (

        ROBOTSEG_ROOT
        / case
        / "robot_masks"
        / f"{seed_frame:06d}.png"
    )

    robot_u8 = cv2.imread(
        str(robot_path),
        cv2.IMREAD_GRAYSCALE
    )

    if robot_u8 is None:

        raise RuntimeError(
            f"Missing robot mask: "
            f"{robot_path}"
        )

    robot_mask = (
        robot_u8 > 0
    )

    # --------------------------------------------------------
    # Static local features
    # --------------------------------------------------------

    ref_points = (
        select_local_reference_points(

            frames[
                seed_frame
            ],

            object_mask,

            robot_mask,
        )
    )

    print(
        "seed:",
        seed_frame,
        "| object points:",
        len(obj_points),
        "| static reference:",
        len(ref_points)
    )

    save_seed_review(

        frames[
            seed_frame
        ],

        object_mask,

        robot_mask,

        obj_points,

        ref_points,

        case_out
        / "local_reference_seed_review.png"
    )


    # ========================================================
    # TRACK ONLY REFERENCE POINTS
    # ========================================================

    video_rgb = (
        frames[
            :, :, :, ::-1
        ].copy()
    )

    video_tensor = (
        torch.from_numpy(
            video_rgb
        )
        .permute(
            0,
            3,
            1,
            2
        )[None]
        .float()
    )

    query_time = np.full(

        (
            len(ref_points),
            1
        ),

        seed_frame,

        dtype=np.float32
    )

    queries_np = np.concatenate(
        [
            query_time,
            ref_points
        ],
        axis=1
    )

    queries = (
        torch.from_numpy(
            queries_np
        )[None]
        .float()
    )

    print(
        "Tracking local reference..."
    )

    with torch.inference_mode():

        (
            pred_tracks,
            pred_vis
        ) = model(

            video_tensor.to(
                device
            ),

            queries=queries.to(
                device
            )
        )

    ref_tracks = (
        pred_tracks[0]
        .detach()
        .cpu()
        .numpy()
    )

    ref_vis = (
        pred_vis[0]
        .detach()
        .cpu()
        .numpy()
        > 0.5
    )


    # ========================================================
    # RELATIVE MOTION
    # ========================================================

    rows = compute_trace(

        obj_tracks,

        obj_vis,

        ref_tracks,

        ref_vis,

        seed_frame,

        object_diag
    )

    (
        new_auto,
        rigid_thr,
        point_thr
    ) = detect_onset(
        rows,
        seed_frame
    )

    save_trace(

        case_out
        / "motion_trace_localref.csv",

        rows
    )

    old_auto = (
        old_auto_map[
            case
        ]
    )

    if new_auto is None:

        error = None

        inside = False

        within2 = False

    else:

        error = (
            new_auto
            -
            human_start
        )

        inside = (
            human_start
            <= new_auto
            <= human_end
        )

        within2 = (
            abs(error)
            <= 2
        )

    review = (
        make_review_video(

            frames,

            fps,

            obj_tracks,

            obj_vis,

            ref_tracks,

            ref_vis,

            rows,

            human_start,

            human_end,

            old_auto,

            new_auto,

            case_out
        )
    )

    print(
        "OLD ABSOLUTE:",
        old_auto
    )

    print(
        "NEW LOCAL REF:",
        new_auto
    )

    print(
        "ERROR:",
        error
    )

    print(
        "IN HUMAN GT:",
        inside
    )

    print(
        "WITHIN ±2:",
        within2
    )

    print(
        "review:",
        review
    )

    np.savez_compressed(

        case_out
        / "local_reference_tracks.npz",

        tracks=
            ref_tracks.astype(
                np.float32
            ),

        visibility=
            ref_vis.astype(
                np.uint8
            ),

        seed_points=
            ref_points.astype(
                np.float32
            ),

        seed_frame=
            np.array(
                [seed_frame],
                dtype=np.int32
            )
    )

    summary.append({

        "case":
            case,

        "contact_gt":
            contact_gt,

        "human_start":
            human_start,

        "human_end":
            human_end,

        "old_absolute_onset":
            (
                old_auto
                if old_auto is not None
                else ""
            ),

        "localref_onset":
            (
                new_auto
                if new_auto is not None
                else ""
            ),

        "error_vs_human_start":
            (
                error
                if error is not None
                else ""
            ),

        "inside_human_interval":
            int(inside),

        "within_2_frames":
            int(within2),
    })

    del (
        video_tensor,
        queries,
        pred_tracks,
        pred_vis
    )

    torch.cuda.empty_cache()


# ============================================================
# FINAL CSV
# ============================================================

summary_csv = (
    OUT
    / "MOTION_PRIMARY9_LOCALREF_AUDIT.csv"
)

fields = [

    "case",

    "contact_gt",

    "human_start",
    "human_end",

    "old_absolute_onset",

    "localref_onset",

    "error_vs_human_start",

    "inside_human_interval",

    "within_2_frames",
]

with open(
    summary_csv,
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
# DISPLAY
# ============================================================

print()
print("=" * 120)

print(
    "PRIMARY9 LOCAL-REFERENCE MOTION AUDIT"
)

print("=" * 120)

print(
    f"{'CASE':29s} "
    f"{'HUMAN':>9s} "
    f"{'OLD':>6s} "
    f"{'LOCAL':>6s} "
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

        f"{str(r['old_absolute_onset']):>6s} "

        f"{str(r['localref_onset']):>6s} "

        f"{str(r['error_vs_human_start']):>5s} "

        f"{r['inside_human_interval']:6d} "

        f"{r['within_2_frames']:4d}"
    )


detected = [
    r
    for r in summary
    if r[
        "localref_onset"
    ] != ""
]

inside = sum(
    r[
        "inside_human_interval"
    ]
    for r in summary
)

near = sum(
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
print("=" * 120)
print("SUMMARY")
print("=" * 120)

print(
    "Detected:",
    f"{len(detected)}/{len(CASES)}"
)

print(
    "Inside human interval:",
    f"{inside}/{len(CASES)}"
)

print(
    "Within ±2:",
    f"{near}/{len(CASES)}"
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
print("Saved:")
print(
    summary_csv
)

print()
print(
    "GREEN = object CoTracker points"
)

print(
    "CYAN = nearby static reference points"
)

print(
    "BLUE border = old absolute-motion onset"
)

print(
    "RED border = new local-reference onset"
)

print(
    "YELLOW border = human motion interval"
)

print()
print(
    "No threshold was changed based on PRIMARY9."
)
