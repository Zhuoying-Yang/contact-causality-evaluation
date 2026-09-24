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

CHECKPOINT = (
    PHYS
    / "ReVidgen"
    / "checkpoints"
    / "Cotracker"
    / "scaled_offline.pth"
)

OUT = (
    ROOT
    / "pipeline20_e2e_v1/motion_abs"
)

OUT.mkdir(
    parents=True,
    exist_ok=True
)

FFMPEG = Path(
    "/data/home/zhuoyingyang/"
    "miniconda3/envs/cosmos25/bin/ffmpeg"
)


# ============================================================
# FROZEN HUMAN OBJECT-RESPONSE LABELS
#
# IMPORTANT:
# These labels are ONLY used AFTER automatic detection.
# They are NOT used to initialize CoTracker or choose onset.
# ============================================================

CASES = [{'seed': 'seed101', 'vid': '0004', 'contact_gt': 'UNLABELED_FRESH', 'human_start': 123, 'human_end': 124, 'phrase': 'yellow cube'}, {'seed': 'seed101', 'vid': '0014', 'contact_gt': 'UNLABELED_FRESH', 'human_start': 123, 'human_end': 124, 'phrase': 'white cup'}, {'seed': 'seed101', 'vid': '0015', 'contact_gt': 'UNLABELED_FRESH', 'human_start': 123, 'human_end': 124, 'phrase': 'white cup'}, {'seed': 'seed101', 'vid': '0029', 'contact_gt': 'UNLABELED_FRESH', 'human_start': 123, 'human_end': 124, 'phrase': 'brown cube'}, {'seed': 'seed101', 'vid': '0034', 'contact_gt': 'UNLABELED_FRESH', 'human_start': 123, 'human_end': 124, 'phrase': 'yellow cube'}, {'seed': 'seed101', 'vid': '0040', 'contact_gt': 'UNLABELED_FRESH', 'human_start': 123, 'human_end': 124, 'phrase': 'brown cube'}, {'seed': 'seed101', 'vid': '0042', 'contact_gt': 'UNLABELED_FRESH', 'human_start': 123, 'human_end': 124, 'phrase': 'brown cube'}, {'seed': 'seed102', 'vid': '0015', 'contact_gt': 'UNLABELED_FRESH', 'human_start': 123, 'human_end': 124, 'phrase': 'white cup'}, {'seed': 'seed102', 'vid': '0019', 'contact_gt': 'UNLABELED_FRESH', 'human_start': 123, 'human_end': 124, 'phrase': 'banana'}, {'seed': 'seed102', 'vid': '0025', 'contact_gt': 'UNLABELED_FRESH', 'human_start': 123, 'human_end': 124, 'phrase': 'yellow cube'}, {'seed': 'seed102', 'vid': '0027', 'contact_gt': 'UNLABELED_FRESH', 'human_start': 123, 'human_end': 124, 'phrase': 'white tape roll'}, {'seed': 'seed102', 'vid': '0029', 'contact_gt': 'UNLABELED_FRESH', 'human_start': 123, 'human_end': 124, 'phrase': 'brown cube'}, {'seed': 'seed102', 'vid': '0036', 'contact_gt': 'UNLABELED_FRESH', 'human_start': 123, 'human_end': 124, 'phrase': 'white tape roll'}, {'seed': 'seed102', 'vid': '0042', 'contact_gt': 'UNLABELED_FRESH', 'human_start': 123, 'human_end': 124, 'phrase': 'brown cube'}, {'seed': 'seed103', 'vid': '0003', 'contact_gt': 'UNLABELED_FRESH', 'human_start': 123, 'human_end': 124, 'phrase': 'yellow cube'}, {'seed': 'seed103', 'vid': '0004', 'contact_gt': 'UNLABELED_FRESH', 'human_start': 123, 'human_end': 124, 'phrase': 'yellow cube'}, {'seed': 'seed103', 'vid': '0026', 'contact_gt': 'UNLABELED_FRESH', 'human_start': 123, 'human_end': 124, 'phrase': 'white tape'}, {'seed': 'seed103', 'vid': '0028', 'contact_gt': 'UNLABELED_FRESH', 'human_start': 123, 'human_end': 124, 'phrase': 'white tape roll'}, {'seed': 'seed103', 'vid': '0035', 'contact_gt': 'UNLABELED_FRESH', 'human_start': 123, 'human_end': 124, 'phrase': 'yellow cube'}, {'seed': 'seed103', 'vid': '0036', 'contact_gt': 'UNLABELED_FRESH', 'human_start': 123, 'human_end': 124, 'phrase': 'white tape roll'}]


# ============================================================
# FROZEN MOTION PARAMETERS
# ============================================================

N_POINTS = 60

INWARD_PX = 4.0

MIN_COMMON_FRACTION = 0.25

BASELINE_VALID_STEPS = 5

MAD_MULTIPLIER = 6.0

MIN_RIGID_SPEED_NORM = 0.002

MIN_POINT_RESPONSE_NORM = 0.002

POINT_RESPONSE_QUANTILE = 0.75

PERSIST_WINDOW = 3

PERSIST_REQUIRED = 2

# Automatic seed:
# earliest valid object mask in first 10 frames.
#
# This does NOT use human response timing.
SEED_SEARCH_FRAMES = 10

MIN_SEED_AREA = 40

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
            f"Cannot open video: {path}"
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
# PAI TARGET OBJECT MASK
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
            W,
        )

    elif x.shape == (H, T * W):

        x = (
            x.reshape(
                H,
                T,
                W,
            )
            .transpose(
                1,
                0,
                2,
            )
        )

    elif x.size == T * H * W:

        x = x.reshape(
            T,
            H,
            W,
        )

    else:

        raise RuntimeError(
            f"Cannot decode RLE: "
            f"{x.shape}, "
            f"expected {(T,H,W)}"
        )

    return x.astype(bool)


def load_object_masks(
    seed_name,
    vid,
    phrase,
):

    path = (
        PAI_ROOT
        / f"cosmos3_{seed_name}"
        / "sam"
        / f"{vid}.pkl"
    )

    with open(
        path,
        "rb",
    ) as f:

        items = pickle.load(f)

    matches = [
        x
        for x in items
        if x.get("phrase")
        == phrase
    ]

    if len(matches) != 1:

        available = [
            x.get("phrase")
            for x in items
        ]

        raise RuntimeError(
            f"{seed_name}_{vid}: "
            f"{phrase!r} matches="
            f"{len(matches)}; "
            f"available={available}"
        )

    return decode_3d_rle(
        matches[0][
            "segmentation_mask_rle"
        ]
    )


# ============================================================
# MASK CLEANING
# ============================================================

def largest_component(mask):

    m = (
        mask.astype(np.uint8)
    )

    n, labels, stats, _ = (
        cv2.connectedComponentsWithStats(
            m,
            connectivity=8,
        )
    )

    if n <= 1:

        return mask.astype(bool)

    areas = (
        stats[
            1:,
            cv2.CC_STAT_AREA
        ]
    )

    lab = (
        int(np.argmax(areas))
        + 1
    )

    return (
        labels == lab
    )


def mask_area(mask):

    return int(
        np.count_nonzero(mask)
    )


def bbox_diag(mask):

    ys, xs = np.nonzero(mask)

    if len(xs) == 0:

        return np.nan

    w = (
        xs.max()
        - xs.min()
        + 1
    )

    h = (
        ys.max()
        - ys.min()
        + 1
    )

    return float(
        np.hypot(
            w,
            h,
        )
    )


def largest_contour(mask):

    contours, _ = cv2.findContours(
        (
            mask.astype(np.uint8)
            * 255
        ),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_NONE,
    )

    if not contours:

        return np.empty(
            (0, 2),
            dtype=np.float32,
        )

    c = max(
        contours,
        key=cv2.contourArea,
    )

    return (
        c[:, 0, :]
        .astype(np.float32)
    )


def uniform_contour_sample(
    contour,
    n,
):

    if len(contour) <= n:

        return contour.copy()

    idx = np.linspace(
        0,
        len(contour),
        n,
        endpoint=False,
    ).astype(int)

    idx = np.clip(
        idx,
        0,
        len(contour) - 1,
    )

    return contour[idx]


# ============================================================
# MOVE CONTOUR POINTS INTO MASK
#
# Use distance transform rather than centroid movement.
# Goal: seed CoTracker slightly inside object surface.
# ============================================================

def move_points_inward(
    mask,
    contour_points,
    target_dist=4.0,
):

    m = (
        mask.astype(np.uint8)
    )

    dt = cv2.distanceTransform(
        m,
        cv2.DIST_L2,
        5,
    )

    H, W = mask.shape

    radius = int(
        np.ceil(
            target_dist + 5
        )
    )

    result = []

    for p in contour_points:

        px = int(
            round(float(p[0]))
        )

        py = int(
            round(float(p[1]))
        )

        x0 = max(
            0,
            px - radius,
        )

        x1 = min(
            W,
            px + radius + 1,
        )

        y0 = max(
            0,
            py - radius,
        )

        y1 = min(
            H,
            py + radius + 1,
        )

        local_dt = dt[
            y0:y1,
            x0:x1
        ]

        ys, xs = np.where(
            local_dt >= target_dist
        )

        # Preferred:
        # nearest pixel that lies at least
        # target_dist inside the mask.
        if len(xs) > 0:

            gx = xs + x0
            gy = ys + y0

            dist_to_contour = (
                (gx - px) ** 2
                +
                (gy - py) ** 2
            )

            k = int(
                np.argmin(
                    dist_to_contour
                )
            )

            qx = gx[k]
            qy = gy[k]

        else:

            # Thin masks may not contain
            # a full 4-px interior.
            #
            # Fallback to deepest local
            # interior pixel.
            k = int(
                np.argmax(
                    local_dt
                )
            )

            ly, lx = np.unravel_index(
                k,
                local_dt.shape,
            )

            qx = lx + x0
            qy = ly + y0

        result.append(
            [
                float(qx),
                float(qy),
            ]
        )

    return np.asarray(
        result,
        dtype=np.float32,
    )


# ============================================================
# AUTOMATIC EARLY SEED
#
# Human onset is NOT consulted here.
# ============================================================

def choose_seed(
    object_masks,
):

    T = object_masks.shape[0]

    end = min(
        T,
        SEED_SEARCH_FRAMES,
    )

    for t in range(end):

        m = largest_component(
            object_masks[t]
        )

        if (
            mask_area(m)
            >= MIN_SEED_AREA
        ):

            contour = (
                largest_contour(m)
            )

            if len(contour) >= 10:

                return t, m

    return None, None


# ============================================================
# MAD
# ============================================================

def robust_mad(x):

    x = np.asarray(
        x,
        dtype=np.float64,
    )

    med = np.median(x)

    return float(
        np.median(
            np.abs(
                x - med
            )
        )
    )


# ============================================================
# SEED REVIEW
# ============================================================

def save_seed_review(
    frame,
    mask,
    points,
    path,
    case,
    seed,
):

    out = frame.copy()

    overlay = out.copy()

    overlay[
        mask.astype(bool)
    ] = (
        0,
        255,
        0,
    )

    out = cv2.addWeighted(
        out,
        0.70,
        overlay,
        0.30,
        0,
    )

    contour = largest_contour(
        mask
    )

    if len(contour):

        cv2.polylines(
            out,
            [
                contour.astype(
                    np.int32
                )
            ],
            True,
            (0, 255, 255),
            1,
        )

    for x, y in points:

        cv2.circle(
            out,
            (
                int(round(x)),
                int(round(y)),
            ),
            3,
            (255, 0, 255),
            -1,
        )

    cv2.rectangle(
        out,
        (10, 10),
        (850, 80),
        (0, 0, 0),
        -1,
    )

    cv2.putText(
        out,
        case,
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    cv2.putText(
        out,
        f"AUTO EARLY SEED FRAME = {seed}",
        (20, 68),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )

    cv2.imwrite(
        str(path),
        out,
    )


# ============================================================
# MOTION TRACE
# ============================================================

def compute_motion_trace(
    tracks,
    visibility,
    seed,
    object_diag,
):

    T, N, _ = tracks.shape

    rows = []

    for t in range(T):

        if t <= seed:

            rows.append({

                "frame": t,

                "common_points": 0,

                "common_fraction":
                    0.0,

                "median_dx":
                    np.nan,

                "median_dy":
                    np.nan,

                "rigid_speed_norm":
                    np.nan,

                "point_q75_norm":
                    np.nan,

                "median_point_speed_norm":
                    np.nan,

                "coherence":
                    np.nan,
            })

            continue

        vis_prev = (
            visibility[t - 1]
        )

        vis_now = (
            visibility[t]
        )

        common = (
            vis_prev
            & vis_now
        )

        idx = np.where(
            common
        )[0]

        n_common = len(idx)

        common_fraction = (
            n_common
            / max(N, 1)
        )

        if n_common == 0:

            rows.append({

                "frame": t,

                "common_points": 0,

                "common_fraction":
                    0.0,

                "median_dx":
                    np.nan,

                "median_dy":
                    np.nan,

                "rigid_speed_norm":
                    np.nan,

                "point_q75_norm":
                    np.nan,

                "median_point_speed_norm":
                    np.nan,

                "coherence":
                    np.nan,
            })

            continue

        # SAME POINT IDs ONLY.
        delta = (
            tracks[t, idx]
            -
            tracks[t - 1, idx]
        )

        dx = delta[:, 0]

        dy = delta[:, 1]

        median_dx = float(
            np.median(dx)
        )

        median_dy = float(
            np.median(dy)
        )

        rigid_speed_px = float(
            np.hypot(
                median_dx,
                median_dy,
            )
        )

        rigid_speed_norm = (
            rigid_speed_px
            / object_diag
        )

        point_speed_px = (
            np.linalg.norm(
                delta,
                axis=1,
            )
        )

        point_q75_norm = float(
            np.quantile(
                point_speed_px,
                POINT_RESPONSE_QUANTILE,
            )
            / object_diag
        )

        median_point_speed_norm = (
            float(
                np.median(
                    point_speed_px
                )
            )
            / object_diag
        )

        coherence = float(
            rigid_speed_px
            /
            max(
                float(
                    np.median(
                        point_speed_px
                    )
                ),
                EPS,
            )
        )

        coherence = min(
            coherence,
            1.0,
        )

        rows.append({

            "frame":
                t,

            "common_points":
                n_common,

            "common_fraction":
                common_fraction,

            "median_dx":
                median_dx,

            "median_dy":
                median_dy,

            "rigid_speed_norm":
                rigid_speed_norm,

            "point_q75_norm":
                point_q75_norm,

            "median_point_speed_norm":
                median_point_speed_norm,

            "coherence":
                coherence,
        })

    return rows


# ============================================================
# AUTOMATIC ONSET
# ============================================================

def detect_onset(
    rows,
    seed,
):

    valid_baseline = [
        r
        for r in rows
        if (
            r["frame"] > seed
            and
            r["common_fraction"]
            >= MIN_COMMON_FRACTION
            and
            np.isfinite(
                r[
                    "rigid_speed_norm"
                ]
            )
            and
            np.isfinite(
                r[
                    "point_q75_norm"
                ]
            )
        )
    ][
        :BASELINE_VALID_STEPS
    ]

    if len(valid_baseline) < 3:

        return {
            "status":
                "INSUFFICIENT_BASELINE"
        }

    rb = np.asarray(
        [
            r["rigid_speed_norm"]
            for r in valid_baseline
        ],
        dtype=float,
    )

    pb = np.asarray(
        [
            r["point_q75_norm"]
            for r in valid_baseline
        ],
        dtype=float,
    )

    rigid_threshold = max(

        MIN_RIGID_SPEED_NORM,

        float(
            np.median(rb)
        )
        +
        MAD_MULTIPLIER
        *
        robust_mad(rb),
    )

    point_threshold = max(

        MIN_POINT_RESPONSE_NORM,

        float(
            np.median(pb)
        )
        +
        MAD_MULTIPLIER
        *
        robust_mad(pb),
    )

    T = len(rows)

    candidate = np.zeros(
        T,
        dtype=bool,
    )

    for r in rows:

        t = r["frame"]

        valid = (
            r[
                "common_fraction"
            ]
            >= MIN_COMMON_FRACTION
            and
            np.isfinite(
                r[
                    "rigid_speed_norm"
                ]
            )
            and
            np.isfinite(
                r[
                    "point_q75_norm"
                ]
            )
        )

        if not valid:
            continue

        rigid_trigger = (
            r[
                "rigid_speed_norm"
            ]
            >
            rigid_threshold
        )

        point_trigger = (
            r[
                "point_q75_norm"
            ]
            >
            point_threshold
        )

        # Frozen conceptual rule:
        # rigid response OR point-level response.
        candidate[t] = (
            rigid_trigger
            or
            point_trigger
        )

    detected = None

    for t in range(
        seed + 1,
        T,
    ):

        if not candidate[t]:
            continue

        stop = min(
            T,
            t + PERSIST_WINDOW,
        )

        count = int(
            candidate[
                t:stop
            ].sum()
        )

        if (
            count
            >= PERSIST_REQUIRED
        ):

            detected = t
            break

    for r in rows:

        t = r["frame"]

        r["rigid_threshold"] = (
            rigid_threshold
        )

        r["point_threshold"] = (
            point_threshold
        )

        r["motion_candidate"] = int(
            candidate[t]
        )

        r["auto_onset"] = (
            detected
            if detected is not None
            else ""
        )

    return {

        "status":
            "OK",

        "auto_onset":
            detected,

        "rigid_threshold":
            rigid_threshold,

        "point_threshold":
            point_threshold,

        "baseline_frames":
            [
                int(
                    r["frame"]
                )
                for r
                in valid_baseline
            ],
    }


# ============================================================
# SAVE TRACE CSV
# ============================================================

def save_trace(
    rows,
    path,
):

    fields = [

        "frame",

        "common_points",
        "common_fraction",

        "median_dx",
        "median_dy",

        "rigid_speed_norm",

        "point_q75_norm",

        "median_point_speed_norm",

        "coherence",

        "rigid_threshold",
        "point_threshold",

        "motion_candidate",

        "auto_onset",
    ]

    with open(
        path,
        "w",
        newline="",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fields,
        )

        writer.writeheader()

        writer.writerows(rows)


# ============================================================
# REVIEW VIDEO
# ============================================================

def make_review_video(
    frames,
    fps,
    tracks,
    visibility,
    rows,
    human_start,
    human_end,
    auto_onset,
    case_name,
    case_out,
):

    T, H, W, _ = frames.shape

    if auto_onset is None:

        focus_start = (
            human_start
        )

        focus_end = (
            human_end
        )

    else:

        focus_start = min(
            human_start,
            auto_onset,
        )

        focus_end = max(
            human_end,
            auto_onset,
        )

    start = max(
        0,
        focus_start - 15,
    )

    end = min(
        T - 1,
        focus_end + 15,
    )

    tmp = (
        case_out
        / "motion_review_mp4v.mp4"
    )

    final = (
        case_out
        / "motion_review_h264.mp4"
    )

    writer = cv2.VideoWriter(

        str(tmp),

        cv2.VideoWriter_fourcc(
            *"mp4v"
        ),

        fps,

        (W, H),
    )

    if not writer.isOpened():

        raise RuntimeError(
            f"Cannot open VideoWriter: "
            f"{tmp}"
        )

    by_frame = {
        int(r["frame"]): r
        for r in rows
    }

    for t in range(
        start,
        end + 1,
    ):

        frame = (
            frames[t]
            .copy()
        )

        vis = (
            visibility[t]
        )

        pts = (
            tracks[t]
        )

        for i in np.where(vis)[0]:

            x, y = pts[i]

            cv2.circle(
                frame,
                (
                    int(round(x)),
                    int(round(y)),
                ),
                3,
                (0, 255, 0),
                -1,
            )

        r = by_frame[t]

        in_human = (
            human_start
            <= t
            <= human_end
        )

        is_auto = (
            auto_onset is not None
            and
            t == auto_onset
        )

        # Human window border.
        if in_human:

            cv2.rectangle(
                frame,
                (2, 2),
                (W - 3, H - 3),
                (0, 255, 255),
                6,
            )

        # Auto onset border.
        if is_auto:

            cv2.rectangle(
                frame,
                (10, 10),
                (W - 11, H - 11),
                (0, 0, 255),
                6,
            )

        cv2.rectangle(
            frame,
            (10, 10),
            (1160, 180),
            (0, 0, 0),
            -1,
        )

        auto_text = (
            str(auto_onset)
            if auto_onset is not None
            else "NONE"
        )

        candidate = (
            "YES"
            if r[
                "motion_candidate"
            ]
            else "NO"
        )

        texts = [

            (
                f"{case_name}   "
                f"FRAME={t}"
            ),

            (
                f"HUMAN MOTION="
                f"[{human_start},"
                f"{human_end}]   "
                f"AUTO={auto_text}"
            ),

            (
                f"rigid="
                f"{r['rigid_speed_norm']:.5f}  "
                f"thr="
                f"{r['rigid_threshold']:.5f}"
            ),

            (
                f"pointQ75="
                f"{r['point_q75_norm']:.5f}  "
                f"thr="
                f"{r['point_threshold']:.5f}"
            ),

            (
                f"common="
                f"{r['common_fraction']:.2f}  "
                f"candidate={candidate}"
            ),
        ]

        y = 38

        for text in texts:

            cv2.putText(
                frame,
                text,
                (22, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.67,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            y += 31

        writer.write(frame)

    writer.release()

    # Transcode for Mac/browser compatibility.
    if FFMPEG.exists():

        cmd = [

            str(FFMPEG),

            "-y",

            "-i",
            str(tmp),

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
            stderr=subprocess.PIPE,
            text=True,
        )

        if (
            p.returncode == 0
            and final.exists()
        ):

            try:
                tmp.unlink()
            except Exception:
                pass

            return final

        print(
            "WARNING: ffmpeg failed, "
            "keeping MP4V:",
            p.stderr[-500:],
        )

    return tmp


# ============================================================
# MODEL
# ============================================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print("=" * 110)
print("PRIMARY9 OBJECT MOTION AUDIT")
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


for case_idx, case in enumerate(
    CASES,
    start=1,
):

    seed_name = (
        case["seed"]
    )

    vid = (
        case["vid"]
    )

    phrase = (
        case["phrase"]
    )

    human_start = int(
        case["human_start"]
    )

    human_end = int(
        case["human_end"]
    )

    case_name = (
        f"Cosmos3_{seed_name}_{vid}"
    )

    case_out = (
        OUT
        / case_name
    )

    case_out.mkdir(
        parents=True,
        exist_ok=True,
    )

    print()
    print("=" * 110)

    print(
        f"[{case_idx}/{len(CASES)}]",
        case_name,
    )

    print(
        "Human motion:",
        f"{human_start}-{human_end}"
    )

    print("=" * 110)

    video_path = (
        VIDEO_ROOT
        / seed_name
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

    object_masks = (
        load_object_masks(
            seed_name,
            vid,
            phrase,
        )
    )

    if object_masks.shape[0] != T:

        raise RuntimeError(
            f"{case_name}: "
            f"video T={T}, "
            f"mask T="
            f"{object_masks.shape[0]}"
        )

    seed_frame, seed_mask = (
        choose_seed(
            object_masks
        )
    )

    if seed_frame is None:

        print(
            "NO VALID EARLY MASK"
        )

        summary.append({

            "case":
                case_name,

            "contact_gt":
                case[
                    "contact_gt"
                ],

            "human_start":
                human_start,

            "human_end":
                human_end,

            "seed_frame":
                "",

            "auto_onset":
                "",

            "start_error":
                "",

            "inside_human_interval":
                0,

            "within_2_frames":
                0,

            "status":
                "NO_VALID_SEED",
        })

        continue

    contour = largest_contour(
        seed_mask
    )

    contour = uniform_contour_sample(
        contour,
        N_POINTS,
    )

    points = move_points_inward(
        seed_mask,
        contour,
        INWARD_PX,
    )

    diag = bbox_diag(
        seed_mask
    )

    print(
        "frames:",
        T,
        "| seed:",
        seed_frame,
        "| points:",
        len(points),
        "| diag:",
        f"{diag:.2f}"
    )

    save_seed_review(

        frames[
            seed_frame
        ],

        seed_mask,

        points,

        case_out
        / "seed_review.png",

        case_name,

        seed_frame,
    )


    # ========================================================
    # COTRACKER
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
            2,
        )[None]
        .float()
    )

    query_time = np.full(
        (
            len(points),
            1,
        ),
        seed_frame,
        dtype=np.float32,
    )

    queries_np = np.concatenate(
        [
            query_time,
            points,
        ],
        axis=1,
    )

    queries = (
        torch.from_numpy(
            queries_np
        )[None]
        .float()
    )

    print(
        "Tracking..."
    )

    with torch.inference_mode():

        (
            pred_tracks,
            pred_visibility,
        ) = model(

            video_tensor.to(
                device
            ),

            queries=queries.to(
                device
            ),
        )

    tracks = (
        pred_tracks[0]
        .detach()
        .cpu()
        .numpy()
    )

    visibility = (
        pred_visibility[0]
        .detach()
        .cpu()
        .numpy()
    )

    visibility = (
        visibility > 0.5
    )

    print(
        "tracks:",
        tracks.shape
    )


    # ========================================================
    # MOTION
    # ========================================================

    rows = compute_motion_trace(

        tracks,

        visibility,

        seed_frame,

        diag,
    )

    detection = detect_onset(
        rows,
        seed_frame,
    )

    if detection[
        "status"
    ] != "OK":

        print(
            "DETECTION STATUS:",
            detection[
                "status"
            ]
        )

        summary.append({

            "case":
                case_name,

            "contact_gt":
                case[
                    "contact_gt"
                ],

            "human_start":
                human_start,

            "human_end":
                human_end,

            "seed_frame":
                seed_frame,

            "auto_onset":
                "",

            "start_error":
                "",

            "inside_human_interval":
                0,

            "within_2_frames":
                0,

            "status":
                detection[
                    "status"
                ],
        })

        continue

    auto = detection[
        "auto_onset"
    ]

    save_trace(
        rows,
        case_out
        / "motion_trace.csv",
    )

    if auto is None:

        inside = False

        within2 = False

        error = ""

    else:

        inside = (
            human_start
            <= auto
            <= human_end
        )

        error = (
            auto
            - human_start
        )

        within2 = (
            abs(error)
            <= 2
        )

    review_video = (
        make_review_video(

            frames,

            fps,

            tracks,

            visibility,

            rows,

            human_start,

            human_end,

            auto,

            case_name,

            case_out,
        )
    )

    print()
    print(
        "AUTO MOTION:",
        auto
    )

    print(
        "HUMAN:",
        f"{human_start}-{human_end}"
    )

    print(
        "ERROR vs human_start:",
        error
    )

    print(
        "INSIDE HUMAN INTERVAL:",
        inside
    )

    print(
        "WITHIN ±2:",
        within2
    )

    print(
        "rigid threshold:",
        f"{detection['rigid_threshold']:.6f}"
    )

    print(
        "point threshold:",
        f"{detection['point_threshold']:.6f}"
    )

    print(
        "review:",
        review_video
    )

    np.savez_compressed(

        case_out
        / "cotracker_motion_tracks.npz",

        tracks=
            tracks.astype(
                np.float32
            ),

        visibility=
            visibility.astype(
                np.uint8
            ),

        seed_points=
            points.astype(
                np.float32
            ),

        seed_frame=
            np.asarray(
                [seed_frame],
                dtype=np.int32,
            ),
    )

    summary.append({

        "case":
            case_name,

        "contact_gt":
            case[
                "contact_gt"
            ],

        "human_start":
            human_start,

        "human_end":
            human_end,

        "seed_frame":
            seed_frame,

        "auto_onset":
            (
                auto
                if auto is not None
                else ""
            ),

        "start_error":
            error,

        "inside_human_interval":
            int(inside),

        "within_2_frames":
            int(within2),

        "status":
            (
                "OK"
                if auto is not None
                else "NO_RESPONSE"
            ),
    })

    del (
        video_tensor,
        queries,
        pred_tracks,
        pred_visibility,
    )

    torch.cuda.empty_cache()


# ============================================================
# SUMMARY CSV
# ============================================================

summary_csv = (
    OUT
    / "MOTION_PRIMARY9_AUDIT.csv"
)

fields = [

    "case",

    "contact_gt",

    "human_start",
    "human_end",

    "seed_frame",

    "auto_onset",

    "start_error",

    "inside_human_interval",

    "within_2_frames",

    "status",
]

with open(
    summary_csv,
    "w",
    newline="",
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=fields,
    )

    writer.writeheader()

    writer.writerows(
        summary
    )


# ============================================================
# PRINT FINAL TABLE
# ============================================================

print()
print("=" * 115)

print(
    "PRIMARY9 MOTION ONSET AUDIT"
)

print("=" * 115)

print(
    f"{'CASE':29s} "
    f"{'HUMAN':>10s} "
    f"{'AUTO':>6s} "
    f"{'ERR':>5s} "
    f"{'IN_GT':>6s} "
    f"{'±2':>4s} "
    f"{'STATUS':>18s}"
)

for r in summary:

    human = (
        f"{r['human_start']}-"
        f"{r['human_end']}"
    )

    auto = str(
        r["auto_onset"]
    )

    err = str(
        r["start_error"]
    )

    print(
        f"{r['case']:29s} "
        f"{human:>10s} "
        f"{auto:>6s} "
        f"{err:>5s} "
        f"{r['inside_human_interval']:6d} "
        f"{r['within_2_frames']:4d} "
        f"{r['status']:>18s}"
    )


valid_auto = [
    r
    for r in summary
    if isinstance(
        r["auto_onset"],
        int,
    )
]

inside_n = sum(
    r[
        "inside_human_interval"
    ]
    for r in valid_auto
)

near_n = sum(
    r[
        "within_2_frames"
    ]
    for r in valid_auto
)

errors = [
    abs(
        int(
            r["start_error"]
        )
    )
    for r in valid_auto
    if r[
        "start_error"
    ] != ""
]

print()
print("=" * 115)
print("SUMMARY")
print("=" * 115)

print(
    "Automatic detections:",
    f"{len(valid_auto)}/{len(CASES)}"
)

print(
    "Inside frozen human interval:",
    f"{inside_n}/{len(CASES)}"
)

print(
    "Within ±2 frames of human_start:",
    f"{near_n}/{len(CASES)}"
)

if errors:

    print(
        "Median |frame error|:",
        float(
            np.median(errors)
        )
    )

    print(
        "Max |frame error|:",
        int(
            np.max(errors)
        )
    )

print()
print("Saved:")
print(summary_csv)
print()
print(
    "Each case directory contains:"
)
print(
    "  seed_review.png"
)
print(
    "  motion_trace.csv"
)
print(
    "  cotracker_motion_tracks.npz"
)
print(
    "  motion_review_h264.mp4"
)

print()
print("=" * 115)

print(
    "IMPORTANT:"
)

print(
    "Human labels were NOT used to choose "
    "the automatic seed or automatic onset."
)

print(
    "If seed_review.png shows a wrong target mask, "
    "mark that case segmentation-invalid rather than "
    "moving the seed based on the motion result."
)

print("=" * 115)
