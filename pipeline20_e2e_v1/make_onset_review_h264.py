#!/usr/bin/env python3

from pathlib import Path
import os
import csv
import cv2
import subprocess
import shutil


# ============================================================
# PATHS
# ============================================================

ROOT = Path(
    "/shared/ssd_30T/zhuoyingyang/physact/"
    "sam3_robowm/pipeline20_e2e_v1"
)

ONSET_CSV = ROOT / "PREDICTED_MOTION_ONSETS.csv"

VIDEO_ROOT = Path(
    "/shared/ssd_30T/zhuoyingyang/physact/"
    "cosmos3/export_robowm68_3seeds"
)

OUT = ROOT / "onset_review_h264"
OUT.mkdir(parents=True, exist_ok=True)


# ============================================================
# SETTINGS
# ============================================================

WINDOW_RADIUS = 24

FONT = cv2.FONT_HERSHEY_SIMPLEX


# ============================================================
# HELPERS
# ============================================================

def parse_case(case):
    # Cosmos3_seed101_0004
    parts = case.split("_")

    if len(parts) != 3:
        raise ValueError(case)

    seed = parts[1]
    vid = parts[2]

    return seed, vid


def h264_encode(src, dst):
    cmd = [
        os.environ["REVIEW_FFMPEG"],
        "-y",
        "-loglevel", "error",
        "-i", str(src),
        "-c:v", "libx264",
        "-preset", "fast",
        "-crf", "18",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        "-an",
        str(dst),
    ]

    subprocess.run(
        cmd,
        check=True,
    )


def draw_label(frame, text, xy, scale=0.75):
    x, y = xy

    (tw, th), baseline = cv2.getTextSize(
        text,
        FONT,
        scale,
        2,
    )

    cv2.rectangle(
        frame,
        (x - 8, y - th - 10),
        (x + tw + 8, y + baseline + 8),
        (0, 0, 0),
        -1,
    )

    cv2.putText(
        frame,
        text,
        (x, y),
        FONT,
        scale,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )


def annotate_frame(
    frame,
    idx,
    n_frames,
    onset,
):
    H, W = frame.shape[:2]

    # --------------------------------------------------------
    # Frame counter
    # --------------------------------------------------------

    draw_label(
        frame,
        f"FRAME {idx:03d} / {n_frames - 1:03d}",
        (25, 42),
        0.8,
    )

    # --------------------------------------------------------
    # Onset label
    # --------------------------------------------------------

    if onset is None:

        text = "HYBRID: NO RESPONSE DETECTED"

        (tw, _), _ = cv2.getTextSize(
            text,
            FONT,
            0.75,
            2,
        )

        draw_label(
            frame,
            text,
            (W - tw - 30, 42),
            0.75,
        )

    else:

        text = f"PREDICTED HYB ONSET = {onset}"

        (tw, _), _ = cv2.getTextSize(
            text,
            FONT,
            0.75,
            2,
        )

        draw_label(
            frame,
            text,
            (W - tw - 30, 42),
            0.75,
        )

        delta = idx - onset

        # Exact onset = RED
        if delta == 0:

            cv2.rectangle(
                frame,
                (5, 5),
                (W - 6, H - 6),
                (0, 0, 255),
                10,
            )

            text2 = "<<< EXACT PREDICTED ONSET >>>"

            (tw2, _), _ = cv2.getTextSize(
                text2,
                FONT,
                1.05,
                3,
            )

            x2 = max(
                10,
                (W - tw2) // 2,
            )

            draw_label(
                frame,
                text2,
                (x2, 95),
                1.05,
            )

        # +/- 2 frames = YELLOW
        elif abs(delta) <= 2:

            cv2.rectangle(
                frame,
                (7, 7),
                (W - 8, H - 8),
                (0, 255, 255),
                7,
            )

            text2 = f"ONSET {delta:+d} FRAME"

            (tw2, _), _ = cv2.getTextSize(
                text2,
                FONT,
                0.9,
                2,
            )

            draw_label(
                frame,
                text2,
                ((W - tw2) // 2, 90),
                0.9,
            )

    # --------------------------------------------------------
    # Bottom timeline
    # --------------------------------------------------------

    left = 25
    right = W - 25
    y = H - 28

    cv2.line(
        frame,
        (left, y),
        (right, y),
        (220, 220, 220),
        4,
    )

    # current frame
    current_x = int(
        left
        +
        (right - left)
        * idx
        / max(1, n_frames - 1)
    )

    cv2.circle(
        frame,
        (current_x, y),
        7,
        (255, 255, 255),
        -1,
    )

    # predicted onset marker
    if onset is not None:

        onset_x = int(
            left
            +
            (right - left)
            * onset
            / max(1, n_frames - 1)
        )

        cv2.line(
            frame,
            (onset_x, y - 18),
            (onset_x, y + 18),
            (0, 0, 255),
            5,
        )

    return frame


def write_review(
    src_video,
    tmp_video,
    onset,
    start_frame=None,
    end_frame=None,
):
    cap = cv2.VideoCapture(
        str(src_video)
    )

    if not cap.isOpened():
        raise RuntimeError(
            f"Cannot open {src_video}"
        )

    fps = cap.get(
        cv2.CAP_PROP_FPS
    )

    width = int(
        cap.get(
            cv2.CAP_PROP_FRAME_WIDTH
        )
    )

    height = int(
        cap.get(
            cv2.CAP_PROP_FRAME_HEIGHT
        )
    )

    n_frames = int(
        cap.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
    )

    if fps <= 0:
        fps = 24.0

    if start_frame is None:
        start_frame = 0

    if end_frame is None:
        end_frame = n_frames - 1

    start_frame = max(
        0,
        start_frame,
    )

    end_frame = min(
        n_frames - 1,
        end_frame,
    )

    writer = cv2.VideoWriter(
        str(tmp_video),
        cv2.VideoWriter_fourcc(
            *"mp4v"
        ),
        fps,
        (width, height),
    )

    if not writer.isOpened():
        raise RuntimeError(
            f"Cannot open writer: {tmp_video}"
        )

    cap.set(
        cv2.CAP_PROP_POS_FRAMES,
        start_frame,
    )

    idx = start_frame

    while idx <= end_frame:

        ok, frame = cap.read()

        if not ok:
            break

        frame = annotate_frame(
            frame,
            idx,
            n_frames,
            onset,
        )

        writer.write(
            frame
        )

        idx += 1

    cap.release()
    writer.release()

    return (
        fps,
        n_frames,
    )


# ============================================================
# READ ONSETS
# ============================================================

with open(
    ONSET_CSV,
    newline="",
) as f:

    rows = list(
        csv.DictReader(f)
    )


index_rows = []


print("=" * 90)
print("GENERATING H264 ONSET REVIEW VIDEOS")
print("=" * 90)


for i, row in enumerate(
    rows,
    1,
):

    case = row["case"].strip()

    onset_raw = (
        row.get(
            "predicted_motion_onset",
            ""
        )
        .strip()
    )

    if onset_raw in {
        "",
        "None",
        "nan",
        "NaN",
    }:
        onset = None
    else:
        onset = int(
            float(onset_raw)
        )

    seed, vid = parse_case(
        case
    )

    src = (
        VIDEO_ROOT
        / seed
        / f"{vid}.mp4"
    )

    if not src.exists():
        print(
            f"[{i:02d}/20] {case} MISSING VIDEO"
        )
        continue

    case_out = OUT / case
    case_out.mkdir(
        parents=True,
        exist_ok=True,
    )

    print()
    print(
        f"[{i:02d}/20] {case}"
    )
    print(
        "source:",
        src,
    )
    print(
        "onset:",
        onset,
    )

    # ========================================================
    # FULL REVIEW
    # ========================================================

    tmp_full = (
        case_out
        / "tmp_full.mp4"
    )

    full_h264 = (
        case_out
        / f"{case}_full_review_h264.mp4"
    )

    fps, n_frames = write_review(
        src,
        tmp_full,
        onset,
    )

    h264_encode(
        tmp_full,
        full_h264,
    )

    tmp_full.unlink(
        missing_ok=True
    )

    print(
        "full:",
        full_h264,
    )

    # ========================================================
    # ONSET WINDOW
    # ========================================================

    window_h264 = ""

    if onset is not None:

        start = (
            onset
            - WINDOW_RADIUS
        )

        end = (
            onset
            + WINDOW_RADIUS
        )

        tmp_window = (
            case_out
            / "tmp_window.mp4"
        )

        window_h264 = (
            case_out
            / f"{case}_onset_window_h264.mp4"
        )

        write_review(
            src,
            tmp_window,
            onset,
            start,
            end,
        )

        h264_encode(
            tmp_window,
            window_h264,
        )

        tmp_window.unlink(
            missing_ok=True
        )

        print(
            f"window: frames "
            f"{max(0,start)}-{min(n_frames-1,end)}"
        )

        print(
            "window:",
            window_h264,
        )

    index_rows.append({
        "case":
            case,
        "predicted_onset":
            "" if onset is None else onset,
        "fps":
            fps,
        "frames":
            n_frames,
        "full_review":
            str(full_h264),
        "onset_window":
            str(window_h264),
    })


# ============================================================
# INDEX
# ============================================================

index_csv = (
    OUT
    / "REVIEW_INDEX.csv"
)

with open(
    index_csv,
    "w",
    newline="",
) as f:

    fields = [
        "case",
        "predicted_onset",
        "fps",
        "frames",
        "full_review",
        "onset_window",
    ]

    w = csv.DictWriter(
        f,
        fieldnames=fields,
    )

    w.writeheader()
    w.writerows(
        index_rows
    )


print()
print("=" * 90)
print("DONE")
print("=" * 90)
print(
    "Cases:",
    len(index_rows),
)
print(
    "Output:",
    OUT,
)
print(
    "Index:",
    index_csv,
)
