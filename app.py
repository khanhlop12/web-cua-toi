import os
import uuid
import json
import shutil
import threading
import subprocess
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles
import uvicorn

# =========================================================
# BASIC CONFIG
# =========================================================

BASE_DIR = Path(__file__).resolve().parent
STORAGE_DIR = BASE_DIR / "storage"
INPUT_DIR = STORAGE_DIR / "inputs"
TEMP_DIR = STORAGE_DIR / "temp"
OUTPUT_DIR = STORAGE_DIR / "outputs"
JOB_DIR = STORAGE_DIR / "jobs"

for d in [INPUT_DIR, TEMP_DIR, OUTPUT_DIR, JOB_DIR]:
    d.mkdir(parents=True, exist_ok=True)

FPS = 16
MAX_VIDEO_SECONDS = 5

# =========================================================
# EXTERNAL MODEL REPOS
# =========================================================

DWPOSE_DIR = BASE_DIR / "external" / "dwpose"
MUSEPOSE_DIR = BASE_DIR / "external" / "musepose"

# =========================================================
# CONFIG MODEL COMMANDS
# =========================================================
# Bạn sửa 3 phần ở đây theo repo thật của bạn.
#
# 1) POSE_MODE:
#    - "frames_to_pose_frames"  : input là folder frames, output là folder pose PNG
#    - "video_to_pose_frames"   : input là video, output là folder pose PNG
#
# 2) MUSEPOSE_OUTPUT_MODE:
#    - "frames" : MusePose output ra thư mục PNG frames
#    - "video"  : MusePose output ra 1 file mp4
#
# 3) Lệnh CLI tương ứng của repo thật
# =========================================================

POSE_MODE = "frames_to_pose_frames"
MUSEPOSE_OUTPUT_MODE = "frames"

def build_pose_command(frames_dir: Path, video_path: Path, pose_dir: Path):
    """
    SỬA CHỖ NÀY theo repo DWPose/OpenPose thật của bạn.
    """
    if POSE_MODE == "frames_to_pose_frames":
        # Ví dụ giả định:
        return [
            "python",
            "infer.py",
            "--input_dir", str(frames_dir.resolve()),
            "--output_dir", str(pose_dir.resolve())
        ]

    elif POSE_MODE == "video_to_pose_frames":
        # Ví dụ nếu repo nhận trực tiếp video:
        return [
            "python",
            "infer_video.py",
            "--input_video", str(video_path.resolve()),
            "--output_dir", str(pose_dir.resolve())
        ]

    else:
        raise ValueError(f"Unsupported POSE_MODE: {POSE_MODE}")


def build_musepose_command(image_path: Path, pose_dir: Path, output_dir: Path, output_video: Path):
    """
    SỬA CHỖ NÀY theo repo MusePose thật của bạn.
    """

    if MUSEPOSE_OUTPUT_MODE == "frames":
        # Ví dụ giả định:
        return [
            "python",
            "inference.py",
            "--source_image", str(image_path.resolve()),
            "--pose_dir", str(pose_dir.resolve()),
            "--output_dir", str(output_dir.resolve())
        ]

    elif MUSEPOSE_OUTPUT_MODE == "video":
        # Ví dụ nếu MusePose output ra mp4:
        return [
            "python",
            "inference.py",
            "--source_image", str(image_path.resolve()),
            "--pose_dir", str(pose_dir.resolve()),
            "--output_video", str(output_video.resolve())
        ]

    else:
        raise ValueError(f"Unsupported MUSEPOSE_OUTPUT_MODE: {MUSEPOSE_OUTPUT_MODE}")


# =========================================================
# APP
# =========================================================

app = FastAPI(title="MusePose Motion Transfer API")
app.mount("/files", StaticFiles(directory=str(STORAGE_DIR)), name="files")


# =========================================================
# JOB STORE
# =========================================================

def job_file(job_id: str) -> Path:
    return JOB_DIR / f"{job_id}.json"

def update_job(job_id: str, status: str, progress: int = 0, output_path: str = None, error: str = None):
    data = {
        "job_id": job_id,
        "status": status,
        "progress": progress,
        "output_path": output_path,
        "error": error
    }
    with open(job_file(job_id), "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def read_job(job_id: str):
    path = job_file(job_id)
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# =========================================================
# SHELL
# =========================================================

def run_command(cmd, cwd=None):
    print("=" * 100)
    print("RUN CMD:", " ".join(map(str, cmd)))
    print("CWD:", str(cwd) if cwd else None)
    print("=" * 100)

    result = subprocess.run(
        list(map(str, cmd)),
        cwd=str(cwd) if cwd else None,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True
    )

    print(result.stdout)

    if result.returncode != 0:
        raise RuntimeError(f"Command failed:\n{' '.join(map(str, cmd))}\n\n{result.stdout}")

    return result.stdout


# =========================================================
# VIDEO HELPERS
# =========================================================

def probe_video(video_path: Path):
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "json",
        str(video_path)
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
    data = json.loads(result.stdout)
    duration = float(data["format"]["duration"])
    return {"duration": duration}

def trim_video(video_path: Path, output_path: Path, max_seconds: int = 5):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y",
        "-i", str(video_path),
        "-t", str(max_seconds),
        "-c:v", "libx264",
        "-c:a", "aac",
        str(output_path)
    ]
    run_command(cmd)

def extract_frames(video_path: Path, output_dir: Path, fps: int = 16):
    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y",
        "-i", str(video_path),
        "-vf", f"fps={fps}",
        str(output_dir / "%05d.png")
    ]
    run_command(cmd)

def make_video(frames_dir: Path, output_path: Path, audio_source: Path = None, fps: int = 16):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_video = output_path.with_name(output_path.stem + "_noaudio.mp4")

    cmd1 = [
        "ffmpeg", "-y",
        "-framerate", str(fps),
        "-i", str(frames_dir / "%05d.png"),
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        str(temp_video)
    ]
    run_command(cmd1)

    if audio_source:
        cmd2 = [
            "ffmpeg", "-y",
            "-i", str(temp_video),
            "-i", str(audio_source),
            "-c:v", "copy",
            "-c:a", "aac",
            "-shortest",
            str(output_path)
        ]
        run_command(cmd2)
    else:
        shutil.move(str(temp_video), str(output_path))


# =========================================================
# POSE + MUSEPOSE
# =========================================================

def extract_pose_sequence(frames_dir: Path, video_path: Path, pose_dir: Path):
    pose_dir.mkdir(parents=True, exist_ok=True)
    pngs = sorted([p for p in frames_dir.iterdir() if p.suffix.lower() == ".png"])
    if not pngs:
        raise RuntimeError(f"No frames found in {frames_dir}")
    for p in pngs:
        shutil.copy(str(p), str(pose_dir / p.name))


def run_musepose(image_path: Path, pose_dir: Path, gen_frames_dir: Path, gen_video_path: Path):
    from PIL import Image

    gen_frames_dir.mkdir(parents=True, exist_ok=True)

    pose_frames = sorted([p for p in pose_dir.iterdir() if p.suffix.lower() == ".png"])
    if not pose_frames:
        raise RuntimeError(f"No pose frames found in {pose_dir}")

    person_img = Image.open(image_path).convert("RGBA")

    for frame_path in pose_frames:
        frame = Image.open(frame_path).convert("RGBA")

        fw, fh = frame.size
        pw, ph = person_img.size

        # scale ảnh người về khoảng 35% chiều cao frame
        target_h = int(fh * 0.35)
        scale = target_h / ph
        target_w = int(pw * scale)

        resized_person = person_img.resize((target_w, target_h))

        # đặt ảnh vào giữa frame
        x = (fw - target_w) // 2
        y = (fh - target_h) // 2

        composed = frame.copy()
        composed.alpha_composite(resized_person, (x, y))

        out = composed.convert("RGB")
        out.save(gen_frames_dir / frame_path.name)
# =========================================================
# REFINE
# =========================================================

def refine_frames(gen_dir: Path, refined_dir: Path):
    refined_dir.mkdir(parents=True, exist_ok=True)
    pngs = sorted([p for p in gen_dir.iterdir() if p.suffix.lower() == ".png"])

    if not pngs:
        raise RuntimeError(f"No PNG frames found in generated dir: {gen_dir}")

    for p in pngs:
        shutil.copy(str(p), str(refined_dir / p.name))


# =========================================================
# PIPELINE
# =========================================================

def process_job(job_id: str, image_path: Path, video_path: Path):
    try:
        update_job(job_id, "processing", 5)

        temp_root = TEMP_DIR / job_id
        temp_root.mkdir(parents=True, exist_ok=True)

        trimmed_video = temp_root / "motion_trimmed.mp4"
        frames_dir = temp_root / "frames"
        pose_dir = temp_root / "poses"
        gen_frames_dir = temp_root / "generated_frames"
        gen_video_path = temp_root / "generated.mp4"
        refined_dir = temp_root / "refined"

        output_root = OUTPUT_DIR / job_id
        output_root.mkdir(parents=True, exist_ok=True)
        output_video = output_root / "output.mp4"

        # 1. trim nếu > 5s
        meta = probe_video(video_path)
        duration = meta["duration"]

        if duration > MAX_VIDEO_SECONDS:
            trim_video(video_path, trimmed_video, MAX_VIDEO_SECONDS)
            video_for_process = trimmed_video
        else:
            video_for_process = video_path

        update_job(job_id, "processing", 15)

        # 2. extract source frames từ motion video
        extract_frames(video_for_process, frames_dir, fps=FPS)
        update_job(job_id, "processing", 30)

        # 3. extract pose
        extract_pose_sequence(frames_dir, video_for_process, pose_dir)
        update_job(job_id, "processing", 50)

        # 4. run MusePose
        run_musepose(image_path, pose_dir, gen_frames_dir, gen_video_path)
        update_job(job_id, "processing", 70)

        # 5. xử lý tùy loại output của MusePose
        if MUSEPOSE_OUTPUT_MODE == "frames":
            source_for_refine = gen_frames_dir
        elif MUSEPOSE_OUTPUT_MODE == "video":
            # convert generated video thành frames
            extract_frames(gen_video_path, gen_frames_dir, fps=FPS)
            source_for_refine = gen_frames_dir
        else:
            raise ValueError(f"Unsupported MUSEPOSE_OUTPUT_MODE: {MUSEPOSE_OUTPUT_MODE}")

        update_job(job_id, "processing", 80)

        # 6. refine
        refine_frames(source_for_refine, refined_dir)
        update_job(job_id, "processing", 90)

        # 7. encode final video
        make_video(refined_dir, output_video, audio_source=video_for_process, fps=FPS)

        rel_output = output_video.relative_to(STORAGE_DIR)
        update_job(
            job_id,
            "completed",
            100,
            output_path=f"/files/{rel_output.as_posix()}",
            error=None
        )

    except Exception as e:
        update_job(job_id, "failed", 100, error=str(e))


# =========================================================
# API
# =========================================================

@app.get("/")
def root():
    return {
        "message": "MusePose Motion Transfer API",
        "fps": FPS,
        "max_video_seconds": MAX_VIDEO_SECONDS,
        "pose_mode": POSE_MODE,
        "musepose_output_mode": MUSEPOSE_OUTPUT_MODE
    }

@app.post("/jobs")
async def create_job(image: UploadFile = File(...), video: UploadFile = File(...)):
    image_name = image.filename.lower()
    video_name = video.filename.lower()

    if not image_name.endswith((".jpg", ".jpeg", ".png")):
        raise HTTPException(status_code=400, detail="Image must be jpg/jpeg/png")

    if not video_name.endswith(".mp4"):
        raise HTTPException(status_code=400, detail="Video must be mp4")

    job_id = str(uuid.uuid4())
    input_root = INPUT_DIR / job_id
    input_root.mkdir(parents=True, exist_ok=True)

    image_ext = ".png" if image_name.endswith(".png") else ".jpg"
    image_path = input_root / f"source{image_ext}"
    video_path = input_root / "motion.mp4"

    with open(image_path, "wb") as f:
        shutil.copyfileobj(image.file, f)

    with open(video_path, "wb") as f:
        shutil.copyfileobj(video.file, f)

    update_job(job_id, "queued", 0)

    thread = threading.Thread(
        target=process_job,
        args=(job_id, image_path, video_path),
        daemon=True
    )
    thread.start()

    return {
        "job_id": job_id,
        "status": "queued"
    }

@app.get("/jobs/{job_id}")
def get_job(job_id: str):
    data = read_job(job_id)
    if not data:
        raise HTTPException(status_code=404, detail="Job not found")
    return data

@app.get("/jobs/{job_id}/result")
def get_result(job_id: str):
    data = read_job(job_id)
    if not data:
        raise HTTPException(status_code=404, detail="Job not found")
    return data


# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)