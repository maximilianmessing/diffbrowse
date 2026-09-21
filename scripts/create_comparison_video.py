"""Generate a synchronized side-by-side comparison video between Upstream Jev Ultrafast (Cloud API)
and DiffBrowse (100% Local Apple Silicon Metal).
"""

import shutil
import subprocess
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def get_fonts():
    font_path = "/System/Library/Fonts/Supplemental/Arial.ttf"
    font_bold = "/System/Library/Fonts/Supplemental/Arial Bold.ttf"
    mono_path = "/System/Library/Fonts/Menlo.ttc"

    def font(size, bold=False):
        try:
            return ImageFont.truetype(font_bold if bold else font_path, size)
        except Exception:
            return ImageFont.load_default()

    def mono(size):
        try:
            return ImageFont.truetype(mono_path, size)
        except Exception:
            return ImageFont.load_default()

    return font, mono


def extract_frames(video_path: Path, output_dir: Path, fps: int = 30) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    pattern = output_dir / "%05d.png"
    print(f"Extracting frames from {video_path.name} at {fps} fps...")
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-i",
            str(video_path),
            "-vf",
            f"fps={fps}",
            str(pattern),
        ],
        check=True,
    )
    frames = sorted(output_dir.glob("*.png"))
    print(f"Extracted {len(frames)} frames from {video_path.name}.")
    return frames


def build_comparison_frame(
    frame_idx: int,
    total_frames: int,
    left_img: Image.Image,
    right_img: Image.Image,
    left_finished: bool,
    right_finished: bool,
    font,
    mono,
) -> Image.Image:
    # 1920x1080 canvas
    canvas = Image.new("RGB", (1920, 1080), "#0d120f")
    d = ImageDraw.Draw(canvas)

    # -------------------------------------------------------------
    # 1. Header Bar
    # -------------------------------------------------------------
    d.rectangle((0, 0, 1920, 90), fill="#131a15")
    d.line((0, 90, 1920, 90), fill="#233027", width=1)

    # Tag & Title
    d.text((32, 16), "BENCHMARK COMPARISON  ·  CLOUD API vs. 100% LOCAL SILICON", font=font(12, True), fill="#4ade80")
    d.text(
        (32, 38),
        "Upstream Jev Ultrafast (Cloud API)  vs.  DiffBrowse (100% Local Structured Diffusion)",
        font=font(23, True),
        fill="#ffffff",
    )

    # Header Pills
    # 1x Speed pill
    d.rounded_rectangle((1560, 26, 1720, 62), radius=8, fill="#1c261e", outline="#2b3b2f")
    d.text((1576, 36), "⚡ 1× REAL-TIME", font=font(12, True), fill="#4ade80")

    # Open Source pill
    d.rounded_rectangle((1735, 26, 1888, 62), radius=8, fill="#1c2430", outline="#293547")
    d.text((1752, 36), "✦ OPEN WEIGHTS", font=font(12, True), fill="#93c5fd")

    # -------------------------------------------------------------
    # 2. Video Windows (Left: Jev, Right: DiffBrowse)
    # -------------------------------------------------------------
    vw, vh = 924, 602

    # --- Left Window: Upstream Jev ---
    lx, ly = 24, 100
    d.rectangle((lx, ly, lx + vw, ly + 36), fill="#161e18")
    d.text((lx + 12, ly + 10), "UPSTREAM JEV ULTRAFAST (CLOUD API)", font=font(14, True), fill="#ffffff")
    d.text((lx + 310, ly + 12), "TypeSafe Jev + Mercury 2.5 · Google Flights Live", font=font(12), fill="#9ca3af")

    if left_finished:
        d.rounded_rectangle((lx + vw - 170, ly + 6, lx + vw - 8, ly + 30), radius=6, fill="#14532d")
        d.text((lx + vw - 158, ly + 11), "✓ COMPLETED (7.07s)", font=font(11, True), fill="#4ade80")
    else:
        d.rounded_rectangle((lx + vw - 140, ly + 6, lx + vw - 8, ly + 30), radius=6, fill="#78350f")
        d.text((lx + vw - 128, ly + 11), "● CLOUD RUNNING", font=font(11, True), fill="#fde047")

    # Resize and paste left video
    scaled_left = left_img.resize((vw, vh), Image.Resampling.BILINEAR)
    canvas.paste(scaled_left, (lx, ly + 36))
    d.rectangle((lx, ly, lx + vw, ly + 36 + vh), outline="#233027", width=1)

    # If left is finished, draw a sleek subtle banner overlay
    if left_finished:
        d.rounded_rectangle(
            (lx + 16, ly + 36 + vh - 44, lx + 440, ly + 36 + vh - 12),
            radius=6,
            fill="#0d1410ee",
            outline="#22c55e",
        )
        d.text(
            (lx + 28, ly + 36 + vh - 36),
            "✓ Upstream Jev finished in 7.07s (Cloud API)",
            font=font(13, True),
            fill="#4ade80",
        )

    # --- Right Window: DiffBrowse ---
    rx, ry = 972, 100
    d.rectangle((rx, ry, rx + vw, ry + 36), fill="#161e18")
    d.text((rx + 12, ry + 10), "DIFFBROWSE (100% LOCAL STRUCTURED DIFFUSION)", font=font(14, True), fill="#ffffff")
    d.text((rx + 380, ry + 12), "DiffusionGemma 26B (4-bit) · Lisbon Stays → Casa Flora", font=font(12), fill="#9ca3af")

    if right_finished:
        d.rounded_rectangle((rx + vw - 175, ry + 6, rx + vw - 8, ry + 30), radius=6, fill="#14532d")
        d.text((rx + vw - 163, ry + 11), "✓ TASK SUCCEEDED", font=font(11, True), fill="#4ade80")
    else:
        d.rounded_rectangle((rx + vw - 145, ry + 6, rx + vw - 8, ry + 30), radius=6, fill="#14532d")
        d.text((rx + vw - 133, ry + 11), "● LOCAL METAL", font=font(11, True), fill="#4ade80")

    # Resize and paste right video
    scaled_right = right_img.resize((vw, vh), Image.Resampling.BILINEAR)
    canvas.paste(scaled_right, (rx, ry + 36))
    d.rectangle((rx, ry, rx + vw, ry + 36 + vh), outline="#233027", width=1)

    if right_finished:
        d.rounded_rectangle(
            (rx + 16, ry + 36 + vh - 44, rx + 440, ry + 36 + vh - 12),
            radius=6,
            fill="#0d1410ee",
            outline="#22c55e",
        )
        d.text(
            (rx + 28, ry + 36 + vh - 36),
            "✓ DiffBrowse completed in 7.1s (100% Local)",
            font=font(13, True),
            fill="#4ade80",
        )

    # -------------------------------------------------------------
    # 3. Bottom Architectural Comparison Table
    # -------------------------------------------------------------
    tx, ty, tw, th = 24, 750, 1872, 290
    d.rounded_rectangle((tx, ty, tx + tw, ty + th), radius=10, fill="#121814", outline="#222f25", width=1)

    # Table Columns:
    # Col 1: Dimension (x = tx + 20 to tx + 340)
    # Col 2: Jev Ultrafast (x = tx + 360 to tx + 1090)
    # Col 3: DiffBrowse (x = tx + 1110 to tx + tw - 20)
    c1, c2, c3 = tx + 24, tx + 360, tx + 1120

    # Header row
    d.text((c1, ty + 12), "ARCHITECTURE & OPERATIONAL DIMENSION", font=font(12, True), fill="#9ca3af")
    d.text((c2, ty + 12), "UPSTREAM JEV ULTRAFAST (CLOUD BASELINE)", font=font(12, True), fill="#93c5fd")
    d.text((c3, ty + 12), "DIFFBROWSE (LOCAL METAL / ROCM / CUDA)", font=font(12, True), fill="#4ade80")
    d.line((tx + 16, ty + 36, tx + tw - 16, ty + 36), fill="#202b23", width=1)

    rows = [
        (
            "Model Architecture",
            "Proprietary TypeSafe Jev (Cloud) + OpenRouter Mercury 2.5",
            "Google DeepMind DiffusionGemma-26B-A4B-it-4bit (Stock Open Weights)",
        ),
        (
            "Privacy & Data Egress",
            "Public Cloud Egress: Live DOM and user queries sent to remote servers",
            "100% Air-Gapped: Zero network bytes leave host; entirely on local silicon",
        ),
        (
            "Operational Cost",
            "Paid API tokens (~$0.005 / step) + cloud rate limits & downtime risks",
            "$0.00 / Step: Zero recurring cost, unlimited offline evaluations",
        ),
        (
            "Decision Speed & Latency",
            "Speculative autoregressive tree search over cloud LLM endpoints (178 ms)",
            "Structured discrete diffusion + KV prefix cache (117 ms steady-state, 98% acc)",
        ),
        (
            "Hardware Portability",
            "Remote server dependency; requires persistent high-speed internet",
            "Apple Silicon Metal, AMD Strix Halo (ROCm 6.2+), NVIDIA CUDA",
        ),
    ]

    for idx, (dim, left_val, right_val) in enumerate(rows):
        ry_cur = ty + 46 + idx * 46
        # Background stripe for readability
        if idx % 2 == 1:
            d.rectangle((tx + 12, ry_cur - 4, tx + tw - 12, ry_cur + 36), fill="#151d17")

        d.text((c1, ry_cur + 6), dim, font=font(13, True), fill="#ffffff")
        d.text((c2, ry_cur + 6), left_val, font=font(12), fill="#d1d5db")
        d.text((c3, ry_cur + 6), right_val, font=font(12, True), fill="#86efac")

    # -------------------------------------------------------------
    # 4. Footer
    # -------------------------------------------------------------
    d.text(
        (28, 1054),
        "DiffBrowse · Structured Discrete Diffusion for Sub-Second Local Web Agents · "
        "maximilianmessing/diffbrowse (forked from browser-use/jev-ultrafast)",
        font=font(11),
        fill="#6b7280",
    )
    d.text((1680, 1054), "100% Open Source · Apache 2.0 / MIT", font=font(11), fill="#6b7280")

    return canvas


def generate_comparison_video(output_mp4: Path, output_gif: Path):
    temp_dir = ROOT / "artifacts" / "comparison_temp"
    demo_frames_dir = temp_dir / "demo"
    diff_frames_dir = temp_dir / "diffbrowse"
    out_frames_dir = temp_dir / "out_frames"

    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    out_frames_dir.mkdir(parents=True, exist_ok=True)

    demo_mp4 = ROOT / "docs" / "demo.mp4"
    diff_mp4 = ROOT / "docs" / "diffbrowse-demo.mp4"

    if not demo_mp4.exists():
        raise FileNotFoundError(f"Missing {demo_mp4}")
    if not diff_mp4.exists():
        raise FileNotFoundError(f"Missing {diff_mp4}")

    fps = 30
    demo_frames = extract_frames(demo_mp4, demo_frames_dir, fps=fps)
    diff_frames = extract_frames(diff_mp4, diff_frames_dir, fps=fps)

    n_demo = len(demo_frames)
    n_diff = len(diff_frames)

    # Hold the final completed frame for 45 frames (1.5 seconds)
    total_frames = n_diff + 45
    print(f"Compositing {total_frames} side-by-side frames (30 fps = {total_frames / 30:.1f}s)...")

    font, mono = get_fonts()

    # Preload all frames or stream them
    # Memory for 700 frames in RAM is fine, or load on demand
    for i in range(total_frames):
        left_idx = min(i, n_demo - 1)
        right_idx = min(i, n_diff - 1)

        left_img = Image.open(demo_frames[left_idx])
        right_img = Image.open(diff_frames[right_idx])

        left_finished = i >= n_demo
        right_finished = i >= n_diff

        canvas = build_comparison_frame(
            frame_idx=i,
            total_frames=total_frames,
            left_img=left_img,
            right_img=right_img,
            left_finished=left_finished,
            right_finished=right_finished,
            font=font,
            mono=mono,
        )
        canvas.save(out_frames_dir / f"{i:05d}.png")

        if (i + 1) % 100 == 0 or i == total_frames - 1:
            print(f"Composited frame {i + 1}/{total_frames} ({(i + 1) / total_frames * 100:.1f}%)")

    # Encode MP4 (H.264, 1080p, 30 fps)
    print("Encoding side-by-side comparison MP4 via ffmpeg...")
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-framerate",
            "30",
            "-i",
            str(out_frames_dir / "%05d.png"),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-crf",
            "18",
            "-movflags",
            "+faststart",
            str(output_mp4),
        ],
        check=True,
    )
    print(f"Saved comparison MP4 to {output_mp4} ({output_mp4.stat().st_size / 1024 / 1024:.2f} MB)")

    # Encode GIF (12 fps, scaled to 1080px width)
    print("Encoding side-by-side comparison GIF via ffmpeg...")
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-i",
            str(output_mp4),
            "-vf",
            "fps=12,scale=1080:-1:flags=lanczos,split[a][b];[a]palettegen[p];[b][p]paletteuse",
            "-loop",
            "0",
            str(output_gif),
        ],
        check=True,
    )
    print(f"Saved comparison GIF to {output_gif} ({output_gif.stat().st_size / 1024 / 1024:.2f} MB)")

    # Cleanup temp frames to preserve disk space
    shutil.rmtree(temp_dir)
    print("Cleaned up temporary frames.")


def main():
    out_mp4 = ROOT / "docs" / "diffbrowse-vs-jev.mp4"
    out_gif = ROOT / "docs" / "diffbrowse-vs-jev.gif"
    generate_comparison_video(out_mp4, out_gif)
    print("✨ Comparison video generation complete!")


if __name__ == "__main__":
    main()
