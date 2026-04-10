import os
import numpy as np
from moviepy.editor import VideoFileClip, AudioFileClip, CompositeVideoClip, ImageClip
import whisper
from PIL import Image, ImageDraw, ImageFont

# ================================================================
# CONFIG
# ================================================================
ASSETS_DIR       = "assets"
OUTPUT_DIR       = "output"
FONT_PATH        = os.path.abspath("fonts/comic.ttf")
BASE_VIDEO       = os.path.join(ASSETS_DIR, "base_video.mp4")
INPUT_AUDIO      = "input_audio.mp3"

FONTSIZE         = 55
TEXT_COLOR       = "#113426"
WORDS_PER_GROUP  = 3
MIN_WORD_DUR     = 0.08
MAX_GAP_MERGE    = 0.35

SUBTITLE_Y_RATIO = 0.15
BOTTOM_MARGIN    = 24

USE_PILL_BG      = True
PILL_COLOR       = (244, 243, 211, 255)
PILL_RADIUS      = 18

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ================================================================
# STEP 1 - DRAW A SINGLE SUBTITLE FRAME (Pillow, no ImageMagick)
# ================================================================
def render_subtitle_clip(text: str, font_path: str, fontsize: int,
                          text_color: str, video_width: int) -> ImageClip:
    font = ImageFont.truetype(font_path, fontsize)

    dummy = Image.new("RGBA", (1, 1))
    draw  = ImageDraw.Draw(dummy)
    bbox  = draw.textbbox((0, 0), text, font=font)
    tw    = bbox[2] - bbox[0]
    th    = bbox[3] - bbox[1]

    pad_x, pad_y = 48, 28
    img_w = tw + pad_x * 2
    img_h = th + pad_y * 2

    img  = Image.new("RGBA", (img_w, img_h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    if USE_PILL_BG:
        draw.rounded_rectangle([(0, 0), (img_w - 1, img_h - 1)],
                                radius=PILL_RADIUS, fill=PILL_COLOR)

    text_x = pad_x - bbox[0]
    text_y = pad_y - bbox[1]
    draw.text((text_x, text_y), text, font=font, fill=text_color)

    clip = ImageClip(np.array(img))
    return clip, img_h


# ================================================================
# STEP 2 - GROUP WHISPER WORDS INTO READABLE CHUNKS
# ================================================================
def group_words(words: list, words_per_group: int, max_gap: float) -> list:
    groups  = []
    current = []

    for i, w in enumerate(words):
        if current:
            gap = w["start"] - current[-1]["end"]
            if gap > max_gap or len(current) >= words_per_group:
                groups.append({
                    "text":  " ".join(c["text"] for c in current),
                    "start": current[0]["start"],
                    "end":   current[-1]["end"],
                })
                current = []

        current.append(w)

    if current:
        groups.append({
            "text":  " ".join(c["text"] for c in current),
            "start": current[0]["start"],
            "end":   current[-1]["end"],
        })

    return groups


# ================================================================
# STEP 3 - CALCULATE SAFE Y POSITION
# ================================================================
def safe_y_position(video_height: int, clip_height: int) -> float:
    zone_top    = int(video_height * 0.60)
    zone_bottom = video_height - BOTTOM_MARGIN

    available_space = zone_bottom - zone_top - clip_height
    if available_space < 0:
        return float(zone_top)

    y = zone_top + available_space * SUBTITLE_Y_RATIO
    return float(y)


# ================================================================
# MAIN
# ================================================================
def main():
    print("Loading video and audio...")
    video = VideoFileClip(BASE_VIDEO)
    audio = AudioFileClip(INPUT_AUDIO)

    video_w, video_h = video.size
    audio_dur        = audio.duration
    video_dur        = video.duration

    print("Syncing video with audio...")
    if audio_dur > video_dur:
        video = video.loop(duration=audio_dur)
    else:
        video = video.subclip(0, audio_dur)

    video = video.set_audio(audio)

    print("Transcribing audio (Whisper)...")
    model  = whisper.load_model("base")
    result = model.transcribe(INPUT_AUDIO, word_timestamps=True)

    raw_words = []
    for segment in result["segments"]:
        for word in segment.get("words", []):
            text  = word["word"].strip()
            start = word["start"]
            end   = word["end"]
            dur   = end - start

            if not text or dur < MIN_WORD_DUR:
                continue

            raw_words.append({"text": text, "start": start, "end": end})

    print("Whisper returned " + str(len(raw_words)) + " usable words.")

    print("Grouping words into subtitle chunks...")
    subtitle_groups = group_words(raw_words, WORDS_PER_GROUP, MAX_GAP_MERGE)
    print("Grouped into " + str(len(subtitle_groups)) + " subtitle chunks.")

    print("Rendering subtitle clips...")
    overlay_clips = []

    for group in subtitle_groups:
        clip, clip_h = render_subtitle_clip(
            text        = group["text"],
            font_path   = FONT_PATH,
            fontsize    = FONTSIZE,
            text_color  = TEXT_COLOR,
            video_width = video_w,
        )

        y = safe_y_position(video_h, clip_h)

        clip = (
            clip
            .set_start(group["start"])
            .set_end(group["end"])
            .set_position(("center", y))
        )

        overlay_clips.append(clip)

    print("Compositing and exporting final video...")
    final = CompositeVideoClip([video, *overlay_clips])

    out_path = os.path.join(OUTPUT_DIR, "final_video.mp4")
    final.write_videofile(
        out_path,
        fps         = 30,
        codec       = "libx264",
        audio_codec = "aac",
        threads     = 4,
        preset      = "fast",
        logger      = "bar",
    )

    print("Done! Saved to: " + out_path)


if __name__ == "__main__":
    main()