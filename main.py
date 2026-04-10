import os
import numpy as np
from moviepy.editor import VideoFileClip, AudioFileClip, CompositeVideoClip, ImageClip
import whisper
from PIL import Image, ImageDraw, ImageFont

# ---------------- CONFIG ----------------
ASSETS_DIR = "assets"
OUTPUT_DIR = "output"
FONT_PATH = os.path.abspath("fonts/comic.ttf")

BASE_VIDEO = os.path.join(ASSETS_DIR, "base_video.mp4")
INPUT_AUDIO = "input_audio.mp3"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ---------------- TEXT RENDER FUNCTION (NO IMAGEMAGICK) ----------------
def create_text_clip(text, font_path, fontsize, color, stroke_color, stroke_width):
    font = ImageFont.truetype(font_path, fontsize)

    # Calculate text size
    dummy_img = Image.new("RGBA", (1, 1))
    draw = ImageDraw.Draw(dummy_img)
    bbox = draw.textbbox((0, 0), text, font=font, stroke_width=stroke_width)

    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]

    # Create image
    img = Image.new("RGBA", (text_width + 20, text_height + 20), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    draw.text(
        (10, 10),
        text,
        font=font,
        fill=color,
        stroke_width=stroke_width,
        stroke_fill=stroke_color
    )

    return ImageClip(np.array(img))


# ---------------- LOAD FILES ----------------
print("Loading video and audio...")

video = VideoFileClip(BASE_VIDEO)
audio = AudioFileClip(INPUT_AUDIO)

video_duration = video.duration
audio_duration = audio.duration


# ---------------- SYNC VIDEO + AUDIO ----------------
print("Syncing video with audio...")

if audio_duration > video_duration:
    video = video.loop(duration=audio_duration)
else:
    video = video.subclip(0, audio_duration)

video = video.set_audio(audio)


# ---------------- TRANSCRIPTION ----------------
print("Transcribing audio...")

model = whisper.load_model("base")

result = model.transcribe(INPUT_AUDIO, word_timestamps=True)

words = []
for segment in result["segments"]:
    for word in segment["words"]:
        text = word["word"].strip()

        if not text:
            continue

        start = word["start"]
        end = word["end"]

        if end - start <= 0.05:
            continue

        words.append({
            "text": text,
            "start": start,
            "end": end
        })


# ---------------- CREATE SUBTITLES ----------------
print("Creating subtitles...")

W, H = video.size
subtitle_clips = []

for w in words:
    txt_clip = create_text_clip(
        w["text"],
        FONT_PATH,
        fontsize=80,
        color="white",
        stroke_color="black",
        stroke_width=2
    )

    txt_clip = txt_clip.set_start(w["start"]).set_end(w["end"])

    # Bottom 40% placement
    y_position = int(H * 0.75)

    txt_clip = txt_clip.set_position(("center", y_position))

    subtitle_clips.append(txt_clip)


# ---------------- FINAL COMPOSITION ----------------
print("Rendering final video...")

final_video = CompositeVideoClip([video, *subtitle_clips])

output_path = os.path.join(OUTPUT_DIR, "final_video.mp4")

final_video.write_videofile(
    output_path,
    fps=30,
    codec="libx264",
    audio_codec="aac"
)

print(f"Done! Video saved at: {output_path}")