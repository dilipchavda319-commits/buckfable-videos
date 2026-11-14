"""BuckFable folder -> ready-to-upload video.

    python buckfable/engine/build.py --project buckfable/projects/social_security_cuts --out out
    add --selftest for an offline test (made-up pictures and a test voice)

Project folder needs:  scenes.json, timestamps.txt, voice/<one audio file>, images/<pictures named in scenes.json>
Optional:              music/<one mp3>  (ducked under the voice)
Output (in --out):     video.mp4, qc_sheet.jpg, subtitles.srt, report.json
"""
import argparse
import glob
import json
import os
import shutil
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import compose   # noqa: E402
import sfx       # noqa: E402
import tlparse   # noqa: E402

SR = sfx.SR
AUDIO_EXT = (".m4a", ".mp3", ".wav", ".aac", ".ogg", ".flac")


def first_file(folder, exts):
    for p in sorted(glob.glob(os.path.join(folder, "*"))):
        if p.lower().endswith(exts):
            return p
    return None


def synthetic_art(label, path, i):
    """Light stickman-style test picture (offline test only)."""
    im = Image.new("RGB", (1080, 1920), (252, 252, 250))
    d = ImageDraw.Draw(im)
    f = ImageFont.truetype(compose.FONT, 120)
    d.text((60, 190), "TEST", font=f, fill=(200, 30, 30))
    d.text((60, 330), f"#{i}", font=f, fill=(0, 0, 0))
    cx, cy = 540, 1150
    d.ellipse([cx - 90, cy - 330, cx + 90, cy - 150], outline=(0, 0, 0), width=14)
    d.line([cx, cy - 150, cx, cy + 150], fill=(0, 0, 0), width=14)
    d.line([cx, cy - 80, cx - 170, cy + 40], fill=(0, 0, 0), width=14)
    d.line([cx, cy - 80, cx + 170, cy + 40], fill=(0, 0, 0), width=14)
    d.line([cx, cy + 150, cx - 110, cy + 380], fill=(0, 0, 0), width=14)
    d.line([cx, cy + 150, cx + 110, cy + 380], fill=(0, 0, 0), width=14)
    d.rectangle([8, 8, 1071, 1911], outline=(0, 160, 60), width=6)       # edge marker: shows cropping
    d.text((60, 1700), label[:28], font=ImageFont.truetype(compose.FONT, 44), fill=(90, 90, 90))
    im.save(path)
    return path


def fake_voice(words, total):
    out = np.zeros(int(total * SR) + SR, np.float32)
    rng = np.random.default_rng(1)
    for w in words:
        n = int(max(0.05, (w["end"] - w["start"]) * 0.8) * SR)
        x = np.arange(n) / SR
        f0 = rng.uniform(105, 135)
        sig = sum(np.sin(2 * np.pi * f0 * k * x) / k for k in (1, 2, 3)) * np.sin(np.pi * x / (n / SR)) ** 0.6
        i0 = int(w["start"] * SR)
        out[i0:i0 + n] += (sig * 0.25).astype(np.float32)[:len(out) - i0]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", required=True)
    ap.add_argument("--out", default="out")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    proj = os.path.abspath(a.project)
    out = os.path.abspath(a.out)
    shutil.rmtree(out, ignore_errors=True)
    os.makedirs(os.path.join(out, "work"))

    with open(os.path.join(proj, "scenes.json"), encoding="utf-8") as f:
        sheet = json.load(f)
    total = float(sheet["total"])
    segs = tlparse.parse(os.path.join(proj, sheet.get("timestamps", "timestamps.txt")))
    words = tlparse.words_from(segs)
    warnings, errors = [], []

    # ---- validation (before any slow work)
    scenes_in = sheet["scenes"]
    voice_path = None if a.selftest else first_file(os.path.join(proj, "voice"), AUDIO_EXT)
    if not a.selftest and not voice_path:
        errors.append("MISSING voice file: put your narration (m4a/mp3/wav) inside the 'voice' folder")
    prev = 0.0
    for i, s in enumerate(scenes_in, 1):
        if abs(s["start"] - prev) > 0.02:
            errors.append(f"scene {i}: starts at {s['start']} but previous scene ended at {prev}")
        if s["end"] <= s["start"]:
            errors.append(f"scene {i}: end must be after start")
        prev = s["end"]
        if not a.selftest and not os.path.exists(os.path.join(proj, "images", s["image"])):
            errors.append(f"MISSING image: images/{s['image']}")
    if abs(prev - total) > 0.02:
        errors.append(f"last scene ends at {prev} but total is {total}")
    if errors:
        print("\n".join("ERROR: " + e for e in errors))
        sys.exit(2)

    # ---- scenes
    scenes = []
    for i, s in enumerate(scenes_in, 1):
        if a.selftest:
            p = synthetic_art(s["image"], os.path.join(out, "work", f"syn{i}.png"), i)
        else:
            p = os.path.join(proj, "images", s["image"])
        z = s.get("zoom", [1.0, 1.07])
        scenes.append({"t0": s["start"], "t1": s["end"], "canvas": compose.prepare_canvas(p),
                       "z0": z[0], "z1": z[1], "focus": s.get("focus", [0.5, 0.5]), "punches": []})

    # ---- phrase anchors -> punches + sound events
    events, anchors = [], []
    for i, (s, sc) in enumerate(zip(scenes_in, scenes)):
        for ph in s.get("punch", []):
            t = tlparse.find(words, ph, sc["t0"], sc["t1"])
            if t is None:
                warnings.append(f"scene {i + 1}: phrase '{ph}' not found; punch skipped")
            else:
                sc["punches"].append(t)
                anchors.append({"scene": i + 1, "phrase": ph, "t": round(t, 2)})
        if i and s.get("cut_sound", "swish") != "none":
            events.append((sc["t0"] - 0.05, "swish", s.get("cut_gain", 0.6)))
        for e in s.get("sfx", []):
            if "phrase" in e:
                t = tlparse.find(words, e["phrase"], sc["t0"], sc["t1"])
                if t is None:
                    warnings.append(f"scene {i + 1}: sfx phrase '{e['phrase']}' not found; used scene start")
                    t = sc["t0"]
            else:
                t = sc["t0"] + e.get("t", 0.0)
            events.append((t + e.get("offset", 0.0), e["sound"], e.get("gain", 1.0)))

    # ---- audio
    if a.selftest:
        voice = fake_voice(words, total)
    else:
        voice = np.frombuffer(__import__("subprocess").run(
            ["ffmpeg", "-v", "error", "-i", voice_path, "-f", "f32le", "-ac", "1", "-ar", str(SR), "pipe:1"],
            capture_output=True, check=True).stdout, dtype=np.float32).copy()
        print(f"voice: {voice_path}  length {len(voice) / SR:.2f}s (timeline {total}s)")
        if len(voice) / SR > total + 0.3:
            warnings.append(f"voice is {len(voice) / SR:.2f}s but timeline total is {total}s: the end will be cut")
        elif len(voice) / SR < total - 1.0:
            warnings.append(f"voice is {len(voice) / SR:.2f}s, shorter than timeline {total}s")
    music = first_file(os.path.join(proj, "music"), (".mp3", ".m4a", ".wav"))
    wav = os.path.join(out, "work", "mix.wav")
    sfx.write_wav(wav, sfx.mix(total, voice[:int(total * SR)], events, music))

    # ---- render
    ov = [{"text": o["text"], "from": o["from"], "to": o["to"]} for o in sheet.get("overlays", [])]
    r = compose.Renderer(scenes, ov, total, sheet.get("chip_y", 200))
    r.encode(wav, os.path.join(out, "video.mp4"))

    # ---- QC sheet: middle of every scene + last frame
    pts = [(s["t0"] + s["t1"]) / 2 for s in scenes] + [total - 0.1]
    tw, th, cols = 270, 480, 6
    rows = -(-len(pts) // cols)
    sheet_im = Image.new("RGB", (cols * tw, rows * th), (255, 255, 255))
    for k, t in enumerate(pts):
        fr = r.frame(t).resize((tw - 6, th - 6), Image.LANCZOS)
        sheet_im.paste(fr, ((k % cols) * tw + 3, (k // cols) * th + 3))
    sheet_im.save(os.path.join(out, "qc_sheet.jpg"), quality=88)

    with open(os.path.join(out, "subtitles.srt"), "w", encoding="utf-8") as f:
        f.write(tlparse.srt(segs))
    rep = {"title": sheet.get("title"), "total_seconds": total, "scenes": len(scenes),
           "selftest": a.selftest, "music": bool(music), "anchors": anchors, "warnings": warnings}
    with open(os.path.join(out, "report.json"), "w") as f:
        json.dump(rep, f, indent=2)
    print(f"\nDONE {total}s, {len(scenes)} scenes, warnings: {len(warnings)}")
    for w in warnings:
        print("WARNING:", w)


if __name__ == "__main__":
    main()
