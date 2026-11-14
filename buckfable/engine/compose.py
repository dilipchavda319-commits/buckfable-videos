"""BuckFable renderer: still images -> keyframe push/pull, cut punches, timed chips.
Built for LIGHT hand-drawn art (no dark gradient, no vignette, no grain)."""
import bisect
import math
import os
import subprocess

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

W, H, FPS = 1080, 1920, 30
HERE = os.path.dirname(os.path.abspath(__file__))
FONT = os.path.join(os.path.dirname(HERE), "fonts", "Poppins-Bold.ttf")


def smooth(p):
    p = min(1.0, max(0.0, p))
    return p * p * (3 - 2 * p)


def back_out(p):
    p = min(1.0, max(0.0, p))
    c1 = 1.70158
    return 1 + (c1 + 1) * (p - 1) ** 3 + c1 * (p - 1) ** 2


def prepare_canvas(path):
    """Load a picture; if it is not close to 9:16, pad it with its own border colour."""
    im = Image.open(path).convert("RGB")
    s = 2600 / max(im.size)
    if s < 1:
        im = im.resize((int(im.width * s), int(im.height * s)), Image.LANCZOS)
    w, h = im.size
    target = W / H
    if abs((w / h) / target - 1) <= 0.12:
        return im
    arr = np.asarray(im)
    border = np.concatenate([arr[0], arr[-1], arr[:, 0], arr[:, -1]])
    col = tuple(int(x) for x in np.median(border, axis=0))
    if w / h > target:
        cw, ch = w, int(round(w / target))
    else:
        ch, cw = h, int(round(h * target))
    canvas = Image.new("RGB", (cw, ch), col)
    canvas.paste(im, ((cw - w) // 2, (ch - h) // 2))
    return canvas


def camera(img, z, focus, punch):
    sw, sh = img.size
    if sw / sh > W / H:
        bh, bw = sh, sh * W / H
    else:
        bw, bh = sw, sw * H / W
    zz = max(1.0, z * punch)
    vw, vh = bw / zz, bh / zz
    x0 = min(max(focus[0] * sw - vw / 2, 0), sw - vw)
    y0 = min(max(focus[1] * sh - vh / 2, 0), sh - vh)
    return img.resize((W, H), Image.BICUBIC, box=(x0, y0, x0 + vw, y0 + vh))


def chip(text, size=38):
    font = ImageFont.truetype(FONT, size)
    bb = font.getbbox(text)
    padx, pady = 36, 18
    w, h = bb[2] - bb[0] + padx * 2, bb[3] - bb[1] + pady * 2
    im = Image.new("RGBA", (w + 16, h + 16), (0, 0, 0, 0))
    sh = Image.new("RGBA", im.size, (0, 0, 0, 0))
    ImageDraw.Draw(sh).rounded_rectangle([8, 12, w + 8, h + 12], radius=h // 2, fill=(0, 0, 0, 70))
    im.alpha_composite(sh.filter(ImageFilter.GaussianBlur(6)))
    d = ImageDraw.Draw(im)
    d.rounded_rectangle([8, 8, w + 8, h + 8], radius=h // 2, fill=(15, 15, 15, 235))
    d.text((8 + padx - bb[0], 8 + pady - bb[1]), text, font=font, fill=(255, 255, 255, 255))
    return im


def with_alpha(im, k):
    if k >= 0.995:
        return im
    a = np.asarray(im).copy()
    a[..., 3] = (a[..., 3] * max(0.0, k)).astype(np.uint8)
    return Image.fromarray(a, "RGBA")


class Renderer:
    def __init__(self, scenes, overlays, total, chip_y=200):
        self.scenes, self.total, self.chip_y = scenes, total, chip_y
        self.starts = [s["t0"] for s in scenes]
        self.ov = [dict(o, layer=chip(o["text"])) for o in overlays]

    def frame(self, t):
        t = min(max(t, 0), self.total - 1e-3)
        si = max(0, bisect.bisect_right(self.starts, t) - 1)
        sc = self.scenes[si]
        ds, dur = t - sc["t0"], sc["t1"] - sc["t0"]
        z = sc["z0"] + (sc["z1"] - sc["z0"]) * smooth(ds / dur)
        punch = 1.0
        if si:
            punch += 0.035 * math.exp(-ds / 0.12)
        for tp in sc["punches"]:
            d = t - tp
            if 0 <= d < 0.6:
                punch += 0.030 * math.exp(-d / 0.14)
        img = camera(sc["canvas"], z, sc["focus"], punch)
        for o in self.ov:
            if not o["from"] <= t < o["to"]:
                continue
            a = smooth((t - o["from"]) / 0.2) * smooth((o["to"] - t) / 0.2)
            k = back_out((t - o["from"]) / 0.35)
            lay = with_alpha(o["layer"], a)
            img.paste(lay, ((W - lay.width) // 2, self.chip_y + int((1 - min(k, 1.1)) * -40)), lay)
        return img

    def encode(self, wav, out_mp4, log_every=300):
        n = int(round(self.total * FPS))
        tmp = out_mp4 + ".tmp.mp4"
        cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
               "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-", "-i", wav, "-map", "0:v", "-map", "1:a",
               "-c:v", "libx264", "-crf", "19", "-maxrate", "9M", "-bufsize", "18M", "-preset", "veryfast",
               "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart",
               "-t", f"{self.total:.3f}", tmp]
        pr = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        try:
            for i in range(n):
                pr.stdin.write(self.frame(i / FPS).tobytes())
                if i % log_every == 0:
                    print(f"   frame {i}/{n}", flush=True)
            pr.stdin.close()
            if pr.wait() != 0:
                raise RuntimeError("ffmpeg failed")
        except BrokenPipeError:
            raise RuntimeError("ffmpeg stopped early")
        os.replace(tmp, out_mp4)
