"""Synthesised sound effects (no downloads, no licences) + final audio mix with music ducking."""
import subprocess
import wave

import numpy as np

SR = 44100
_rng = np.random.default_rng(7)


def _t(d):
    return np.arange(int(d * SR)) / SR


def _lp(x, c):
    c = np.broadcast_to(c, x.shape)
    y = np.empty_like(x)
    a = 0.0
    for i in range(len(x)):
        a += c[i] * (x[i] - a)
        y[i] = a
    return y


def _norm(x, peak=1.0):
    m = np.max(np.abs(x)) or 1.0
    return (x / m * peak).astype(np.float32)


def hit():
    t = _t(0.9)
    f = 38 + 60 * np.exp(-t * 6)
    body = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t * 4.5)
    click = _rng.normal(0, 1, len(t)) * np.exp(-t * 45) * 0.5
    return _norm(body + click)


def whoosh(d=0.6, up=True):
    t = _t(d)
    n = _rng.normal(0, 1, len(t))
    sweep = np.linspace(0.015, 0.3, len(t))
    if not up:
        sweep = sweep[::-1]
    y = _lp(n, sweep) - _lp(n, sweep * 0.2)
    env = np.sin(np.pi * t / d) ** 2
    return _norm(y * env)


def swish():
    return whoosh(0.2, True)


def riser(d=1.3):
    t = _t(d)
    n = _rng.normal(0, 1, len(t))
    y = _lp(n, np.linspace(0.01, 0.35, len(t)))
    tone = np.sin(2 * np.pi * np.cumsum(180 + 1100 * (t / d) ** 2) / SR) * 0.5
    env = (t / d) ** 1.6
    return _norm((_norm(y) + tone) * env)


def tick():
    t = _t(0.03)
    return _norm(np.sin(2 * np.pi * 2300 * t) * np.exp(-t * 180))


def bell():
    t = _t(1.6)
    y = sum(np.sin(2 * np.pi * f * t) * a for f, a in ((660, 1), (990, .6), (1320, .4), (1980, .2)))
    return _norm(y * np.exp(-t * 3.2) * np.minimum(1, t * 400))


def coin():
    t = _t(0.7)
    y = np.zeros(len(t))
    for off, f in ((0.0, 2637), (0.09, 3951)):
        tt = np.maximum(t - off, 0)
        y += np.sin(2 * np.pi * f * tt) * np.exp(-tt * 13) * (t >= off)
    return _norm(y)


def heartbeat():
    t = _t(0.7)
    y = np.zeros(len(t))
    for off, a in ((0.0, 1.0), (0.26, 0.7)):
        tt = np.maximum(t - off, 0)
        y += a * np.sin(2 * np.pi * 52 * tt) * np.exp(-tt * 13) * (t >= off)
    return _norm(y)


def cashfly():
    return whoosh(0.45, False)


SOUNDS = {"hit": hit, "whoosh": whoosh, "swish": swish, "riser": riser, "tick": tick, "bell": bell,
          "coin": coin, "heartbeat": heartbeat, "cashfly": cashfly}
DURS = {"hit": 0.9, "whoosh": 0.6, "swish": 0.2, "riser": 1.3, "tick": 0.03, "bell": 1.6,
        "coin": 0.7, "heartbeat": 0.7, "cashfly": 0.45}
GAINS = {"hit": 0.55, "whoosh": 0.30, "swish": 0.20, "riser": 0.22, "tick": 0.20, "bell": 0.22,
         "coin": 0.22, "heartbeat": 0.50, "cashfly": 0.28}


def _decode(path):
    p = subprocess.run(["ffmpeg", "-v", "error", "-i", path, "-f", "f32le", "-ac", "1",
                        "-ar", str(SR), "pipe:1"], capture_output=True, check=True)
    return np.frombuffer(p.stdout, dtype=np.float32).copy()


def _smooth(x, sec):
    n = max(1, int(sec * SR))
    c = np.cumsum(np.insert(x, 0, 0))
    y = (c[n:] - c[:-n]) / n
    return np.concatenate([np.full(n - 1, y[0]), y])[:len(x)] if len(y) else x


def mix(total, voice, events, music_path=None):
    """voice: full-length array; events: [(time_sec, name, gain_scale)] -> float32 mono."""
    n = int(total * SR)
    out = np.zeros(n, np.float32)
    v = np.zeros(n, np.float32)
    v[:min(n, len(voice))] = voice[:n]
    pk = np.max(np.abs(v)) or 1.0
    out += v / pk * 0.88
    cache = {}
    for t, name, g in events:
        if name not in SOUNDS:
            continue
        s = cache.setdefault(name, SOUNDS[name]())
        i0 = int(t * SR)
        if i0 < 0:
            s2, i0 = s[-i0:], 0
        else:
            s2 = s
        seg = s2[:max(0, n - i0)]
        out[i0:i0 + len(seg)] += seg * GAINS[name] * g
    if music_path:
        m = _decode(music_path)
        if len(m):
            m = np.tile(m, int(np.ceil(n / len(m))))[:n]
            env = _smooth(np.abs(v / pk), 0.25)
            gate = np.clip(env / 0.05, 0, 1)
            gain = _smooth(0.18 - 0.12 * gate, 0.2)
            fade = np.minimum(1, np.minimum(np.arange(n), n - np.arange(n)) / (0.9 * SR))
            out += m / (np.max(np.abs(m)) or 1) * gain * fade
    out = np.tanh(out * 1.15) / np.tanh(1.15)
    pk = np.max(np.abs(out)) or 1
    if pk > 0.93:
        out *= 0.93 / pk
    return out.astype(np.float32)


def write_wav(path, a):
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes((np.clip(a, -1, 1) * 32767).astype(np.int16).tobytes())
