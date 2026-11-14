"""Read the narrator timestamp file (the format your automation AI produces) and give word times.

Line format:  [00:07.65 --> 00:08.72]  Sounds terrifying,   <COMMA> @ 00:08.72
Words inside a segment are spread by letter count (accurate to about +/-0.2 s).
"""
import re

LINE = re.compile(r"^\[(\d+):(\d+\.\d+)\s*-->\s*(\d+):(\d+\.\d+)\]\s+(.*?)\s+<[^>]*>\s*@\s*[\d:.]+\s*$")


def wn(word):
    """word -> letters/digits only, so '$2,000,' == '2000'"""
    return re.sub(r"[^a-z0-9]", "", word.lower())


def _t(m, s):
    return int(m) * 60 + float(s)


def parse(path):
    segs = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            m = LINE.match(line.strip())
            if m:
                segs.append({"start": _t(m[1], m[2]), "end": _t(m[3], m[4]), "text": m[5].strip()})
    if not segs:
        raise RuntimeError("No timestamp lines found. Expected lines like: [00:00.00 --> 00:07.65]  text  <FULL STOP> @ 00:07.65")
    return segs


def words_from(segs):
    out = []
    for sg in segs:
        toks = sg["text"].split()
        wts = [max(2, len(wn(t))) + 1 for t in toks]
        total = float(sum(wts))
        t = sg["start"]
        for tok, w in zip(toks, wts):
            d = (sg["end"] - sg["start"]) * w / total
            if wn(tok):
                out.append({"w": tok, "n": wn(tok), "start": t, "end": t + d})
            t += d
    return out


def find(words, phrase, t0, t1):
    """start time of the first occurrence of phrase inside [t0, t1), else None"""
    toks = [wn(x) for x in phrase.split() if wn(x)]
    if not toks:
        return None
    for i in range(len(words) - len(toks) + 1):
        if words[i]["start"] < t0 - 0.01 or words[i]["start"] >= t1:
            continue
        if all(words[i + k]["n"] == toks[k] for k in range(len(toks))):
            return words[i]["start"]
    return None


def srt(segs):
    def f(t):
        ms = int(round(t * 1000))
        return f"{ms // 3600000:02d}:{ms // 60000 % 60:02d}:{ms // 1000 % 60:02d},{ms % 1000:03d}"
    cues, cur = [], None
    for sg in segs:
        if cur is None:
            cur = dict(sg)
        else:
            cur["end"], cur["text"] = sg["end"], cur["text"] + " " + sg["text"]
        if cur["end"] - cur["start"] >= 1.2:
            cues.append(cur)
            cur = None
    if cur:
        cues.append(cur)
    return "".join(f"{i}\n{f(c['start'])} --> {f(c['end'])}\n{c['text']}\n\n" for i, c in enumerate(cues, 1))
