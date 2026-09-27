"""Anime-domain frozen-feature classifier experiment (offline, episode-1 dev).

Trains a small pooled-feature classifier on user-verified spans only:
positives = the 24 accepted clips of the verified batch; negatives = the
9 user-flagged error regions plus the user-supplied exclusion reference
clips.  Reports separation on the verified sets and scores the five
identity-limited recall gaps plus known-other dev spans.  This is an
episode-1 domestic experiment: no cross-scene claim is made.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from extractor.audio import load_mono  # noqa: E402

BATCH = "20260927_182743_batch_b382e7"
WORK = ROOT / "work" / f"{BATCH}_001"
STEM = WORK / "stems" / "target_vocals.wav"

GAP_FRAGMENTS = {  # dev set: user-confirmed target tails/heads lost to identity gates
    "gap_308_head": (308.26, 309.405),
    "gap_528_tail": (528.61, 531.13),
    "gap_643_head": (643.55, 644.75),
    "gap_1046_tail": (1046.14, 1048.04),
    "gap_1241_head": (1241.17, 1242.64),
}
DEV_OTHER = {  # spans rejected as other by exclusion audit (semi-verified)
    "dev_448_after": (448.91, 450.45),
    "dev_439_overlap": (439.26, 443.43),
    "dev_531_mixed": (531.41, 535.67),
}
WINDOW = 0.5
STRIDE = 0.25
SEED = 20260927


def frame_features(model, processor, device, wave):
    data = processor(wave.numpy(), sampling_rate=16000, return_tensors="pt")
    enc = model.wavlm(data.input_values.to(device), output_hidden_states=model.config.use_weighted_layer_sum,
                      return_dict=True)
    if model.config.use_weighted_layer_sum:
        layers = torch.stack(enc.hidden_states, dim=1)
        w = model.layer_weights.softmax(dim=0)[None, :, None, None]
        f = (layers * w).sum(dim=1)
    else:
        f = enc.last_hidden_state
    f = model.projector(f)
    for layer in model.tdnn:
        f = layer(f)
    return f[0]


def pooled(frames):
    return torch.cat([frames.mean(dim=0), frames.max(dim=0).values, frames.std(dim=0)])


def windows_for(span, stem, sr):
    start, end = span
    a, b = int(start * sr), int(end * sr)
    seg = stem[a:b]
    if seg.numel() < int(WINDOW * sr):
        return []
    w = int(WINDOW * sr)
    hop = int(STRIDE * sr)
    out = []
    pos = 0
    while pos + w <= seg.numel():
        out.append(seg[pos:pos + w])
        pos += hop
    return out


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    sr = 16000
    stem = load_mono(STEM, sr)

    manifest = json.load(open(ROOT / "output" / BATCH / "batch_manifest.json", encoding="utf-8"))
    positives = [(r["start"], r["end"]) for r in manifest["sentences"] if r["accepted"]]
    review = json.load(open(ROOT / "evaluation" / "episode1_user_review_20260926.json", encoding="utf-8"))
    flagged = [tuple(map(float, c["span"])) for c in review["flagged_outputs"]]
    baseline = json.load(open(ROOT / "output" / "20260825_174500_batch_d90730" / "batch_manifest.json", encoding="utf-8"))
    baseline_spans = [(r["start"], r["end"]) for r in baseline["sentences"] if r["accepted"]]

    def subtract(a, b):
        out = [(max(a[0], s), min(a[1], e)) for s, e in b if min(a[1], e) > max(a[0], s)]
        out.sort()
        merged = []
        for s, e in out:
            if merged and s <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(merged[-1][1], e))
            else:
                merged.append((s, e))
        pieces, cursor = [], a[0]
        for s, e in merged:
            if s - cursor > 0.05:
                pieces.append((cursor, s))
            cursor = max(cursor, e)
        if a[1] - cursor > 0.05:
            pieces.append((cursor, a[1]))
        return pieces

    flagged_clean = [piece for span in flagged for piece in subtract(span, baseline_spans + positives)]
    print("flagged negative pieces after removing verified overlap:", len(flagged_clean))

    from transformers import Wav2Vec2FeatureExtractor, WavLMForXVector
    directory = ROOT / "model" / "speaker" / "wavlm-base-plus-sv"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = WavLMForXVector.from_pretrained(str(directory), local_files_only=True).to(device).eval()
    processor = Wav2Vec2FeatureExtractor.from_pretrained(str(directory), local_files_only=True)
    model.requires_grad_(False)

    def build(spans, label):
        feats, labels = [], []
        for span in spans:
            for w in windows_for(span, stem, sr):
                fr = frame_features(model, processor, device, w)
                feats.append(pooled(fr))
                labels.append(label)
        return feats, labels

    train_x, train_y = [], []
    px, py = build(positives, 1.0)
    ref_clips = sorted((WORK / "reference_voice_clips").glob("*"))
    ref_clips = [p for p in ref_clips if p.suffix.lower() in {".wav", ".mp3", ".flac"}]
    for clip in ref_clips:
        wave = load_mono(clip, sr)
        for pos in range(0, max(1, wave.numel() - int(WINDOW * sr) + 1), int(STRIDE * sr)):
            w = wave[pos:pos + int(WINDOW * sr)]
            if w.numel() < int(WINDOW * sr):
                break
            fr = frame_features(model, processor, device, w)
            px.append(pooled(fr))
            py.append(1.0)
    print(f"reference positive windows added from {len(ref_clips)} clips")
    nx, ny = build(flagged_clean, 0.0)
    train_x, train_y = px + nx, py + ny
    print(f"windows: positive={len(px)} negative={len(ny)}")

    exclude_clips = []
    for group in sorted((WORK / "negative_reference_voice_clips").glob("role_*")):
        exclude_clips.extend(sorted(p for p in group.glob("*") if p.suffix.lower() in {".wav", ".mp3", ".flac"}))
    for clip in exclude_clips:
        wave = load_mono(clip, sr)
        for pos in range(0, max(1, wave.numel() - int(WINDOW * sr) + 1), int(STRIDE * sr)):
            w = wave[pos:pos + int(WINDOW * sr)]
            if w.numel() < int(WINDOW * sr):
                break
            fr = frame_features(model, processor, device, w)
            train_x.append(pooled(fr))
            train_y.append(0.0)
    print(f"train total: {len(train_x)} (exclusion clips: {len(exclude_clips)})")

    X = torch.stack(train_x)
    y = torch.tensor(train_y, device=device)
    dim = X.shape[1]
    torch.manual_seed(SEED)
    clf = torch.nn.Sequential(
        torch.nn.Linear(dim, 256), torch.nn.ReLU(),
        torch.nn.Linear(256, 1),
    ).to(device)
    opt = torch.optim.Adam(clf.parameters(), lr=1e-3, weight_decay=1e-4)
    Xd, yd = X.to(device), y
    for epoch in range(200):
        opt.zero_grad()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(clf(Xd).squeeze(-1), yd)
        loss.backward()
        opt.step()
        if epoch in (0, 49, 99, 199):
            with torch.no_grad():
                score = clf(Xd).squeeze(-1)
                pos_mean = score[yd == 1].mean().item()
                neg_mean = score[yd == 0].mean().item()
                acc = ((score > 0).float() == yd).float().mean().item()
            print(f"epoch {epoch}: loss={loss.item():.4f} pos={pos_mean:.3f} neg={neg_mean:.3f} acc={acc:.3f}")

    import io
    out_dir = ROOT / "models" / "anime_t3"
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "state_dict": clf.state_dict(),
        "input_dim": dim,
        "threshold": 2.0,
        "window_seconds": WINDOW,
        "stride_seconds": STRIDE,
        "feature_version": "wavlm-tdnn-unpooled-v1",
        "training_span_count": len(train_x),
        "seed": SEED,
    }
    buf = io.BytesIO()
    torch.save(payload, buf)
    (out_dir / "classifier.pt").write_bytes(buf.getvalue())
    print("classifier saved:", out_dir / "classifier.pt", (out_dir / "classifier.pt").stat().st_size, "bytes")

    def score_span(name, span):
        ws = windows_for(span, stem, sr)
        if not ws:
            print(f"  {name}: no windows")
            return
        with torch.no_grad():
            feats = torch.stack([pooled(frame_features(model, processor, device, w)) for w in ws]).to(device)
            s = clf(feats).squeeze(-1)
        print(f"  {name}: mean={s.mean().item():+.3f} min={s.min().item():+.3f} max={s.max().item():+.3f} windows={len(ws)}")

    print("== gap fragments (dev: user says target) ==")
    for name, span in GAP_FRAGMENTS.items():
        score_span(name, span)
    print("== known-other dev spans ==")
    for name, span in DEV_OTHER.items():
        score_span(name, span)
    print("== verified positives (sanity, in-train) ==")
    for name, span in list({f"pos_{i}": s for i, s in enumerate(positives)}.items())[:4]:
        score_span(name, span)


if __name__ == "__main__":
    main()
