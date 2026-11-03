"""Local identity and contamination guards for extraction candidates."""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch

from .acoustic_qc import estimate_voiced_f0_stats, snap_crop_boundaries_to_valley
from .audio import load_mono
from .types import TimeSpan

LOGGER = logging.getLogger(__name__)


def subtitle_atomic_spans(guide, spans):
    """Keep each cue handoff as a boundary even when VAD found no pause."""
    if guide is None:
        return list(spans)
    cues = sorted({(interval.start, interval.end)
                   for cue in guide.cues if cue.kind == "speech"
                   for interval in [guide.cue_span(cue) or TimeSpan(
                       max(0., cue.start + guide.offset), max(0., cue.end + guide.offset))]})
    boundaries = set()
    for left, right in zip(cues, cues[1:]):
        if abs(left[0] - right[0]) <= .18 and abs(left[1] - right[1]) <= .18:
            continue
        if right[0] >= left[1]:
            boundaries.add((left[1] + right[0]) / 2)
        else:
            boundaries.update((right[0], min(left[1], right[1])))
    active = []
    for start, end in cues:
        active = [(a, b) for a, b in active if b > start]
        for previous_start, previous_end in active:
            if abs(previous_start - start) > .18 or abs(previous_end - end) > .18:
                boundaries.update((start, min(previous_end, end)))
        active.append((start, end))
    result = []
    for span in spans:
        edges = [span.start, *sorted(edge for edge in boundaries
                                    if span.start + 1e-6 < edge < span.end - 1e-6), span.end]
        result.extend(TimeSpan(start, end) for start, end in zip(edges, edges[1:]))
    return result


def carve_tainted_spans(spans, tainted, waveform=None, minimum_seconds=.35, padding=.03):
    """Subtract padded detector intervals without sacrificing a clean parent."""
    dirty = []
    for span in sorted(tainted, key=lambda item: (item.start, item.end)):
        if span.end <= span.start:
            continue
        start, end = max(0., span.start - padding), span.end + padding
        if dirty and start <= dirty[-1].end:
            dirty[-1] = TimeSpan(dirty[-1].start, max(end, dirty[-1].end))
        else:
            dirty.append(TimeSpan(start, end))
    clean, removed = [], []
    for span in spans:
        intersections = [TimeSpan(max(span.start, part.start), min(span.end, part.end))
                         for part in dirty if part.start < span.end and part.end > span.start]
        if not intersections:
            clean.append(span)
            continue
        cursor = span.start
        retained = []
        for part in [*intersections, TimeSpan(span.end, span.end)]:
            start, end = cursor, part.start
            if end - start >= minimum_seconds - 1e-9:
                if waveform is not None:
                    snapped_start, snapped_end = snap_crop_boundaries_to_valley(
                        waveform, start, end, hard_bounds=(start, end), pad_seconds=0.)
                    if snapped_end - snapped_start >= minimum_seconds - 1e-9:
                        start, end = snapped_start, snapped_end
                retained.append(TimeSpan(start, end))
            cursor = max(cursor, part.end)
        clean.extend(retained)
        cursor = span.start
        for part in [*retained, TimeSpan(span.end, span.end)]:
            if part.start > cursor:
                removed.append(TimeSpan(cursor, part.start))
            cursor = part.end
    return clean, removed


class VocalTractGeometryGuard:
    """A bounded geometry prior with a union of reference-register ellipsoids.

    F0 outside the reference envelope increases measurement uncertainty. Only
    sustained extreme geometry AND an octave contradiction can protect on their
    own; neither measurement ever confers target identity.
    """

    def __init__(self, reference_paths):
        from .nextgen.physiology import vocal_tract_frames
        self.fit([vocal_tract_frames(load_mono(path, 16000)) for path in reference_paths])

    def fit(self, registers):
        self.models, self.reference = [], {}
        usable = [np.asarray(rows) for rows in registers if len(rows) >= 10]
        if not usable:
            return
        all_rows = np.concatenate(usable)
        envelope = np.quantile(all_rows[:, 3], [.05, .95])
        self.envelope = envelope
        self.reference = {"f0_median_hz": float(np.exp(np.median(all_rows[:,3]))),
                          "f0_low_hz": float(np.exp(envelope[0])),
                          "f0_high_hz": float(np.exp(envelope[1])),
                          "voiced_seconds": len(all_rows)*.02}
        for rows in [all_rows, *usable]:
            geometry = rows[:, :3]
            low, high = np.quantile(geometry, [.05, .95], axis=0)
            geometry = np.clip(geometry, low, high)
            center = np.median(geometry, axis=0)
            # LPC uncertainty remains nonzero even for a sustained identical vowel.
            covariance = np.cov(geometry, rowvar=False)
            covariance = .5 * covariance + .5 * np.diag(np.diag(covariance))
            covariance += np.diag(np.array([.12, .10, .18]) ** 2)
            inverse = np.linalg.inv(covariance)
            distances = np.einsum('ij,jk,ik->i', geometry-center, inverse, geometry-center)
            self.models.append((center, inverse, max(16.27, float(np.quantile(distances, .995)))))

    def assess_frames(self, rows):
        rows = np.asarray(rows)
        if not self.models or len(rows) < 40 or self.reference.get('voiced_seconds', 0.) < .8:
            return {"available": False, "mismatch": False, "reason": "insufficient_geometry"}
        outside = np.maximum(0., np.maximum(self.envelope[0]-rows[:,3], rows[:,3]-self.envelope[1]))
        inflation = 1. + (outside / np.log(2.))**2
        relative = []
        for center, inverse, threshold in self.models:
            delta = rows[:,:3]-center
            relative.append(np.einsum('ij,jk,ik->i', delta,inverse,delta) / inflation / threshold)
        distances = np.min(relative, axis=0)
        mismatch = bool(np.mean(distances > 1.) >= .8)
        return {"available": True, "mismatch": mismatch,
                "extreme_conflict": bool(len(rows) >= 80 and np.mean(
                    (distances > 4.) & (outside > np.log(2.))) >= .9),
                "log_prior": -float(np.log(2.) * np.clip(np.median(distances)-1.,0.,1.)) if mismatch else 0.,
                "mahalanobis_relative": float(np.median(distances)),
                "outside_fraction": float(np.mean(distances > 1.)),
                "register_variance_inflation": float(np.median(inflation)),
                "geometry_seconds": len(rows)*.02}

    def evaluate(self, waveform):
        from .nextgen.physiology import vocal_tract_frames
        result = self.assess_frames(vocal_tract_frames(waveform))
        # Preserve old diagnostics consumed by Worker and manifests.
        return {"pitch_register": estimate_voiced_f0_stats(waveform),
                "reference_pitch_register": self.reference,
                "pitch_register_mismatch": result.get('extreme_conflict', False), "vocal_tract_geometry": result}

    @staticmethod
    def condition(diagnostics, acoustic_margins=(), margin_scale=.06):
        """Geometry may adjust an acoustic decision by at most log(2) odds.

        Ordinary deviations require two direct voice margins. A sustained,
        extreme geometry AND octave contradiction may independently reject;
        neither pitch alone nor geometry alone can establish identity.
        """
        result = dict(diagnostics)
        geometry = result.get('vocal_tract_geometry') or {}
        margins = [float(v) for v in acoustic_margins
                   if isinstance(v,(int,float)) and np.isfinite(v)]
        prior = float(np.clip(geometry.get('log_prior',0.),-np.log(2.),0.))
        result['pitch_register_mismatch'] = bool(geometry.get('extreme_conflict') or (
            geometry.get('mismatch') and len(margins)==2
            and np.mean(margins) / max(float(margin_scale),1e-6) + prior <= 0.))
        result['geometry_acoustic_margins'] = margins
        return result


PitchRegisterGuard = VocalTractGeometryGuard


class SubtitleJoinGuard:
    """Require independent local evidence before a join crosses subtitle cues."""

    def __init__(self, models_root: Path, reference_paths, raw_reference_paths, *, encoders=None):
        self.models_root = models_root
        self.reference_paths = {"stem": list(reference_paths), "raw": list(raw_reference_paths)}
        self.encoders = None
        self._shared_encoders = encoders
        self.references = {}
        self.floors = {}
        self.available = True
        self.cache = {}

    def _load(self):
        if self.encoders is not None or not self.available:
            return self.available
        from .nextgen.anime_embedding import AnimeSpeakerOnnx

        self.encoders = {}
        try:
            for variant in ("char", "va"):
                encoder = (self._shared_encoders[("char", "va").index(variant)]
                           if self._shared_encoders is not None else AnimeSpeakerOnnx(self.models_root, variant))
                self.encoders[variant] = encoder
                for channel, paths in self.reference_paths.items():
                    self.references[(variant, channel)] = np.stack([
                        encoder.encode(load_mono(path, 16000), sample_rate=16000) for path in paths])
                    references = self.references[(variant, channel)]
                    floor = .32 if variant == "char" else .20
                    if len(references) >= 2:
                        dots = references @ references.T
                        off_diagonal = dots[~np.eye(len(references), dtype=bool)]
                        floor = float(np.quantile(off_diagonal, .20)) - .04
                    bounds = (.28, .45) if variant == "char" else (.18, .35)
                    self.floors[(variant, channel)] = float(np.clip(floor, *bounds))
        except Exception as exc:
            LOGGER.warning("Subtitle joins disabled: anime identity evidence unavailable: %s", exc)
            self.close()
            self.available = False
        return self.available

    def allows(self, left, right, waveform, raw_waveform, verifier, profile, threshold):
        # A long accepted turn must not lend its embedding to an uncertain edge.
        local = (TimeSpan(max(left.start, left.end - 1.8), left.end),
                 TimeSpan(right.start, min(right.end, right.start + 1.8)))
        key = tuple((item.start, item.end) for item in local)
        if key in self.cache:
            return self.cache[key]
        self.cache[key] = False
        if raw_waveform is None or min(item.duration for item in local) < .6:
            return False

        def parts(source):
            return [source[round(item.start * 16000):round(item.end * 16000)] for item in local]

        stem_parts, raw_parts = parts(waveform), parts(raw_waveform)
        if any(abs(a.numel() - b.numel()) > 160 for a, b in zip(stem_parts, raw_parts)):
            return False
        for span, stem, raw in zip(local, stem_parts, raw_parts):
            match = verifier.verify_dual_channel_waveform(
                stem, raw, profile, threshold, duration=span.duration,
                window_seconds=min(1., span.duration), hop_seconds=.5,
                clean_gate=True, allow_raw_rescue=False,
                audit_raw_on_uvr_accept=True, audit_raw_channel=True)
            decisions = (match.primary, match.secondary, match.raw_primary, match.raw_secondary)
            if not match.accepted or any(item is None or not item.accepted for item in decisions):
                return False
        verifier._ensure_secondary(profile)
        for model in (verifier.primary, verifier.secondary):
            pair_floor = .58 if model is verifier.primary else .48
            for views in (stem_parts, raw_parts):
                vectors = model._embeddings_from_waveforms(views)
                vectors = torch.nn.functional.normalize(vectors, dim=-1)
                if float(vectors[0] @ vectors[1]) < pair_floor:
                    return False
        if self._shared_encoders is None:
            self._shared_encoders = verifier.anime_pair()
        if not self._load():
            return False
        passed_views = set()
        for variant, encoder in self.encoders.items():
            for channel, views in (("stem", stem_parts), ("raw", raw_parts)):
                try:
                    vectors = np.stack([encoder.encode(part, sample_rate=16000) for part in views])
                except ValueError:
                    continue
                scores = self.references[(variant, channel)] @ vectors.T
                reference_scores = np.sort(scores, axis=0)[-2:].mean(axis=0)
                floor = self.floors[(variant, channel)]
                if (float(vectors[0] @ vectors[1]) >= floor - .04
                        and np.all(reference_scores >= floor)):
                    passed_views.add((variant, channel))
        allowed = len(passed_views) >= 3
        self.cache[key] = allowed
        return allowed

    def close(self):
        if self._shared_encoders is None:
            for encoder in (self.encoders or {}).values():
                encoder.close()
        self.encoders = None
        self.references.clear()
        self.floors.clear()
