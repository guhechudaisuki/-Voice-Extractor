"""Integrate cached identity evidence and decode locally supported speech intervals."""
from __future__ import annotations

from bisect import bisect_left, bisect_right
from dataclasses import dataclass
import math
from statistics import NormalDist, median
import sys


def _finite(value):
    return isinstance(value, (int, float)) and math.isfinite(value)


def _span(value):
    if (isinstance(value, (list, tuple)) and len(value) == 2
            and all(_finite(item) for item in value) and value[1] > value[0]):
        return float(value[0]), float(value[1])
    return None


def _scale(values, center=0.):
    """Gaussian-consistent MAD, with an RMS fallback for constant samples."""
    if not values:
        return 1.
    location = median(values)
    deviation = median(abs(value - location) for value in values)
    if deviation > 0:
        return deviation / NormalDist().inv_cdf(.75)
    return max(math.sqrt(sum((value - center) ** 2 for value in values) / len(values)),
               math.sqrt(sys.float_info.epsilon))


@dataclass(frozen=True)
class _Observation:
    start: float
    end: float
    value: float
    kind: str

    @property
    def center(self):
        return (self.start + self.end) / 2


@dataclass(frozen=True)
class _Cell:
    start: float
    end: float
    left: float
    right: float
    observed: bool
    identity: bool
    blocked: bool

    @property
    def integral(self):
        return (self.end - self.start) * (self.left + self.right) / 2


class IdentityTrajectory:
    """A signed evidence field; scores are not calibrated posterior probabilities.

    Reference floors define zero evidence. Robust episode scales put channels on
    comparable axes. Overlapping observations are averaged within each family,
    so duplicating a window cannot increase its vote. Fine evidence adjusts a
    coarse identity prior at its own temporal resolution.
    """

    def __init__(self, local, profile_floors, source_pool, blocked_spans=()):
        groups = local.values() if isinstance(local, dict) else (local or (),)
        rows = [row for group in groups if isinstance(group, (list, tuple))
                for row in group if isinstance(row, dict) and _span(row.get("span"))]
        self.rows = rows
        self.floors = profile_floors or {}
        self.observations = []
        self.blocked = [span for item in blocked_spans if (span := _span(item))]
        self.fine_sources = []
        self.precise_negative_spans = []
        self._coarse(rows)
        self._sources(source_pool or ())
        self.observations = sorted(set(self.observations), key=lambda row: row.start)
        self.blocked = sorted(set(self.blocked))
        self._starts = [row.start for row in self.observations]
        self._prefix_ends = []
        for row in self.observations:
            self._prefix_ends.append(max(row.end,self._prefix_ends[-1] if self._prefix_ends else row.end))
        self._source_bounds = [(min(w["start"] for w in audit.get("windows",()) if _finite(w.get("start"))),
                                max(w["end"] for w in audit.get("windows",()) if _finite(w.get("end"))),audit)
                               for audit in self.fine_sources if any(_finite(w.get("start")) and _finite(w.get("end"))
                                                                    for w in audit.get("windows",()))]
        coarse_widths = [row.end - row.start for row in self.observations
                         if row.kind == "coarse"]
        micro_widths = [row.end - row.start for row in self.observations if row.kind == "micro"]
        widths = coarse_widths or micro_widths or [row.end - row.start for row in self.observations]
        self.resolution = median(widths) if widths else 1.
        coarse = [row for row in self.observations if row.kind == "coarse"]
        differences = [abs(right.value - left.value) for left, right in zip(coarse, coarse[1:])
                       if right.start < left.end and right.start > left.start
                       and right.value * left.value > 0]
        self.noise = (median(differences) / (NormalDist().inv_cdf(.75) * math.sqrt(2))
                      if differences else 0.)
        self._micro_barriers = self._resolved_micro_barriers()
        self._field = self._build_field()
        self._ends = [cell.end for cell in self._field]
        self._cache = {}
        self._assessments = {}
        self._direction_scales = {}
        for channel in ("primary", "secondary"):
            angles = [math.acos(max(-1., min(1., row["direction_to_previous"][channel])))
                      for row in rows
                      if _finite((row.get("direction_to_previous") or {}).get(channel))]
            if angles:
                self._direction_scales[channel] = (median(angles), _scale(angles))

    def _coarse(self, rows):
        channels = (("primary", "primary"), ("secondary", "secondary"),
                    ("tertiary", "tertiary"), ("anime_char_score", "anime_char"),
                    ("anime_va_score", "anime_va"), ("cm_p", None), ("cm_s", None))
        calibrated = []
        for name, floor_name in channels:
            floor = self.floors.get(floor_name) if floor_name else 0.
            values = [float(row[name]) for row in rows if _finite(row.get(name))]
            if not values or not _finite(floor):
                continue
            scale = _scale(values, floor)
            calibrated.append((name, float(floor), scale))
        for row in rows:
            kind = "micro" if row.get("scale") == "micro" else "coarse"
            if kind == "micro" and row.get("micro_state") == "uncertain":
                self.observations.append(_Observation(*_span(row["span"]), 0., kind))
                continue
            detail = row.get("identity_evidence") or {}
            if (kind == 'coarse' and not row.get('accepted')
                    and all(_finite(row.get(c)) and row[c] < self.floors.get(c, 0.)-.15
                            for c in ('primary','secondary'))
                    and all(_finite(row.get(c)) and row[c] < 0. for c in ('cm_p','cm_s'))):
                self.observations.append(_Observation(*_span(row['span']), 0., 'coarse_boundary'))
            if row.get("accepted") and detail.get("accepted"):
                self.observations.append(_Observation(*_span(row["span"]), 0., "consensus"))
            channels = detail.get("channel_evidence") or {}
            identity = [channels[name] for name in ("primary", "secondary")
                        if _finite(channels.get(name))]
            fused = sum(identity) / len(identity) if identity else None
            raw = detail.get('raw_channel_evidence') or {}
            if row.get('accepted') and row.get('reference_consensus'):
                # Direct dual-reference certification remains independent of a
                # noisy episode calibration. It still needs positive measured
                # acoustic margins; a flag alone contributes no evidence.
                names = ('secondary','tertiary') if detail.get('reference_only_consensus') else ('primary','secondary')
                direct = [raw.get(name) for name in names]
                if all(_finite(value) and value > 0. for value in direct):
                    fused = max(fused or 0., min(direct))
            if detail.get("accepted") and _finite(detail.get("appearance_evidence")):
                fused = max(fused or 0., detail["appearance_evidence"])
            duration_evidence = row.get('duration_reference_evidence')
            if (kind == 'micro' and row.get('accepted') and row.get('micro_state') == 'target'
                    and _finite(duration_evidence) and duration_evidence > 0.):
                fused = max(fused or 0., duration_evidence)
            if _finite(fused):
                fused = math.asinh(float(fused))
                if not row.get("accepted"):
                    fused = min(0., fused)
                self.observations.append(_Observation(*_span(row["span"]), fused, kind))
                continue
            evidence = []
            for name, floor, scale in calibrated:
                if not _finite(row.get(name)):
                    continue
                value = math.asinh((float(row[name]) - floor) / scale)
                # Anime references support register changes without vetoing them.
                if name.startswith("anime_"):
                    value = max(0., value)
                evidence.append(value)
            if evidence:
                value = sum(evidence) / len(evidence)
                if not row.get("accepted"):
                    value = min(0., value)
                self.observations.append(_Observation(*_span(row["span"]), value, kind))
            else:
                self.observations.append(_Observation(*_span(row["span"]),
                    0., kind))  # A flag without acoustic scores is not identity evidence.

    def _sources(self, sources):
        frames = {}
        islands = []
        channel_scales = {
            name: _scale([float(row[name]) for row in self.rows if _finite(row.get(name))],
                         self.floors.get(name, 0.))
            for name in ("primary", "secondary")}
        for name in channel_scales:
            calibrated = []
            for row in self.rows:
                score = row.get(name)
                z = (row.get("identity_evidence") or {}).get("channel_evidence", {}).get(name)
                floor = self.floors.get(name)
                if all(_finite(value) for value in (score, z, floor)) and z:
                    scale = (score - floor) / z
                    if scale > 0:
                        calibrated.append(scale)
            if calibrated:
                channel_scales[name] = median(calibrated)
        for candidate in sources:
            diagnostics = getattr(candidate, "diagnostics", {}) or {}
            if diagnostics.get("excluded_role_rejected") or diagnostics.get("structural_hard_reject"):
                self.blocked.append((float(candidate.start), float(candidate.end)))
            for item in diagnostics.get("local_identity_evidence") or ():
                span = _span(item.get("span")) if isinstance(item, dict) else None
                if not span:
                    continue
                evidence = []
                for channel, score_name in (("primary", "eres_score"),
                                             ("secondary", "camplus_score")):
                    score = diagnostics.get(score_name)
                    floor = self.floors.get(channel)
                    margin = diagnostics.get("excluded_" + channel + "_direct_margin")
                    support = []
                    if _finite(score) and _finite(floor):
                        support.append((score - floor) / channel_scales[channel])
                    if _finite(margin):
                        support.append(margin / channel_scales[channel])
                    if support:
                        evidence.append(min(support))
                if evidence:
                    value = math.asinh(sum(evidence) / math.sqrt(len(evidence)))
                    self.observations.append(_Observation(*span, value, "local"))
            audit = diagnostics.get("local_classifier_audit") or {}
            threshold = audit.get("reject_threshold") if isinstance(audit, dict) else None
            if _finite(threshold):
                self.fine_sources.append(audit)
                strong = [row for row in audit.get("windows", ()) if isinstance(row, dict)
                          and row.get("voiced") and _finite(row.get("score"))
                          and row["score"] <= threshold and _span((row.get("start"),row.get("end")))]
                scores = [float(row["score"]) for row in audit.get("windows", ())
                          if isinstance(row,dict) and row.get("voiced") and _finite(row.get("score"))]
                mean = sum(scores) / len(scores) if scores else 0.
                width = median([row["end"]-row["start"] for row in strong]) if strong else 1.
                error = math.sqrt(sum((value-mean)**2 for value in scores) / max(1,len(scores))
                                  / max(1.,candidate.duration/width))
                if len(strong) >= 2 and mean - error <= 0.:
                    self.precise_negative_spans.extend((row["start"],row["end"]) for row in strong)
                for row in audit.get("windows") or ():
                    if not isinstance(row, dict) or not row.get("voiced"):
                        continue
                    span = _span((row.get("start"), row.get("end")))
                    if span and _finite(row.get("score")):
                        key = (*span, float(threshold), float(row["score"]))
                        frames[key] = float(row["score"]) / max(abs(float(threshold)), sys.float_info.epsilon)
            consensus = diagnostics.get("experimental_final_island_consensus") or {}
            direct = diagnostics.get("excluded_primary_direct_margin")
            if consensus.get("whole_conflict") and not consensus.get("suspect_islands") and _finite(direct):
                # An unlocalized conflict is one uncertain observation, not a mask.
                self.observations.append(_Observation(float(candidate.start), float(candidate.end),
                    -math.log1p(1. / max(abs(direct), sys.float_info.epsilon)), "source"))
            evidence = diagnostics.get("experimental_final_island_evidence") or {}
            if not isinstance(evidence, dict):
                continue
            source_islands = [row for row in evidence.get("islands", ())
                              if isinstance(row, dict) and _span(row.get("span"))]
            source_islands.sort(key=lambda row: row["span"][0])
            for index, island in enumerate(source_islands):
                start, end = _span(island["span"])
                if island.get("excluded_role_rejected"):
                    self.blocked.append((start, end))
                state = island.get("state")
                if state not in ("target_supported", "other_supported", "unresolved"):
                    continue
                margins = island.get("domain_margins") or {}
                values = [float(value) for value in margins.values() if _finite(value)]
                if state == "unresolved" and not values:
                    continue
                paired_views = [[margins.get(encoder + "_" + track)
                                 for track in ("stem", "raw")]
                                for encoder in ("char", "va")]
                domain_conflict = all(all(_finite(value) for value in pair)
                                      and max(pair) < 0. for pair in paired_views)
                corroborated = bool(values and median(values) < 0 and isinstance(audit,dict)
                                    and _finite(threshold) and sum(
                                        _finite(row.get("score")) and row["score"] <= threshold
                                        and row.get("voiced") and row.get("start",end) < end
                                        and row.get("end",start) > start
                                        for row in audit.get("windows", ())) >= 2)
                if state == "other_supported" or corroborated or domain_conflict:
                    previous = source_islands[index - 1] if index else None
                    following = source_islands[index + 1] if index + 1 < len(source_islands) else None
                    if previous and previous.get("state") == "target_supported" and previous["span"][1] <= start:
                        start = (previous["span"][1] + start) / 2
                    if following and following.get("state") == "target_supported" and end <= following["span"][0]:
                        end = (end + following["span"][0]) / 2
                    contained = [r for r in self.rows if r.get("accepted")
                                 and r.get("identity_evidence", {}).get("accepted")
                                 and start <= r["span"][0] and r["span"][1] <= end]
                    # A crossing wide window cannot overrule an island boundary.
                    overlapping = [r for r in self.rows if r.get("accepted")
                                   and min(end,r["span"][1]) > max(start,r["span"][0])]
                    robust = any(r.get("identity_evidence", {}).get("accepted")
                                 and (r.get("identity_evidence",{}).get("appearance_evidence") or 0.)
                                 > abs(median(values)) / _scale(values) if values else False
                                 for r in overlapping)
                    # LLR and raw source margins have different fitted scales.
                    # A louder calibrated score from a CROSSING window cannot
                    # erase agreement of both identity encoders on both tracks.
                    if (not contained and (domain_conflict or not robust)) or corroborated:
                        self.observations.append(_Observation(start, end, -math.inf, "boundary"))
                islands.append((start, end, state, median(values) if values else None))
        frame_scale = 1.
        for (start, end, _, _), value in frames.items():
            self.observations.append(_Observation(start, end,
                math.asinh(value / frame_scale), "fine"))
        island_scale = _scale([value for _, _, _, value in islands if value is not None])
        for start, end, state, value in islands:
            signed = 1. if state == "target_supported" else -1.
            strength = math.asinh(value / island_scale) if value is not None else signed
            self.observations.append(_Observation(start, end, strength, "island"))

    def _density(self, point, active):
        # A micro observation elsewhere in the episode must not weaken an
        # existing local classifier or island veto here.
        micro_widths = [row.end-row.start for row in active if row.kind == 'micro']
        resolution = median(micro_widths) if micro_widths else self.resolution
        families = {}
        for row in active:
            if row.kind in ("boundary", "coarse_boundary"):
                continue
            half = (row.end - row.start) / 2
            weight = max(sys.float_info.epsilon, 1 - abs(point - row.center) / half)
            total, mass, width = families.get(row.kind, (0., 0., 0.))
            value = row.value * resolution / (row.end - row.start) if row.kind == "source" else row.value
            families[row.kind] = (total + value * weight, mass + weight,
                                  width + (row.end - row.start) * weight)
        means = {kind: total / mass for kind, (total, mass, _) in families.items()}
        base = means.get("coarse", 0.)
        positive_coarse = [row.value for row in active if row.kind == "coarse" and row.value > 0.]
        if positive_coarse:
            base = max(positive_coarse)
        if "micro" in means:
            # An abstention or isolated phoneme cannot erase positive mesoscopic
            # identity. Resolved negative runs/edges are fenced in _build_field.
            micro = means["micro"]
            base = (base / (1. + abs(min(0., micro))) + max(0., micro)
                    if base > 0. else micro)
        if "source" in means:
            base += means["source"]
        if "local" in means:
            _, mass, width = families["local"]
            local = means["local"]
            if local < 0. and "consensus" in means and base > 0.:
                local /= 1. + base * base
            base += local * max(1., resolution / (width / mass))
        if "island" in means:
            island = means["island"]
            base += island / (1 + abs(base))
        if "fine" in means and ("coarse" in means or "island" in means):
            _, mass, width = families["fine"]
            precision = resolution / (width / mass)
            detail = means["fine"] if base > 0 else min(0., means["fine"])
            if -math.asinh(1.) < detail < 0. and base > 0.:
                detail /= 1. + base * base
            base += detail * (precision if detail <= -math.asinh(1.) else math.sqrt(precision))
        elif "fine" in means:
            base = min(0., means["fine"])
        return base

    def _resolved_micro_barriers(self):
        micro = sorted((row for row in self.observations if row.kind == 'micro'),
                       key=lambda row: (row.start, row.end))
        positive = []
        for row in sorted((r for r in self.observations if r.kind == 'coarse' and r.value > 0.),
                          key=lambda row: row.start):
            if positive and row.start <= positive[-1][1]:
                positive[-1][1] = max(positive[-1][1], row.end)
            else:
                positive.append([row.start, row.end])
        starts = [a for a, _ in positive]
        barriers = set()
        for index, row in enumerate(micro):
            if row.value >= 0.:
                continue
            adjacent = any(other.value < 0. and abs(other.start - row.start) > 1e-6
                           and max(row.start, other.start) <= min(row.end, other.end) + .02
                           for other in micro[max(0, index-1):index] + micro[index+1:index+2])
            region_index = bisect_right(starts, row.start) - 1
            interior = (region_index >= 0 and row.start > positive[region_index][0] + .02
                        and row.end < positive[region_index][1] - .02)
            if adjacent or not interior:
                barriers.add(row)
        return barriers

    def _build_field(self):
        points = {point for row in self.observations
                  for point in (row.start, row.center, row.end)}
        points.update(point for span in [*self.blocked, *self.precise_negative_spans] for point in span)
        points = sorted(points)
        cells = []
        active, cursor = [], 0
        for a, b in zip(points, points[1:]):
            while cursor < len(self.observations) and self.observations[cursor].start < b:
                active.append(self.observations[cursor])
                cursor += 1
            active = [row for row in active if row.end > a]
            barrier = (any(x < b and y > a for x, y in [*self.blocked, *self.precise_negative_spans])
                       or any(row.kind == "boundary" for row in active))
            identity = any(row.kind != "fine" and row.value > 0 for row in active)
            micro = [row for row in active if row.kind == "micro"]
            if micro:
                local = self._density((a + b) / 2, micro)
                coarse_positive = any(row.kind == 'coarse' and row.value > 0. for row in active)
                identity = local > 0. or coarse_positive
                barrier = barrier or any(row in self._micro_barriers for row in micro)
            elif any(row.kind == 'coarse_boundary' for row in active):
                barrier = True
            left, right = self._density(a, active), self._density(b, active)
            if barrier:
                left = right = 0.
            if left * right < 0:
                zero = a + (b - a) * abs(left) / (abs(left) + abs(right))
                if a < zero < b:
                    cells.append(_Cell(a, zero, left, 0., bool(active), identity, barrier))
                    a, left = zero, 0.
            cells.append(_Cell(a, b, left, right, bool(active), identity, barrier))
        return cells

    def _cells(self, start, end):
        key = (float(start), float(end))
        if key in self._cache:
            return self._cache[key]
        if not _span(key):
            return []
        cells = []
        index = bisect_right(self._ends, start)
        while index < len(self._field):
            cell = self._field[index]
            if cell.start >= end:
                break
            a, b = max(start, cell.start), min(end, cell.end)
            slope = (cell.right - cell.left) / (cell.end - cell.start)
            cells.append(_Cell(a, b, cell.left + (a - cell.start) * slope,
                              cell.left + (b - cell.start) * slope,
                              cell.observed, cell.identity, cell.blocked))
            index += 1
        self._cache[key] = cells
        return cells

    def _penalty(self, cells):
        observed = sum(cell.end - cell.start for cell in cells if cell.observed)
        effective = observed / self.resolution
        # One fitted onset and offset cost two half-log-sample BIC terms.
        return self.noise * self.resolution * math.log1p(effective)

    def assess(self, start, end):
        key = (float(start), float(end))
        if key in self._assessments:
            return self._assessments[key]
        cells = self._cells(start, end)
        duration = end - start
        integral = sum(cell.integral for cell in cells)
        positive = sum(cell.end - cell.start for cell in cells if cell.integral > 0)
        observed = sum(cell.end - cell.start for cell in cells if cell.observed)
        identity = sum(cell.end - cell.start for cell in cells if cell.identity)
        blocked = sum(cell.end - cell.start for cell in cells if cell.blocked)
        penalty = self._penalty(cells)
        negative = -sum(min(0., cell.integral) for cell in cells)
        probe = self._occlusion_probe(cells)
        relevant = self.observations[bisect_right(self._prefix_ends,start):bisect_left(self._starts,end)]
        edge_identity_conflict = any(
            row.kind == "local" and row.value < 0.
            and (row.start <= start < row.end or row.start < end <= row.end)
            for row in relevant)
        sources = [audit for a,b,audit in self._source_bounds if a < end and b > start]
        fine = [row for row in relevant if row.kind == "fine" and row.end > start]
        fine_mass = sum(min(end, row.end) - max(start, row.start) for row in fine)
        fine_mean = sum(row.value * (min(end, row.end) - max(start, row.start)) for row in fine) / fine_mass if fine_mass else 0.
        fine_variance = sum((row.value-fine_mean)**2 * (min(end, row.end)-max(start,row.start)) for row in fine) / fine_mass if fine_mass else 0.
        fine_width = median([row.end-row.start for row in fine]) if fine else 1.
        fine_error = math.sqrt(fine_variance / max(1., duration / fine_width))
        strong_negative = any(sum(
            isinstance(row.get("score"), (int, float)) and row.get("voiced")
            and row["score"] <= audit["reject_threshold"]
            and min(end, row.get("end", start)) - max(start, row.get("start", end))
                >= (row.get("end", 0.) - row.get("start", 0.)) / 2
            for row in audit.get("windows", ())) >= 2 for audit in sources)
        fine_conflict = bool(strong_negative and fine_mean - fine_error <= 0.)
        fine_covered, cursor = 0., start
        for row in sorted(fine, key=lambda row: row.start):
            fine_covered += max(0., min(end,row.end)-max(cursor,row.start))
            cursor = max(cursor,row.end)
        if (fine_covered > duration - fine_covered + fine_width / 2 and fine_mean + fine_error < 0.
                and any(audit.get("rejected") and any(
                    row.get("start",end) < end and row.get("end",start) > start
                    for row in audit.get("windows", ())) for audit in sources)):
            fine_conflict = True
        cut_at_contamination = any(
            math.isclose(b, start, abs_tol=1e-6) or math.isclose(a, end, abs_tol=1e-6)
            for a, b in self.precise_negative_spans)
        if cut_at_contamination and fine and fine_mean - NormalDist().inv_cdf(.95) * fine_error <= 0.:
            fine_conflict = True
        boundaries = sorted({point for cell in cells for point, value in
                             ((cell.start, cell.left), (cell.end, cell.right))
                             if value == 0. or cell.blocked})
        assessment = {"integral": integral, "positive_fraction": positive / duration if duration > 0 else 0.,
                "observed_fraction": observed / duration if duration > 0 else 0.,
                "identity_fraction": identity / duration if duration > 0 else 0.,
                "blocked_fraction": blocked / duration if duration > 0 else 0.,
                "mean_evidence": integral / observed if observed else 0.,
                "negative_integral": negative, "effective_observations": observed / self.resolution,
                "change_penalty": penalty, "noise_scale": self.noise, "boundaries": boundaries,
                "occlusion": probe, "fine_mean": fine_mean, "fine_error": fine_error,
                "fine_covered_fraction": fine_covered / duration if duration > 0 else 0.,
                "fine_conflict": fine_conflict,
                "accepted": bool(cells and identity > 0 and blocked == 0
                                 and cells[0].identity and cells[-1].identity
                                 and math.isclose(observed, duration) and integral > 0.
                                 and (not edge_identity_conflict
                                      or cells[0].left >= 0. and cells[-1].right >= 0.
                                      or probe["homogeneous_shift"])
                                 and not probe["localized_contamination"] and not fine_conflict)}
        self._assessments[key] = assessment
        return assessment

    def _occlusion_probe(self, cells):
        """Measure the gain from hiding either edge at observed change points."""
        if not cells:
            return {"edge_gain": 0., "directional_divergence": 0.,
                    "localized_contamination": False, "homogeneous_shift": False}
        total = sum(c.integral for c in cells)
        duration = sum(c.end - c.start for c in cells)
        average = total / duration
        prefix, elapsed, gain = 0., 0., 0.
        edge_negative = False
        for cell in cells[:-1]:
            prefix += cell.integral
            elapsed += cell.end - cell.start
            gain = max(gain, (total - prefix) / (duration - elapsed) - average,
                       prefix / elapsed - average)
        values = [(c.left + c.right) / 2 for c in cells]
        spread = _scale(values)
        start, end = cells[0].start, cells[-1].end
        rows = [row for row in self.rows
                if min(end, row["span"][1]) > max(start, row["span"][0])]
        direction = []
        for row in rows:
            previous = _span(row.get("previous_span"))
            if not previous or previous[1] <= start:
                continue
            for channel, (center, scale) in self._direction_scales.items():
                cosine = (row.get("direction_to_previous") or {}).get(channel)
                if _finite(cosine):
                    angle = math.acos(max(-1., min(1., cosine)))
                    direction.append((angle - center) / scale)
        directional = max(direction) if direction else None
        invariant = []
        for row in rows:
            channels = (row.get("identity_evidence") or {}).get("channel_evidence") or {}
            values = [channels[name] for name in ("anime_va", "tertiary")
                      if _finite(channels.get(name))]
            invariant.append(bool(values) and min(values) > 0.)
        edge_negative = cells[0].integral < 0 or cells[-1].integral < 0
        return {"edge_gain": gain, "directional_divergence": directional,
                "localized_contamination": bool(edge_negative and gain > spread
                                                 and directional is not None
                                                 and directional > NormalDist().inv_cdf(.95)),
                "homogeneous_shift": bool(direction and not edge_negative
                                           and directional <= NormalDist().inv_cdf(.95)
                                           and invariant and all(invariant))}

    def decode(self, start, end, min_seconds):
        """Maximize integrated evidence minus a BIC cost for each retained interval."""
        if not _finite(min_seconds) or min_seconds <= 0:
            raise ValueError("min_seconds must be finite and positive")
        if end - start >= min_seconds - 1e-9 and self.assess(start, end)["accepted"]:
            return [(start, end)]
        cells = self._cells(start, end)
        runs, current = [], []
        for cell in cells:
            if cell.blocked or not cell.observed:
                if current:
                    runs.append(current)
                    current = []
            else:
                current.append(cell)
        if current:
            runs.append(current)
        spans = []
        for run in runs:
            points = [run[0].start] + [cell.end for cell in run]
            prefix = [0.]
            for cell in run:
                prefix.append(prefix[-1] + cell.integral)
            penalty = self._penalty(run)
            best, trace = [0.] * len(points), [None] * len(points)
            cursor, origin, gain = 0, None, -math.inf
            for index in range(1, len(points)):
                best[index], trace[index] = best[index - 1], (index - 1, None)
                while cursor < index and points[index] - points[cursor] >= min_seconds - 1e-9:
                    if run[cursor].integral > 0 and run[cursor].identity:
                        candidate = best[cursor] - prefix[cursor] - penalty
                        if candidate > gain:
                            gain, origin = candidate, cursor
                    cursor += 1
                score = prefix[index] + gain
                if origin is not None and run[index - 1].integral > 0 and score > best[index]:
                    best[index], trace[index] = score, (origin, (points[origin], points[index]))
            index, selected = len(points) - 1, []
            while index:
                index, span = trace[index]
                if span is not None:
                    selected.append(span)
            spans.extend(reversed(selected))
        return [span for span in spans if self.assess(*span)["accepted"]]
