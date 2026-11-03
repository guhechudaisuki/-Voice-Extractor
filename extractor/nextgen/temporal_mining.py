"""Find clean target speech inside clustered and unclustered turns."""
from __future__ import annotations

from dataclasses import replace

import numpy as np

from .bootstrap_bank import BootstrapVector, evidence_spans, independent_vectors, merge_profile
from .episode_manifold import Anchor, EpisodeManifold
from .identity_cluster import Cluster, cohort_as_norm_z, eligible_competitor
from .scene_dialogue import SceneDialogue
from .feature_pyramid import MIN_MICRO_SECONDS, pyramid_spans

class TemporalMiner:
    def __init__(self, analyzer, turns, candidates, margin, bank, previous):
        from .cluster_runtime import unit

        self.analyzer, self.turns, self.candidates = analyzer, turns, candidates
        self.margin, self.bank, self.previous = margin, bank, previous
        self.profile = analyzer.profile
        self.certificate_profile = analyzer._certification_profile()
        self.targets = list(candidates.target_clusters())
        self.reference_mining = not self.targets
        if self.reference_mining:
            self.targets = [Cluster(-2, (), self.profile.primary_centroid,
                                   self.profile.secondary_centroid, 1., 1., 0.)]
        family = set(candidates.diagnostics.get("qualified_cluster_ids") or ())
        family.update(c.cluster_id for c in self.targets)
        family.update(c.cluster_id for c in candidates.clusters
                      if c.primary_target_score >= .66 and c.secondary_target_score >= .60)
        self.competitors = [c for c in candidates.clusters if c.cluster_id not in family
                            and eligible_competitor(c, turns)]
        self.target_primary = np.stack([c.posterior_primary if c.posterior_primary is not None
                                        else c.primary_centroid for c in self.targets])
        self.target_secondary = np.stack([c.posterior_secondary if c.posterior_secondary is not None
                                          else c.secondary_centroid for c in self.targets])
        self.target_medians = np.array([c.primary_median for c in self.targets])
        self.competitor_primary = np.stack([c.primary_centroid for c in self.competitors]) if self.competitors else None
        self.competitor_secondary = np.stack([c.secondary_centroid for c in self.competitors]) if self.competitors else None
        self._rival_masks = {}
        self.membership = list(candidates.membership)
        self.borderline = (list(candidates.borderline_membership)
                           if len(candidates.borderline_membership) == len(turns)
                           else [False] * len(turns))
        self.local, self.clean, self.supports = {}, {}, {}
        self.turn_vectors = {(float(t.start), float(t.end)): (t.primary, t.secondary)
                             for t in turns if t.usable}
        self.embeddings = getattr(analyzer, "_temporal_embeddings", {})
        self.anime = getattr(analyzer, "_temporal_anime", {})
        analyzer._temporal_embeddings, analyzer._temporal_anime = self.embeddings, self.anime
        self.window_scores, self.cohort_scores = {}, {}
        self.scoring_profiles = {}
        self.uncached_turns = []
        self.whole_turn_fallbacks = []
        self.unit = unit
        self.high_confidence, self.prototypes = [], []
        self.manifold = EpisodeManifold(self.profile, self.competitor_primary,
                                       self.competitor_secondary, margin)
        self.dialogue = None
        self.scene_exclusion = None
        self.blocked = tuple(getattr(analyzer, "temporal_blocked_spans", ()))
        unblocked_turns = [turn for turn in turns if not any(
            min(turn.end, b) > max(turn.start, a) for a, b in self.blocked)]
        self.manifold.fit_competitors(unblocked_turns, self.competitors, analyzer)
        self.manifold.discover_same_va(unblocked_turns, candidates.clusters, analyzer)
        self.demoted, self.cross_rescued = set(), set()
        self.raw_rescued = set(candidates.diagnostics.get("raw_rescued_turns") or ())
        old = previous[0].diagnostics if previous else {}
        self.core_certificates = dict(old.get("clean_core_certificates") or {})
        self.short_certificates = dict(old.get("short_dialogue_certificates") or {})
        self.diagnostics = {
            **candidates.diagnostics,
            "reference_mining": self.reference_mining,
            "certification_floors": {
                "primary": self.certificate_profile.primary_reference_floor,
                "secondary": self.certificate_profile.secondary_reference_floor},
            "profile_floors": {
                "primary": self.profile.primary_reference_floor,
                "secondary": self.profile.secondary_reference_floor,
                "anime_char": self.profile.anime_char_reference_floor,
                "anime_va": self.profile.anime_va_reference_floor},
            "turn_reference_scores": {
                str(t.index): dict(zip(("primary", "secondary"),
                                      analyzer.scores(t.primary, t.secondary)[0]))
                for t in turns if t.usable},
            "turn_cohort_z": {
                str(t.index): {
                    "cohort_z_primary": cohort_as_norm_z(t.primary,
                        [c.primary_centroid for c in self.targets],
                        [c.primary_centroid for c,keep in zip(self.competitors,
                            self._independent_rivals(t.start,t.end)) if keep]),
                    "cohort_z_secondary": cohort_as_norm_z(t.secondary,
                        [c.secondary_centroid for c in self.targets],
                        [c.secondary_centroid for c,keep in zip(self.competitors,
                            self._independent_rivals(t.start,t.end)) if keep])}
                for t in turns if t.usable},
        }


    def _vectors(self, start, end):
        key = (float(start), float(end))
        if key in self.turn_vectors:
            return self.turn_vectors[key]
        if key not in self.embeddings:
            turn = next((t for t in self.turns if t.usable
                         and abs(start - t.start) < 1e-4 and abs(end - t.end) < 1e-4), None)
            self.embeddings[key] = ((turn.primary, turn.secondary) if turn is not None
                                    else self.analyzer.embed(*key))
        return self.embeddings[key]

    def _anime_scores(self, start, end):
        key = (float(start), float(end))
        if key not in self.anime:
            method = getattr(self.analyzer, "anime_scores", None)
            self.anime[key] = method(*key) if method else None
        return self.anime[key]


    def _independent_rivals(self, start, end):
        """A cluster containing the probe's audio is not an independent rival."""
        indexes = tuple(t.index for t in self.turns if min(end,t.end) > max(start,t.start))
        if indexes not in self._rival_masks:
            self._rival_masks[indexes] = np.array([
                not set(indexes).intersection(c.members) for c in self.competitors], dtype=bool)
        return self._rival_masks[indexes]

    def _cohort(self, p, s, start=None, end=None):
        if not self.competitors:
            return None, None
        mask = self._independent_rivals(start,end) if start is not None else np.ones(len(self.competitors),dtype=bool)
        if not mask.any():
            return None, None
        p, s = self.unit(p), self.unit(s)
        return (float(np.einsum("ij,j->i", self.target_primary, p).max()
                      - np.einsum("ij,j->i", self.competitor_primary[mask], p).max()),
                float(np.einsum("ij,j->i", self.target_secondary, s).max()
                      - np.einsum("ij,j->i", self.competitor_secondary[mask], s).max()))

    def _scene_distance(self, start, end):
        exclude_start, exclude_end = self.scene_exclusion or (start, end)
        gaps = [max(0., a - end, start - b) for a, b in self.high_confidence
                if min(exclude_end, b) <= max(exclude_start, a)]
        return min(gaps) if gaps else None

    def _check(self, spans, profile, *, strict=False, scene=False):
        from .cluster_runtime import (BREATH_DIP_TOLERANCE, BREATH_NEIGHBOR_FLOOR,
                                      WINDOW_COHORT_PRIMARY, WINDOW_COHORT_SECONDARY,
                                      WINDOW_COHORT_STRONG)

        rows = []
        previous_vectors = None
        previous_span = None
        self.scoring_profiles[id(profile)] = profile
        for start, end in spans:
            p, s = self._vectors(start, end)
            vectors = {"primary": self.unit(p), "secondary": self.unit(s)}
            if previous_span is not None and abs((end-start)-(previous_span[1]-previous_span[0])) > .15:
                previous_vectors = None
            direction = ({channel: float(np.clip(vector @ previous_vectors[channel], -1., 1.))
                          for channel, vector in vectors.items()}
                         if previous_vectors is not None else None)
            score_key = (start, end, id(profile))
            if score_key not in self.window_scores:
                self.window_scores[score_key] = self.analyzer.scores(p, s, profile)
            scores, negatives = self.window_scores[score_key]
            p_score, s_score = scores
            if (start, end) not in self.cohort_scores:
                self.cohort_scores[start, end] = self._cohort(p, s, start, end)
            cm_p, cm_s = self.cohort_scores[start, end]
            cohort_ok = (cm_p is None or
                         (cm_p >= WINDOW_COHORT_PRIMARY and cm_s >= WINDOW_COHORT_SECONDARY)
                         or (cm_p >= 0. and cm_s >= WINDOW_COHORT_STRONG)
                         or cm_p >= WINDOW_COHORT_STRONG + .045)
            p_floor, s_floor = profile.primary_reference_floor, profile.secondary_reference_floor
            profile_ok = p_score >= p_floor - .02 and s_score >= s_floor - .02
            if not strict and not profile_ok:
                coherent = np.any(
                    (np.einsum("ij,j->i", self.target_primary, self.unit(p)) >= self.target_medians - .20)
                    & (np.einsum("ij,j->i", self.target_secondary, self.unit(s)) >= s_floor - .02))
                profile_ok = coherent and p_score >= p_floor - .10 and s_score >= s_floor - .10
            anime = self._anime_scores(start, end)
            exact_turn = next((t for t in self.turns if t.start == start and t.end == end), None)
            char_score, va_score = anime if anime is not None else (None, None)
            tertiary = self.analyzer.tertiary_score(exact_turn) if exact_turn is not None else None
            tertiary_vector = exact_turn.tertiary if tertiary is not None else None
            has_target_support = (p_score >= p_floor - .06 or s_score >= s_floor - .06
                or (char_score is not None and char_score >= profile.anime_char_reference_floor)
                or (va_score is not None and va_score >= profile.anime_va_reference_floor))
            if (tertiary is None and end-start >= MIN_MICRO_SECONDS - 1e-9
                    and (not self.competitors or has_target_support or exact_turn is not None)
                    and getattr(self.analyzer,'tertiary',None) is not None):
                # Short phonetic probes need their own independent encoder vote.
                tertiary_vector = self.analyzer.embed3(start,end)
                if tertiary_vector is not None:
                    tertiary = float(self.unit(tertiary_vector) @ self.unit(profile.tertiary_centroid))
            observations = {"primary": p_score, "secondary": s_score,
                            "anime_char": char_score, "anime_va": va_score, "tertiary": tertiary}
            if anime is not None and hasattr(self.analyzer, "embed_anime"):
                pair = self.analyzer.embed_anime(start, end)
                if pair is not None:
                    vectors.update(anime_char=pair[0], anime_va=pair[1])
            if tertiary is not None:
                vectors["tertiary"] = tertiary_vector
            independent_margins = {}
            if tertiary_vector is not None:
                vector = self.unit(tertiary_vector)
                independent_margins = {
                    role: {"tertiary": float(tertiary - vector @ self.unit(centroid))}
                    for role, centroid in getattr(self.analyzer, "tertiary_negatives", ())}
            exclusion = self.manifold.exclusion(
                scores, negatives, getattr(self.analyzer, "exclusion_audit", None),
                independent_margins)
            geometry = (self.manifold.geometry(p, s, start, end, self.scene_exclusion, observations)
                        if scene else None)
            dialogue = (self.dialogue.evaluate(p, s, start, end, self.scene_exclusion)
                        if scene and self.dialogue is not None else None)
            physiology = None
            invariant_support = any(
                observations.get(channel) is not None
                and observations[channel] > self.manifold.floors[channel]
                for channel in ("anime_va", "tertiary"))
            if (invariant_support and (p_score < p_floor or s_score < s_floor)
                    and exclusion["passed"] and not exclusion["conflict"]):
                measure = getattr(self.analyzer, "physiology_evidence", None)
                physiology = measure(start, end) if measure is not None else None
            excluded_clusters = {c.cluster_id for c, independent in zip(
                self.competitors, self._independent_rivals(start,end)) if not independent}
            prosody = (self.analyzer.prosody(start,end) if self.manifold.prosody_reference is not None
                       and end-start >= .8 else None)
            evidence = self.manifold.evaluate(observations, (cm_p, cm_s), exclusion,
                geometry, dialogue, physiology, vectors, excluded_cluster_ids=excluded_clusters, prosody=prosody,
                vocal_geometry=(self.analyzer.vocal_geometry_evidence(start,end)
                    if end-start >= .8 and hasattr(self.analyzer,'references') else None))
            margins_ok = exclusion["strict_passed"]
            base_accepted = bool(profile_ok and cohort_ok and margins_ok)
            micro = end - start <= .6 + 1e-9
            if micro:
                # A short embedding needs direct agreement, never a neighbour's identity.
                base_accepted = bool(p_score >= p_floor - .06 and s_score >= s_floor - .06
                                     and cohort_ok and margins_ok)
            base_accepted = (base_accepted and not evidence.get('same_va', {}).get('conflict', False)
                             and not evidence.get('vocal_geometry', {}).get('extreme_conflict', False))
            fused = evidence["accepted"]
            anime_shift = bool(fused and evidence["invariant_evidence"] is not None
                               and evidence["invariant_evidence"] > 0.)
            scene_match = bool(fused and geometry is not None and geometry["margin"] is not None
                               and geometry["margin"] > 0.)
            rows.append({"span": [float(start), float(end)], "primary": float(p_score),
                         "scale": "micro" if micro else "meso",
                         "secondary": float(s_score), "cm_p": cm_p, "cm_s": cm_s,
                         "accepted": bool(base_accepted or fused),
                         "reference_consensus": evidence.get('reference_consensus', False),
                         "cohort_passed": bool(cohort_ok),
                         "exclusion_margins_passed": bool(margins_ok or fused),
                         "anime_char_score": char_score, "anime_va_score": va_score,
                         "tertiary_score": tertiary, "tertiary": tertiary,
                         "anime_register_shift": bool(anime_shift),
                         "scene_supported": scene_match,
                         "physiology": physiology,
                         "direction_to_previous": direction,
                         "previous_span": previous_span,
                         "direction_reference_spread": self.manifold.direction_spread,
                         "llr": evidence["llr"], "manifold_log_odds": evidence["llr"],
                         "identity_evidence": evidence})
            if micro:
                raw = evidence.get("raw_channel_evidence", evidence["channel_evidence"])
                strong_dual = min(raw.get("primary", 0.), raw.get("secondary", 0.)) >= 1.
                exclusive_dual = (cm_p is not None and cm_s is not None
                                  and cm_p >= .15 and cm_s >= .25 and base_accepted)
                duration_supported = (base_accepted and p_score >= p_floor - .06
                                      and s_score >= s_floor - .06)
                positive = bool((strong_dual or exclusive_dual or fused or duration_supported) and margins_ok
                                and not exclusion["conflict"] and (base_accepted or fused))
                # Short phonetic fragments often score low against EVERY voice.
                # Absence of target evidence is not positive evidence of a rival.
                rival_supported = False
                if self.competitor_primary is not None:
                    mask = self._independent_rivals(start,end)
                    rp = self.competitor_primary[mask] @ vectors['primary']
                    rs = self.competitor_secondary[mask] @ vectors['secondary']
                    tp = float(np.max(self.target_primary @ vectors['primary']))
                    ts = float(np.max(self.target_secondary @ vectors['secondary']))
                    rival_supported = bool(np.any((rp > p_floor) & (rs > s_floor)
                        & (rp-tp > self.margin) & (rs-ts > self.margin)))
                excluded_supported = any(np_ > p_floor and ns > s_floor
                    and np_-p_score > self.margin and ns-s_score > self.margin
                    for _,np_,ns in negatives)
                negative = bool(rival_supported or excluded_supported
                                or evidence.get('same_va', {}).get('conflict'))
                energy_method = getattr(self.analyzer, 'relative_window_energy', None)
                parent = next((t for t in self.turns if t.start <= start+1e-6 and t.end >= end-1e-6),None)
                if energy_method is not None and parent is not None:
                    relative_energy = energy_method(start,end,parent.start,parent.end)
                    rows[-1]['relative_energy'] = relative_energy
                    if relative_energy < .01:
                        positive = negative = False
                rows[-1].update(micro_state="target" if positive else "other" if negative else "uncertain",
                                accepted=positive)
                if positive and duration_supported:
                    # Carry measured duration-adjusted evidence into trajectory
                    # decoding; a target flag with negative long-window z-scores
                    # would otherwise become a false micro barrier downstream.
                    rows[-1]['duration_reference_evidence'] = min(
                        (p_score - p_floor + .06) / self.manifold.scales['primary'],
                        (s_score - s_floor + .06) / self.manifold.scales['secondary'])
            previous_vectors = vectors
            previous_span = [float(start), float(end)]
        neighbor_floor = profile.primary_reference_floor + BREATH_NEIGHBOR_FLOOR
        for index in range(1, len(rows) - 1):
            left, mid, right = rows[index - 1:index + 2]
            if mid.get("scale") == "micro":
                continue
            if mid["accepted"] or not left["accepted"] or not right["accepted"]:
                continue
            if mid.get("identity_evidence", {}).get("exclusion_evidence", {}).get("conflict"):
                continue
            if mid.get('identity_evidence', {}).get('same_va', {}).get('conflict'):
                continue
            if mid.get('identity_evidence', {}).get('vocal_geometry', {}).get('extreme_conflict'):
                continue
            if (min(left["primary"], right["primary"]) >= neighbor_floor
                    and mid["primary"] >= min(left["primary"], right["primary"]) - BREATH_DIP_TOLERANCE
                    and mid["secondary"] >= profile.secondary_reference_floor - .12):
                mid.update(accepted=True, smoothed=True)
        return rows

    def _eligible(self, turn):
        return turn.usable

    def _scan(self, turn):
        # Cache state affects cost only, never which acoustic windows are examined.
        rows = []
        levels = pyramid_spans(turn.start, turn.end)
        prefetch = getattr(self.analyzer, 'prefetch', None)
        if prefetch is not None:
            prefetch([span for _,spans in levels for span in spans])
        for scale, spans in levels:
            rows.extend(dict(row, scale=scale) for row in self._check(spans, self.profile))
        return rows


    def _expanded_rows(self, turn, rows, vectors, label, *, scene=False):
        if not vectors or all(row["accepted"] for row in rows):
            return rows
        expanded = self._check([r["span"] for r in rows],
                               merge_profile(self.profile, vectors), strict=True, scene=scene)
        def supported(row):
            channels = row.get('identity_evidence', {}).get('channel_evidence', {})
            return row['accepted'] or (row.get('micro_state') == 'uncertain'
                and min(channels.get('primary', 0.), channels.get('secondary', 0.)) > 0.
                and row.get('exclusion_margins_passed') and row.get('cohort_passed'))
        if label in ("bootstrap", "cross_episode") and not all(supported(row) for row in expanded):
            return rows
        combined = []
        used = False
        for original, candidate in zip(rows, expanded):
            if not original["accepted"] and supported(candidate):
                combined.append({**candidate, "reference": label})
                used = True
            else:
                combined.append(original)
        if used:
            key = str(turn.index)
            self.supports[key] = tuple(sorted(set(self.supports.get(key, ()))
                                               | set(evidence_spans(vectors))))
            if label == "cross_episode":
                self.cross_rescued.add(turn.index)
        return combined

    def _extract(self, turn, rows):
        from .cluster_runtime import MIN_CLEAN_SUBSPAN, WHOLE_TURN_PASS_RATIO
        if any(row.get("scale") == "micro" for row in rows):
            from .identity_trajectory import IdentityTrajectory
            trajectory = IdentityTrajectory(rows, self.manifold.floors, ())
            return [list(span) for span in trajectory.decode(
                turn.start, turn.end, MIN_MICRO_SECONDS)]

        accepted = [row["accepted"] for row in rows]
        if not accepted or not any(accepted):
            return []
        if (sum(accepted) / len(accepted) >= WHOLE_TURN_PASS_RATIO
                and accepted[0] and accepted[-1]):
            return [[turn.start, turn.end]]
        spans, run = [], None
        for row, ok in zip(rows, accepted):
            if ok:
                run = list(row["span"]) if run is None else [run[0], row["span"][1]]
            elif run is not None:
                if run[1] - run[0] >= MIN_CLEAN_SUBSPAN - 1e-9:
                    spans.append(run)
                run = None
        if run is not None and run[1] - run[0] >= MIN_CLEAN_SUBSPAN - 1e-9:
            spans.append(run)
        return spans

    def _nonmember_window_passes(self, row):
        evidence = row.get("identity_evidence", {})
        if not row.get("accepted") or evidence.get("exclusion_evidence", {}).get("conflict"):
            return False
        if evidence.get("accepted"):
            return True
        channels = evidence.get("channel_evidence", {})
        return (row.get("exclusion_margins_passed") and channels.get("primary", 0.) > 0.
                and channels.get("secondary", 0.) > 0.
                and (evidence.get("cohort_evidence") or 0.) > 0.)

    def _certify(self, turn, rows, *, scene=False):
        from .cluster_runtime import (MIN_CLEAN_SUBSPAN, SHORT_TURN_SECONDS,
                                      WEAK_SECONDARY_SLACK, WINDOW_COHORT_PRIMARY,
                                      _clean_core_certified, _short_dialogue_path)

        key = str(turn.index)
        if any(row.get("scale") == "micro" for row in rows):
            # Preserve independently certified mesoscopic speech when a short
            # embedding abstains. Only resolved micro evidence may move a boundary.
            coarse = [row for row in rows if row.get("scale") != "micro"]
            legacy_clean, coarse = self._certify(turn, coarse, scene=scene) if coarse else ([], [])
            coarse = [dict(row, accepted=bool(row["accepted"] and any(
                a <= row["span"][0]+1e-6 and b >= row["span"][1]-1e-6 for a,b in legacy_clean)))
                for row in coarse]
            combined = [*coarse, *(row for row in rows if row.get("scale") == "micro")]
            from ..candidate_lattice import CandidateLattice
            from .identity_trajectory import IdentityTrajectory
            supported_spans = [*legacy_clean, *(row['span'] for row in rows
                                                if row.get('micro_state') == 'target')]
            trajectory = IdentityTrajectory(combined, self.manifold.floors, ())
            clean = [list(span) for span in CandidateLattice(trajectory,supported_spans).decode(MIN_MICRO_SECONDS)]
            micro_core = any(row.get('micro_state') == 'target' and any(
                a <= row['span'][0]+1e-6 and b >= row['span'][1]-1e-6 for a,b in clean)
                for row in rows)
            if clean and micro_core:
                self.core_certificates[key] = clean
                self.short_certificates.setdefault(key, "multiscale_acoustic_consensus")
            elif key in self.core_certificates:
                self.core_certificates[key] = [span for span in self.core_certificates[key]
                    if any(a <= span[0]+1e-6 and b >= span[1]-1e-6 for a,b in clean)]
            return clean, combined
        self.core_certificates.pop(key, None)
        self.short_certificates.pop(key, None)
        clean = self._extract(turn, rows)
        partial = bool(clean) and (len(clean) != 1
                   or turn.duration - (clean[0][1] - clean[0][0]) > .15 + 1e-9)
        t_score = self.analyzer.tertiary_score(turn)
        t_strong = t_score is not None and t_score >= self.profile.tertiary_reference_floor + .03
        certified = []
        short_member = (self.candidates.membership[turn.index] and not partial
                        and MIN_MICRO_SECONDS <= turn.duration < 1.20 and clean == [[turn.start, turn.end]])
        if partial or not self.candidates.membership[turn.index] or short_member:
            for start, end in clean:
                core = [r for r in rows if r["accepted"] and not r.get("smoothed")
                        and r["span"][0] >= start - 1e-6 and r["span"][1] <= end + 1e-6]
                certificate = self.certificate_profile if turn.duration <= SHORT_TURN_SECONDS else self.profile
                anime_core = core and all(r.get("anime_register_shift") for r in core)
                fused_core = core and all(r.get("identity_evidence",{}).get("accepted") for r in core)
                dual_core = bool(core) and _clean_core_certified(core, certificate, False)
                core_t_strong = t_strong
                nonmember_core = (not self.candidates.membership[turn.index] and core
                                  and end - start >= MIN_MICRO_SECONDS - 1e-9
                                  and all(self._nonmember_window_passes(r) for r in core))
                if (nonmember_core or (short_member and core and all(
                        self._nonmember_window_passes(row) for row in core))
                        or (end - start >= MIN_CLEAN_SUBSPAN - 1e-9 and core
                        and (dual_core or _clean_core_certified(core, certificate, core_t_strong) or anime_core or fused_core))):
                    certified.append([start, end])
            if certified:
                self.core_certificates[key] = certified
        if clean and turn.duration <= SHORT_TURN_SECONDS:
            score, _ = self.analyzer.scores(turn.primary, turn.secondary)
            p_slack = score[0] - self.certificate_profile.primary_reference_floor
            s_slack = score[1] - self.profile.secondary_reference_floor
            cm_p, _ = self._cohort(turn.primary, turn.secondary, turn.start, turn.end)
            cohort = cm_p if cm_p is not None else float("inf")
            path = None if partial else _short_dialogue_path(
                turn.duration, p_slack, s_slack, rows, t_strong, cohort,
                scene_distance=self._scene_distance(turn.start, turn.end) if scene else None)
            if not path and not partial and rows and all(r.get("anime_register_shift") and r["accepted"] for r in rows):
                path = "invariant_consensus"
            if (not path and not partial and rows and all(
                    r["accepted"] and r.get("identity_evidence", {}).get("accepted") for r in rows)):
                path = "multiview_consensus"
            if path:
                self.short_certificates[key] = path
            if not path and (s_slack < WEAK_SECONDARY_SLACK / 2
                    or (s_slack < WEAK_SECONDARY_SLACK and (cohort < WINDOW_COHORT_PRIMARY or partial))):
                clean = certified
                if not clean:
                    self.demoted.add(turn.index)
                rows = [{**r, "accepted": False, "demoted": True}
                        if not any(a <= r["span"][0] + 1e-6 and b >= r["span"][1] - 1e-6
                                   for a, b in certified) else r for r in rows]
        if not self.candidates.membership[turn.index]:
            clean = [[a, b] for a, b in clean if b - a >= MIN_MICRO_SECONDS - 1e-9 and
                     (core := [r for r in rows if r["accepted"]
                               and a <= r["span"][0] + 1e-6 and b >= r["span"][1] - 1e-6])
                     and all(self._nonmember_window_passes(r) for r in core)]
            if clean:
                self.core_certificates[key] = clean
            else:
                self.core_certificates.pop(key, None)
            rows = [dict(row, accepted=False) if not any(
                    a <= row["span"][0] + 1e-6 and b >= row["span"][1] - 1e-6
                    for a, b in clean) else row for row in rows]
        return clean, rows

    def _collect_prototypes(self):
        candidates = []
        for turn in self.turns:
            for row in self.local.get(str(turn.index), ()):
                start, end = row["span"]
                if (not row["accepted"] or row.get("smoothed") or row.get("reference")
                        or row.get("scale") == "micro"
                        or row.get("scene_supported")
                        or row.get("identity_evidence", {}).get("physiology_compensation")
                        or any(min(end, b) > max(start, a) for a, b in self.blocked)
                        or row.get("cm_p") is None or row.get("cm_s") is None
                        or min(row["cm_p"], row["cm_s"]) <= 0.
                        or not any(a <= start + 1e-6 and b >= end - 1e-6
                                   for a, b in self.clean.get(str(turn.index), ()))):
                    continue
                evidence = row.get("identity_evidence", {})
                channels = evidence.get("channel_evidence", {})
                if (not evidence.get("exclusion_evidence", {}).get("strict_passed")
                        or min(channels.get("primary", 0.), channels.get("secondary", 0.)) <= 0.):
                    continue
                p, s = self._vectors(start, end)
                strength = min(channels["primary"], channels["secondary"],
                               row["cm_p"] / self.manifold.scales["primary"],
                               row["cm_s"] / self.manifold.scales["secondary"])
                confidence = float(1. / (1. + np.exp(-strength)))
                observations = {"primary": row["primary"], "secondary": row["secondary"],
                                "anime_char": row.get("anime_char_score"),
                                "anime_va": row.get("anime_va_score"),
                                "tertiary": row.get("tertiary_score")}
                candidates.append(Anchor(turn.index, start, end, p, s, confidence, observations))
        anchors = []
        median = float(np.median([a.confidence for a in candidates])) if candidates else 0.
        for anchor in sorted(candidates, key=lambda a: (-a.confidence, a.start, a.end)):
            if anchor.confidence < median:
                continue
            if any(min(anchor.end, old.end) > max(anchor.start, old.start) for old in anchors):
                continue
            anchors.append(anchor)
        self.manifold.fit(anchors)
        complete = {turn.index for turn in self.turns
                    if self.clean.get(str(turn.index)) == [[turn.start, turn.end]]}
        dialogue_turns = [turn for turn in self.turns if not any(
            min(turn.end, b) > max(turn.start, a) for a, b in self.blocked)]
        self.dialogue = SceneDialogue(dialogue_turns, self.competitors, self.certificate_profile,
                                     [anchor for anchor in anchors if anchor.turn in complete],
                                     self.manifold.scales)
        self.high_confidence = [(a.start, a.end) for a in anchors]
        self.prototypes = [BootstrapVector(self.analyzer.source_hash, a.start, a.end,
                           a.end - a.start, a.primary, a.secondary, a.confidence) for a in anchors]

    def run(self):
        for turn in self.turns:
            key = str(turn.index)
            if not self._eligible(turn):
                self.membership[turn.index] = False
                continue
            if self.previous and self.previous[0].membership[turn.index]:
                self.local[key] = self.previous[1].get(key, [])
                self.clean[key] = self.previous[0].diagnostics.get("turn_clean_spans", {}).get(key, [])
                self.membership[turn.index] = bool(self.clean[key])
                self.supports[key] = self.previous[0].diagnostics.get("bootstrap_supports", {}).get(key, ())
                continue
            if turn.index in self.raw_rescued and self.candidates.membership[turn.index]:
                self.clean[key], self.local[key] = [[turn.start, turn.end]], []
                continue
            rows = self._scan(turn)
            independent = independent_vectors(self.bank, self.analyzer.source_hash, turn.start, turn.end)
            rows = self._expanded_rows(turn, rows, independent, "bootstrap")
            rows = self._expanded_rows(turn, rows, getattr(self.analyzer, "cross_bank", ()), "cross_episode")
            self.clean[key], self.local[key] = self._certify(turn, rows)
            self.membership[turn.index] = bool(self.clean[key])
            if not self.membership[turn.index] and self.candidates.membership[turn.index]:
                self.borderline[turn.index] = True
            if (turn.index + 1) % 25 == 0:
                self.analyzer.progress(f"全集局部核验 {turn.index + 1}/{len(self.turns)}")
        self._collect_prototypes()
        rescued = []
        for turn in self.turns:
            key = str(turn.index)
            if key not in self.local:
                continue
            rows = self.local[key]
            if not rows or all(r["accepted"] for r in rows):
                continue
            self.scene_exclusion = (turn.start, turn.end)
            contextual = self._check([r["span"] for r in rows], self.profile, scene=True)
            combined = []
            for original, candidate in zip(rows, contextual):
                if original["accepted"]:
                    combined.append(original)
                    continue
                if not candidate.get("identity_evidence", {}).get("accepted"):
                    combined.append({**original, "scene_probe": candidate.get("identity_evidence", {})})
                    continue
                candidate["reference"] = "episode_manifold"
                combined.append(candidate)
                geometry = candidate["identity_evidence"].get("manifold") or {}
                supports = {(self.analyzer.source_hash, a, b) for a, b in geometry.get("supports", ())}
                self.supports[key] = tuple(sorted(set(self.supports.get(key, ())) | supports))
            old_clean = self.clean.get(key, [])
            clean, rows = self._certify(turn, combined, scene=True)
            merged = []
            for start, end in sorted(clean + old_clean):
                if merged and start <= merged[-1][1]:
                    merged[-1][1] = max(merged[-1][1], end)
                else:
                    merged.append([start, end])
            clean = merged
            self.local[key] = rows
            if clean:
                if not self.membership[turn.index]:
                    rescued.append(turn.index)
                self.demoted.discard(turn.index)
                self.clean[key], self.local[key] = clean, rows
                self.membership[turn.index], self.borderline[turn.index] = True, False
        accepted = sorted(span for key, spans in self.clean.items()
                          if self.membership[int(key)] for span in spans)
        self.diagnostics.update(
            accepted_spans=accepted, turn_clean_spans=self.clean,
            certified_short_spans=[span for span in accepted
                                   if MIN_MICRO_SECONDS - 1e-9 <= span[1] - span[0] < 1.2
                                   and any(row.get("micro_state") == "target"
                                           and row["span"][0] >= span[0]-1e-6
                                           and row["span"][1] <= span[1]+1e-6
                                           for rows in self.local.values() for row in rows)],
            feature_pyramid={"micro_seconds": .5, "micro_hop": .1,
                             "meso_seconds": 1.8, "meso_hop": .9},
            temporal_rejected_count=sum(old and not new for old, new in
                                        zip(self.candidates.membership, self.membership)),
            global_mined_turns=[t.index for t in self.turns
                               if self.membership[t.index] and not self.candidates.membership[t.index]],
            uncached_mining_turns=self.uncached_turns,
            whole_turn_mining_fallbacks=self.whole_turn_fallbacks,
            episode_register_prototypes=[{"span": [v.start, v.end], "confidence": v.confidence}
                                         for v in self.prototypes],
            episode_manifold=self.manifold.diagnostics(),
            episode_register_rescued_turns=rescued,
            short_dialogue_certificates=self.short_certificates,
            clean_core_certificates=self.core_certificates,
            short_weak_demoted=sorted(self.demoted),
            cross_episode_rescued_turns=sorted(self.cross_rescued),
            bootstrap_supports=self.supports)
        return replace(self.candidates, membership=tuple(self.membership),
                       borderline_membership=tuple(self.borderline),
                       diagnostics=self.diagnostics), self.local
