"""Calibrated multiview identity evidence and frozen episode anchor geometry."""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from .exclusion_reliability import margin_consensus
from .score_calibration import AdaptiveLLR


_EPS = np.finfo(np.float64).eps
CHANNELS = ("primary", "secondary", "anime_char", "anime_va", "tertiary")


def _unit(vector):
    value = np.asarray(vector, dtype=np.float64)
    return value / max(float(np.linalg.norm(value)), _EPS)


def _spread(values, prior):
    values = np.asarray(values, dtype=np.float64)
    if values.size < 2:
        return float(prior)
    median = np.median(values)
    # Normal-consistent median absolute deviation, regularized by one prior sample.
    variance = (1.482602218505602 * np.median(np.abs(values - median))) ** 2
    return float(math.sqrt((len(values) * variance + prior * prior) / (len(values) + 1)))


@dataclass(frozen=True)
class Anchor:
    turn: int
    start: float
    end: float
    primary: np.ndarray
    secondary: np.ndarray
    confidence: float
    observations: dict


class RegisterSubspace:
    """Low-rank spherical span with an orientation and dimension correction."""

    def __init__(self, vectors, weights=None):
        matrix = np.stack([_unit(vector) for vector in vectors])
        weights = np.ones(len(matrix)) if weights is None else np.asarray(weights, dtype=float)
        self.center = _unit(np.average(matrix, axis=0, weights=weights))
        _, singular, directions = np.linalg.svd(matrix * np.sqrt(weights[:, None]), full_matrices=False)
        energy = singular ** 2
        mass = energy / max(float(energy.sum()), _EPS)
        # Entropy rank keeps resolved voice directions without spanning sample noise.
        rank = math.ceil(math.exp(-float(np.sum(mass * np.log(np.maximum(mass, _EPS))))))
        rank = min(rank, int(np.sum(singular > singular[0] * max(matrix.shape) * _EPS)))
        self.basis = directions[:max(1, rank)]
        self.chance_energy = len(self.basis) / matrix.shape[1]

    def residual(self, vector):
        probe = _unit(vector)
        energy = min(1., float(np.sum((self.basis @ probe) ** 2)))
        explained = max(0., (energy - self.chance_energy) / max(1. - self.chance_energy, _EPS))
        # A linear span contains antipodes; a speaker's spherical patch does not.
        explained *= max(0., float(self.center @ probe))
        return max(_EPS, 1. - explained)


class EpisodeManifold:
    """Use only frozen, independently certified anchors for a single propagation pass."""

    def __init__(self, profile, primary_competitors, secondary_competitors, margin):
        self.profile = profile
        self.margin = float(margin)
        self.competitors = {"primary": primary_competitors, "secondary": secondary_competitors}
        self.floors, self.scales, self.reliability = {}, {}, {}
        self.anchors = ()
        self.matrices = {}
        self.radius = {}
        self.scene_scale = None
        self.reference_matrices = {}
        self.direction_spread = {}
        self.rival_subspaces = {}
        self._rival_vectors = {}
        self._subspace_cache = {}
        self.calibrators = {}
        self.whiteners = {}
        self.cohort_cluster_ids = {}
        self._calibration_cache = {}
        self.same_va_rivals = []
        self.prosody_reference = None
        for channel in CHANNELS:
            floor = float(getattr(profile, channel + "_reference_floor"))
            self.floors[channel] = floor
            references = getattr(profile, channel + "_references", None)
            if channel in ("primary", "secondary"):
                self.reference_matrices[channel] = np.stack([
                    _unit(row) for row in (references if references is not None and len(references)
                                           else [getattr(profile, channel + "_centroid")])])
                matrix = self.reference_matrices[channel]
                if len(matrix) > 1:
                    distances = np.maximum(0., 1. - matrix @ matrix.T)
                    self.direction_spread[channel] = float(np.median(
                        distances[np.triu_indices(len(matrix), k=1)]))
                else:
                    self.direction_spread[channel] = None
                rivals = self.competitors[channel]
                self._rival_vectors[channel] = [[row] for row in rivals] if rivals is not None else []
                self.rival_subspaces[channel] = [RegisterSubspace([row]) for row in rivals] if rivals is not None else []
            prior = max(1. - floor, self.margin) / math.sqrt(12.)
            scores = []
            if references is not None and len(references) > 1:
                matrix = np.stack([_unit(row) for row in references])
                for index, row in enumerate(matrix):
                    other = np.delete(matrix, index, axis=0)
                    scores.append(float(row @ _unit(other.mean(axis=0))))
            scale = self.scales[channel] = _spread(scores, prior)
            signal = max(float(np.median(scores)) - floor, 0.) if scores else prior
            self.reliability[channel] = signal / max(signal + scale, _EPS)
        self._refresh_geometry()

    def _geometry_vector(self, channel, vectors):
        whitening = self.whiteners.get(channel)
        values = np.asarray(vectors,dtype=float)
        if whitening is not None and len(whitening.basis):
            return whitening.transform(values)
        return values/np.maximum(np.linalg.norm(values,axis=-1,keepdims=True),_EPS)

    def _refresh_geometry(self):
        """One metric for queries, refs, anchors, rival subspaces and radii.

        Original-domain reference scores, floors, direction tests and LLR
        calibrators are deliberately separate from this relative geometry.
        """
        self._subspace_cache.clear()
        self._geometry_references, self._geometry_anchors = {}, {}
        self._geometry_competitors, self._geometry_radius = {}, {}
        for channel in ('primary','secondary'):
            refs = self._geometry_vector(channel,self.reference_matrices[channel])
            self._geometry_references[channel] = refs
            anchors = np.asarray([getattr(a,channel) for a in self.anchors]).reshape(-1,refs.shape[1])
            matrix = self._geometry_vector(channel,anchors)
            self._geometry_anchors[channel] = matrix
            rivals = self.competitors[channel]
            self._geometry_competitors[channel] = (self._geometry_vector(channel,rivals)
                                                   if rivals is not None else None)
            self.rival_subspaces[channel] = [RegisterSubspace(self._geometry_vector(channel,rows))
                for rows in self._rival_vectors[channel]]
            if channel not in self.whiteners:
                self._geometry_radius[channel] = self.radius.get(channel,self.scales[channel])
                continue
            distance = np.maximum(0.,1.-matrix@matrix.T)
            for i,a in enumerate(self.anchors):
                distance[i,[b.turn==a.turn for b in self.anchors]] = np.inf
            nearest = distance.min(axis=1) if len(matrix) else np.array([])
            nearest = nearest[np.isfinite(nearest)]
            reference_distances = (1.-refs@refs.T)[np.triu_indices(len(refs),k=1)]
            prior = float(np.median(reference_distances)) if len(reference_distances) else .1
            self._geometry_radius[channel] = max(float(np.median(nearest)) if len(nearest) else prior,1e-6)

    def clear_whitening(self):
        for whitening in self.whiteners.values():
            whitening.clear()
        self.whiteners.clear()
        self._refresh_geometry()

    def fit(self, anchors):
        self.anchors = tuple(anchors)
        self._subspace_cache.clear()
        if not self.anchors:
            self._refresh_geometry()
            return
        for channel in ("primary", "secondary"):
            matrix = np.stack([_unit(getattr(anchor, channel)) for anchor in self.anchors])
            self.matrices[channel] = matrix
            distance = np.maximum(0., 1. - matrix @ matrix.T)
            for index, anchor in enumerate(self.anchors):
                distance[index, [other.turn == anchor.turn for other in self.anchors]] = np.inf
            nearest = distance.min(axis=1)
            nearest = nearest[np.isfinite(nearest)]
            self.radius[channel] = (float(np.median(nearest)) if len(nearest)
                                    else self.scales[channel])
            self.radius[channel] = max(self.radius[channel], self.scales[channel] ** 2)
        for channel in CHANNELS:
            values = [a.observations[channel] for a in self.anchors
                      if a.observations.get(channel) is not None]
            if not values:
                continue
            scale = _spread(values, self.scales[channel])
            self.scales[channel] = scale
            signal = max(float(np.median(values)) - self.floors[channel], 0.)
            self.reliability[channel] = signal / max(signal + scale, _EPS)
        times = sorted({(a.start + a.end) / 2. for a in self.anchors})
        gaps = np.diff(times)
        if len(gaps):
            self.scene_scale = float(np.median(gaps[gaps > 0.]))
        self._refresh_geometry()

    def fit_competitors(self, turns, clusters, analyzer=None):
        """Keep each rival's register space separate, using its coherent members."""
        by_index = {turn.index: turn for turn in turns if turn.usable}
        for channel in ("primary", "secondary"):
            spaces, raw_spaces = [], []
            for cluster in clusters:
                members = [by_index[index] for index in cluster.members
                           if index in by_index and index not in cluster.outliers]
                centroid = getattr(cluster, channel + "_centroid")
                affinity = [float(_unit(getattr(turn, channel)) @ centroid) for turn in members]
                center = float(np.median(affinity)) if affinity else 0.
                vectors = [getattr(turn, channel) for turn, value in zip(members, affinity) if value >= center]
                spaces.append(RegisterSubspace([centroid, *vectors]))
                raw_spaces.append([centroid,*vectors])
            self.rival_subspaces[channel] = spaces
            self._rival_vectors[channel] = raw_spaces
        cohort_vectors = {channel: [] for channel in CHANNELS}
        cohort_ids = {channel: [] for channel in CHANNELS}
        for cluster in clusters:
            for channel in ("primary", "secondary"):
                cohort_vectors[channel].append(getattr(cluster, channel + "_centroid"))
                cohort_ids[channel].append(cluster.cluster_id)
            members = [by_index[index] for index in cluster.members
                       if index in by_index and index not in cluster.outliers]
            # One observation per rival, independent of its number of lines.
            members = sorted(members, key=lambda turn: -turn.speech_seconds)[:3]
            tertiary = [turn.tertiary for turn in members if turn.tertiary is not None]
            if tertiary:
                cohort_vectors["tertiary"].append(_unit(np.mean(tertiary, axis=0)))
                cohort_ids['tertiary'].append(cluster.cluster_id)
            if analyzer is not None and getattr(self.profile, "anime_char_centroid", None) is not None:
                pairs = [analyzer.embed_anime(turn.start, turn.end) for turn in members]
                pairs = [pair for pair in pairs if pair is not None]
                for i, channel in enumerate(("anime_char", "anime_va")):
                    if pairs:
                        cohort_vectors[channel].append(_unit(np.mean([pair[i] for pair in pairs], axis=0)))
                        cohort_ids[channel].append(cluster.cluster_id)
                if pairs:
                    self.register_same_va(cluster.cluster_id,
                        cohort_vectors['anime_char'][-1], cohort_vectors['anime_va'][-1])
        self.cohort_cluster_ids = cohort_ids
        self._calibration_cache.clear()
        for whitening in self.whiteners.values():
            whitening.clear()
        self.whiteners.clear()
        # TemporalMiner supplies qualified non-target cohorts, after removing
        # blocked turns and the target family. Limit each speaker's influence.
        for channel in ('primary','secondary'):
            observations, groups = [], []
            for cluster in clusters:
                members = [by_index[i] for i in cluster.members if i in by_index and i not in cluster.outliers]
                members = sorted(members,key=lambda t:(t.start,t.end))
                for index in np.linspace(0,len(members)-1,min(16,len(members))).astype(int):
                    observations.append(getattr(members[index],channel))
                    groups.append(cluster.cluster_id)
            if len(observations)>=8 and len(set(groups))>=2:
                from .episode_whitening import EpisodeWhitening
                whitening = EpisodeWhitening(observations,groups)
                if len(whitening.basis):
                    self.whiteners[channel] = whitening
        for channel, rivals in cohort_vectors.items():
            refs = getattr(self.profile, channel + "_references", None)
            center = getattr(self.profile, channel + "_centroid", None)
            if center is not None:
                refs = refs if refs is not None and len(refs) else [center]
                self.calibrators[channel] = AdaptiveLLR(refs, rivals, self.floors[channel], self.scales[channel])
        self._refresh_geometry()

    def register_same_va(self, cluster_id, character, actor):
        """Detect an acoustic actor/character disagreement, never using text labels."""
        char_ref = getattr(self.profile, 'anime_char_centroid', None)
        va_ref = getattr(self.profile, 'anime_va_centroid', None)
        if char_ref is None or va_ref is None:
            return
        va_score = float(_unit(actor) @ _unit(va_ref))
        char_score = float(_unit(character) @ _unit(char_ref))
        actor_support = (va_score-self.floors['anime_va'])/self.scales['anime_va']
        character_gap = (self.floors['anime_char']-char_score)/self.scales['anime_char']
        if actor_support > 1. and character_gap > 1.:
            self.same_va_rivals = [row for row in self.same_va_rivals if row['cluster_id'] != cluster_id]
            self.same_va_rivals.append({'cluster_id': cluster_id, 'character': _unit(character),
                'actor': _unit(actor), 'strength': min(3., actor_support, character_gap),
                'actor_similarity': va_score, 'character_similarity': char_score})

    def discover_same_va(self, turns, clusters, analyzer):
        """A role mistakenly grouped into the target family must also be checked."""
        from .identity_cluster import eligible_competitor
        if getattr(self.profile, 'anime_char_centroid', None) is None:
            return
        by_index = {turn.index: turn for turn in turns if turn.usable}
        known = {row['cluster_id'] for row in self.same_va_rivals}
        for cluster in clusters:
            if cluster.cluster_id in known or not eligible_competitor(cluster, turns):
                continue
            members = sorted((by_index[i] for i in cluster.members if i in by_index),
                             key=lambda turn:-turn.speech_seconds)[:3]
            pairs = [analyzer.embed_anime(turn.start,turn.end) for turn in members]
            pairs = [pair for pair in pairs if pair is not None]
            if pairs:
                self.register_same_va(cluster.cluster_id,
                    np.mean([p[0] for p in pairs],axis=0), np.mean([p[1] for p in pairs],axis=0))
        if self.same_va_rivals and hasattr(analyzer, 'reference_prosody'):
            references = analyzer.reference_prosody()
            if len(references) >= 2:
                references = np.asarray(references)
                center = np.median(references,axis=0)
                scale = np.maximum(1.4826*np.median(abs(references-center),axis=0),[.1,.02,.02,.15])
                self.prosody_reference = (center,scale)
                for rival in self.same_va_rivals:
                    cluster = next(c for c in clusters if c.cluster_id==rival['cluster_id'])
                    members = sorted((by_index[i] for i in cluster.members if i in by_index),
                                     key=lambda t:-t.speech_seconds)[:3]
                    values = [v for t in members if (v := analyzer.prosody(t.start,t.end)) is not None]
                    # A single emotional performance is not a habitual contour.
                    if len(values) >= 2:
                        rival['prosody'] = np.median(values,axis=0)

    def same_va_evidence(self, vectors, prosody=None, excluded_cluster_ids=()):
        if not self.same_va_rivals or vectors.get('anime_char') is None or vectors.get('anime_va') is None:
            return {"active": False, "conflict": False, "strength": 0.}
        char, va = _unit(vectors['anime_char']), _unit(vectors['anime_va'])
        target_char = float(char @ _unit(self.profile.anime_char_centroid))
        related = [row for row in self.same_va_rivals
                   if row['cluster_id'] not in excluded_cluster_ids
                   and float(va @ row['actor']) > self.floors['anime_va']]
        if not related:
            return {"active": False, "conflict": False, "strength": 0.}
        rival = max(related, key=lambda row: float(char @ row['character']))
        rival_score = float(char @ rival['character'])
        margin = target_char - rival_score
        prosody_gain = 0.
        if self.prosody_reference is not None and prosody is not None and 'prosody' in rival:
            center,scale = self.prosody_reference
            separation = float(np.mean(((rival['prosody']-center)/scale)**2))
            target_distance = float(np.mean(((prosody-center)/scale)**2))
            rival_distance = float(np.mean(((prosody-rival['prosody'])/scale)**2))
            # Rhythm can strengthen character discrimination by at most one
            # weight unit, and only after independent character opposition.
            if separation > 1. and rival_score > self.floors['anime_char'] and margin < 0.:
                prosody_gain = float(np.clip((target_distance-rival_distance)/(1.+separation),0.,1.))
        return {"active": True, "cluster_id": rival['cluster_id'], "strength": rival['strength']+prosody_gain,
                "prosody_weight_gain": prosody_gain,
                "character_margin": margin, "conflict": bool(
                    rival_score > self.floors['anime_char'] and margin < -self.scales['anime_char'])}

    def _target_subspace(self, channel, indexes):
        key = (channel, tuple(indexes))
        if key not in self._subspace_cache:
            references = self._geometry_references[channel]
            anchors = [self._geometry_anchors[channel][index] for index in indexes]
            # Reference pseudo-count is measured in seconds. One short line has
            # a small effect; many independent turns gradually dominate.
            weights = [6. / len(references)] * len(references)
            weights += [min(3., self.anchors[index].end-self.anchors[index].start) for index in indexes]
            self._subspace_cache[key] = RegisterSubspace([*references, *anchors], weights)
        return self._subspace_cache[key]

    def geometry(self, primary, secondary, start, end, exclude_span=None, observations=None):
        primary = self._geometry_vector('primary',primary)
        secondary = self._geometry_vector('secondary',secondary)
        domain = 'episode_whitened' if self.whiteners else 'original'
        left, right = exclude_span or (start, end)
        indexes = [i for i, a in enumerate(self.anchors)
                   if min(right, a.end) <= max(left, a.start)]
        if not indexes:
            margins = {}
            for channel, vector in (("primary", primary), ("secondary", secondary)):
                target = self._target_subspace(channel, ()).residual(vector)
                rivals = self.rival_subspaces[channel]
                if rivals:
                    rival = min(space.residual(vector) for space in rivals)
                    margins[channel] = (rival-target) / max(rival+target, _EPS)
            return {"margin": float(np.mean(list(margins.values()))) if margins else None,
                    "channel_margins": margins, "context_affinity": 0., "supports": [],
                    "reference_prior": True, "geometry_domain":domain}
        distances, rival_distances = {}, {}
        weights = {}
        for channel, vector in (("primary", primary), ("secondary", secondary)):
            probe = _unit(vector)
            radius = self._geometry_radius[channel]
            distances[channel] = np.maximum(0., 1. - self._geometry_anchors[channel][indexes] @ probe) / radius
            rivals = self._geometry_competitors[channel]
            if rivals is not None and len(rivals):
                rival_distances[channel] = float(np.maximum(0., 1. - rivals @ probe).min() / radius)
            weights[channel] = self.reliability[channel]
        observations = observations or {}
        invariant = [(float(observations[c]) - self.floors[c]) / self.scales[c]
                     for c in ("anime_va", "tertiary") if observations.get(c) is not None]
        secondary_score = observations.get("secondary")
        if invariant and secondary_score is not None:
            residual = (float(secondary_score) - self.floors["secondary"]) / self.scales["secondary"]
            consensus = float(np.mean(invariant))
            if residual < 0. and consensus > 0.:
                weights["secondary"] /= 1. + (consensus - residual) ** 2
        total_weight = max(sum(weights.values()), _EPS)
        joint = sum(weights[c] * distances[c] for c in distances) / total_weight
        count = max(1, math.ceil(math.sqrt(len(indexes))))
        selected = np.argsort(joint)[:count]
        affinity = np.exp(-(joint[selected] - joint[selected].min()))
        affinity *= [self.anchors[indexes[i]].confidence for i in selected]
        affinity /= max(float(affinity.sum()), _EPS)
        target_distance = float(affinity @ joint[selected])
        margins, projection = {}, {}
        for channel, rival in rival_distances.items():
            target = float(affinity @ distances[channel][selected])
            graph_margin = (rival - target) / max(rival + target, _EPS)
            probe = primary if channel == "primary" else secondary
            target_space = self._target_subspace(channel, indexes)
            target_residual = target_space.residual(probe)
            rival_residual = min((space.residual(probe) for space in self.rival_subspaces[channel]), default=1.)
            projection_margin = (rival_residual - target_residual) / max(rival_residual + target_residual, _EPS)
            margins[channel] = (graph_margin + projection_margin) / 2.
            projection[channel] = {"target_residual": target_residual, "rival_residual": rival_residual,
                                   "margin": projection_margin, "rank": len(target_space.basis)}
        margin = (sum(weights[c] * value for c, value in margins.items())
                  / max(sum(weights[c] for c in margins), _EPS)) if margins else None
        selected_anchors = [self.anchors[indexes[i]] for i in selected]
        nearest_gap = min(max(0., a.start - end, start - a.end) for a in selected_anchors)
        context = (math.exp(-nearest_gap / self.scene_scale)
                   if self.scene_scale is not None and self.scene_scale > 0. else 0.)
        return {"margin": margin, "channel_margins": margins, "geometry_domain":domain,
                "distance": target_distance, "context_affinity": context,
                "subspace_projection": projection,
                "supports": [[a.start, a.end] for a in selected_anchors]}

    def exclusion(self, scores, negatives, audit=None, independent_margins=None,
                  *, observations=None):
        """Discount a dissenting view only with two same-role supporting views."""
        rows = []
        for role, primary, secondary in negatives:
            purity = (audit or {}).get(role, {})
            differences = {"primary": float(scores[0] - primary),
                           "secondary": float(scores[1] - secondary)}
            strict = all(value >= self.margin for value in differences.values())
            for channel, value in (independent_margins or {}).get(role, {}).items():
                if channel in self.scales and channel not in differences and np.isfinite(value):
                    differences[channel] = float(value)
            weights = {channel: float(np.clip(purity.get(channel, 1.), 0., 1.))
                       * self.reliability[channel] for channel in differences}
            target_scores = dict(observations or {}, primary=float(scores[0]),
                                 secondary=float(scores[1]))
            excluded_scores = {channel: float(target_scores[channel] - gap)
                               for channel, gap in differences.items()
                               if target_scores.get(channel) is not None
                               and np.isfinite(target_scores[channel])}
            absolute_support = {channel: score > self.floors[channel]
                                for channel, score in excluded_scores.items()}
            consensus = margin_consensus(differences, weights, self.margin,
                                         absolute_support=absolute_support)
            total = sum(weights.values())
            net = sum(weights[channel] * (value - self.margin) / self.scales[channel]
                      for channel, value in differences.items()) / max(total, _EPS)
            rows.append({"role": role, "llr": net,
                         "weights": weights,
                         "margins": [differences["primary"], differences["secondary"]],
                         "channel_margins": differences, "consensus": consensus,
                         "excluded_scores": excluded_scores,
                         "absolute_support": absolute_support,
                         "strict_passed": strict,
                         "passed": bool(strict or consensus["accepted"]),
                         "consensus_conflict": consensus["corroborated_conflict"]})
        return {"llr": min((row["llr"] for row in rows), default=0.),
                "strict_passed": all(row["strict_passed"] for row in rows),
                "passed": all(row["passed"] for row in rows),
                "conflict": any(row["consensus_conflict"] for row in rows),
                "conflict_basis": "positive_excluded_identity", "roles": rows}

    def _calibrators_for(self, excluded_cluster_ids):
        key = tuple(sorted(excluded_cluster_ids))
        if not key:
            return self.calibrators
        if key not in self._calibration_cache:
            result = {}
            for channel, calibrator in self.calibrators.items():
                ids = self.cohort_cluster_ids.get(channel, ())
                keep = [i for i,c in enumerate(ids) if c not in excluded_cluster_ids]
                result[channel] = (calibrator if len(keep) == len(ids) else AdaptiveLLR(
                    calibrator.references, calibrator.cohorts[keep], calibrator.floor,
                    calibrator.scale, calibrator.top_k))
            self._calibration_cache[key] = result
        return self._calibration_cache[key]

    def evaluate(self, observations, cohort, exclusion, geometry=None, dialogue=None,
                 physiology=None, vectors=None, *, excluded_cluster_ids=(), prosody=None,
                 vocal_geometry=None, physical_quality=None, raw_channel_evidence=None):
        evidence = {channel: (float(value) - self.floors[channel]) / self.scales[channel]
                    for channel, value in observations.items()
                    if channel in self.floors and value is not None}
        raw_evidence = dict(evidence)
        calibrators = self._calibrators_for(excluded_cluster_ids)
        for channel, calibrator in calibrators.items():
            if channel in evidence and calibrator.available and (vectors or {}).get(channel) is not None:
                vector, score = vectors[channel], observations[channel]
                llr = calibrator.evaluate(score, vector)
                evidence[channel] = llr if raw_evidence[channel] > 0. else min(0., llr)
        weights = {channel: self.reliability[channel] for channel in evidence}
        if raw_channel_evidence is not None and np.isfinite(raw_channel_evidence):
            # Original-track dual encoders belong to the SAME correlated group;
            # they cannot manufacture an additional independent identity vote.
            evidence['raw'] = float(raw_channel_evidence)
            weights['raw'] = min(self.reliability['primary'],self.reliability['secondary'])
        same_va = self.same_va_evidence(vectors or {}, prosody, excluded_cluster_ids)
        from .quality_fusion import quality_weights
        physical = dict(physical_quality or {}, same_va_conflict=same_va['strength'])
        gates = quality_weights(physical, channels=weights)
        if physical_quality is not None:
            weights = {c:w*gates[c]*len(gates) for c,w in weights.items()}
        invariants = [evidence[c] for c in ("anime_va", "tertiary") if c in evidence]
        invariant = float(np.mean(invariants)) if invariants else None
        if invariant is not None and invariant > 0. and evidence.get("primary", 0.) > 0.:
            # A one-view residual increases that view's noise, not the other views' scores.
            variable = evidence.get("secondary")
            if variable is not None and variable < 0.:
                weights["secondary"] /= 1. + (invariant - variable) ** 2
        compensation = {}
        acoustic_confidence = (physiology or {}).get("artifact_confidence", 0.)
        if ((physiology or {}).get("available") and np.isfinite(acoustic_confidence)
                and acoustic_confidence > 0. and invariant is not None and invariant > 0.
                and exclusion["passed"] and not exclusion["conflict"]):
            for channel, companion in (("primary", "secondary"), ("secondary", "primary")):
                residual = evidence.get(channel)
                if residual is None or residual >= 0. or evidence.get(companion, 0.) <= 0.:
                    continue
                # Paired acoustics explain measurement noise, not speaker identity.
                inflation = 1. + min(1., acoustic_confidence) * (invariant - residual) ** 2
                weights[channel] /= inflation
                compensation[channel] = 1. / inflation
        groups, group_weights = [], []
        for channels in (("primary", "secondary", "raw"), ("anime_char",), ("anime_va", "tertiary")):
            available = [c for c in channels if c in evidence]
            denominator = sum(weights[c] for c in available)
            if denominator:
                groups.append(sum(weights[c] * evidence[c] for c in available) / denominator)
                gain = 1. + same_va['strength']
                quality_mass = (sum(gates[c] for c in available)/len(available)*len(gates)
                                if physical_quality is not None else 1.)
                group_weights.append(quality_mass*(gain if channels == ('anime_char',) else
                                      1./gain if channels == ('anime_va', 'tertiary') else 1.))
        appearance = float(np.average(groups, weights=group_weights)) if groups else 0.
        uncertainty = float(np.std(groups) / math.sqrt(len(groups))) if groups else 0.
        competitive = [float(value) / self.scales[channel]
                       for channel, value in zip(("primary", "secondary"), cohort)
                       if value is not None]
        competition = float(np.mean(competitive)) if competitive else None
        geometric = geometry.get("margin") if geometry is not None else None
        local_competition = dialogue.get("acoustic_log_odds") if dialogue else None
        scene_exclusive = local_competition is not None and local_competition > 0.
        exclusive = (competition is not None and competition > 0.) or (
            geometric is not None and geometric > 0.)
        # Temporal proximity strengthens an existing geometric match, never supplies identity.
        geometry_llr = 0.
        if geometric is not None:
            bounded = float(np.clip(geometric, -1. + _EPS, 1. - _EPS))
            geometry_llr = math.log1p(bounded) - math.log1p(-bounded)
            if geometry_llr > 0.:
                geometry_llr *= 1. + geometry["context_affinity"]
        scene_llr = 0.
        if scene_exclusive and appearance > 0. and invariant is not None and invariant > 0.:
            # Context can spend only acoustic evidence already earned locally.
            scene_llr = min(local_competition, max(0., dialogue["markov_gain"]))
            exclusive = exclusive or scene_llr > 0.
        vocal_prior = float(np.clip((vocal_geometry or {}).get('log_prior',0.),-np.log(2.),0.))
        llr = appearance + (competition or 0.) + geometry_llr + scene_llr + min(0., exclusion["llr"]) + vocal_prior
        independent = sum(value > 0. for value in groups)
        # Cohort margins are cosine differences. Combine them with the reference
        # margin on that same scale, never with a fitted Gaussian log likelihood.
        # Changing cohort size otherwise changes an acoustic support decision.
        complementary = all(raw_evidence.get(channel, -math.inf) + max(0., gap or 0.) / self.scales[channel] > 0.
                            for channel, gap in zip(("primary", "secondary"), cohort))
        identity = (any(raw_evidence.get(c, 0.) > 0. for c in ("primary", "secondary", "anime_char"))
                    and (evidence.get("primary", 0.) > 0. or complementary
                         or (geometric is not None and geometric > 0.)))
        # With a small episode cohort, aggregate LLR can straddle zero even when
        # all four direct reference views agree. Preserve that independent
        # certificate only inside the estimated uncertainty and with dual rival
        # separation; neither context nor physiology can supply it.
        reference_consensus = bool(appearance + uncertainty > 0.
            and all(raw_evidence.get(c, -math.inf) > 0.
                    for c in ('primary','secondary','anime_char','anime_va'))
            and all(gap is not None and gap >= self.margin for gap in cohort)
            and exclusion.get('strict_passed') and not exclusion['conflict']
            and not same_va['conflict'] and not (vocal_geometry or {}).get('extreme_conflict'))
        reference_only = bool(all(gap is None for gap in cohort)
            and raw_evidence.get('secondary',0.) > 0.
            and raw_evidence.get('tertiary',0.) > 0.
            and raw_evidence.get('anime_va',0.) > 0.
            and observations.get('primary',-math.inf) >= self.floors['primary']-.10
            and exclusion.get('strict_passed') and not exclusion['conflict']
            and not same_va['conflict'] and not (vocal_geometry or {}).get('extreme_conflict'))
        certificate = (min(raw_evidence[c] for c in ('secondary','tertiary')) if reference_only
                       else min(raw_evidence[c] for c in ('primary','secondary')) if reference_consensus else 0.)
        accepted = ((llr > 0. or reference_only) and (appearance > 0. or reference_consensus or reference_only)
                    and (identity or reference_only) and (exclusive or reference_only) and (independent >= 2 or reference_only)
                    and not (vocal_geometry or {}).get('extreme_conflict')
                    and (reference_only or invariant is not None and invariant > 0.)
                    and not same_va['conflict']
                    and not exclusion["conflict"]
                    and exclusion["passed"])
        return {"llr": float(llr), "appearance_evidence": appearance, "appearance_uncertainty": uncertainty, "accepted": bool(accepted),
                "reference_consensus": bool((reference_consensus or reference_only) and accepted),
                "reference_certificate_evidence": float(certificate) if accepted else 0.,
                "reference_only_consensus": reference_only,
                "vocal_geometry": vocal_geometry or {}, "vocal_geometry_prior": vocal_prior,
                "channel_evidence": evidence, "channel_reliability": weights,
                "physical_quality": physical_quality or {}, "quality_gates": gates,
                "raw_channel_evidence": raw_evidence,
                "same_va": same_va, "group_weights": group_weights,
                "calibration": {c: item.diagnostics() for c, item in calibrators.items()},
                "invariant_evidence": invariant, "cohort_evidence": competition,
                "manifold_margin": geometric, "manifold": geometry,
                "scene_dialogue": dialogue, "scene_evidence": scene_llr,
                "physiology_compensation": compensation,
                "exclusion_evidence": exclusion, "independent_views": independent}

    def diagnostics(self):
        return {"anchor_count": len(self.anchors), "scales": self.scales,
                "whitening": {c:w.diagnostics() for c,w in self.whiteners.items()},
                "same_va_rivals": [{key:value.tolist() if isinstance(value,np.ndarray) else value
                                    for key,value in row.items() if key not in ('character','actor')}
                                   for row in self.same_va_rivals],
                "calibration": {c: item.diagnostics() for c, item in self.calibrators.items()},
                "reliability": self.reliability, "radius": self.radius,
                "scene_scale": self.scene_scale,
                "rival_subspace_ranks": {channel: [len(space.basis) for space in spaces]
                                          for channel, spaces in self.rival_subspaces.items()}}
