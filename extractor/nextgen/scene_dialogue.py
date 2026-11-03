"""Frozen local cast evidence and smoothed bidirectional turn transitions."""
from __future__ import annotations

import math

import numpy as np

from .identity_cluster import profile_score


def _logsumexp(values):
    values = np.asarray(values, dtype=np.float64)
    peak = float(values.max())
    return peak + math.log(float(np.exp(values - peak).sum()))


class SceneDialogue:
    def __init__(self, turns, clusters, profile, anchors, scales, *, radius=20.):
        self.turns = tuple(turns)
        self.profile = profile
        self.scales = scales
        self.radius = radius
        self.clusters = {cluster.cluster_id: cluster for cluster in clusters}
        self.labels = {anchor.turn: "target" for anchor in anchors}
        by_index = {turn.index: turn for turn in turns if turn.usable}
        for cluster in clusters:
            members = [by_index[index] for index in cluster.members
                       if index in by_index and index not in cluster.outliers]
            if not members:
                continue
            affinities = np.array([[float(turn.primary @ cluster.primary_centroid),
                                    float(turn.secondary @ cluster.secondary_centroid)] for turn in members])
            central = np.median(affinities, axis=0)
            for turn, affinity in zip(members, affinities):
                if np.all(affinity >= central) and turn.index not in self.labels:
                    self.labels[turn.index] = str(cluster.cluster_id)

    def evaluate(self, primary, secondary, start, end, exclude_span=None):
        left, right = exclude_span or (start, end)
        nearby = [turn for turn in self.turns
                  if turn.start < right + self.radius and turn.end > left - self.radius
                  and min(right, turn.end) <= max(left, turn.start)]
        active = {self.labels[turn.index] for turn in nearby if turn.index in self.labels}
        if "target" not in active or len(active) < 2:
            return None
        # Unknown remains in competition even inside an apparently closed scene.
        states = ["target", *sorted(active - {"target"}), "unknown"]
        index = {state: position for position, state in enumerate(states)}
        logits = []
        p = primary / max(float(np.linalg.norm(primary)), np.finfo(float).eps)
        s = secondary / max(float(np.linalg.norm(secondary)), np.finfo(float).eps)
        for state in states:
            if state == "unknown":
                logits.append(0.)
                continue
            if state == "target":
                scores = (profile_score(p, self.profile.primary_centroid, self.profile.primary_references),
                          profile_score(s, self.profile.secondary_centroid, self.profile.secondary_references))
            else:
                cluster = self.clusters[int(state)]
                scores = (float(p @ cluster.primary_centroid), float(s @ cluster.secondary_centroid))
            logits.append(sum((score - getattr(self.profile, channel + "_reference_floor"))
                              / self.scales[channel] for channel, score in
                              zip(("primary", "secondary"), scores)) / 2.)
        acoustic = logits[0] - _logsumexp(logits[1:])
        # Jeffreys' half-count prior is invariant to relabeling a cast member.
        counts = np.full((len(states), len(states)), .5)
        ordered = sorted(nearby, key=lambda turn: (turn.start, turn.end))
        transitions = 0
        for before, after in zip(ordered, ordered[1:]):
            if before.end <= left and after.start >= right:
                continue
            a, b = self.labels.get(before.index), self.labels.get(after.index)
            if a in index and b in index:
                gap = max(0., after.start - before.end)
                counts[index[a], index[b]] += math.exp(-gap / self.radius)
                transitions += 1
        transition = counts / counts.sum(axis=1, keepdims=True)
        prior = np.zeros(len(states))
        before = next((turn for turn in reversed(ordered) if turn.end <= left), None)
        after = next((turn for turn in ordered if turn.start >= right), None)
        if before is not None and self.labels.get(before.index) in index:
            prior += np.log(transition[index[self.labels[before.index]], :])
        if after is not None and self.labels.get(after.index) in index:
            prior += np.log(transition[:, index[self.labels[after.index]]])
        posterior = np.asarray(logits) + prior
        posterior_odds = float(posterior[0] - _logsumexp(posterior[1:]))
        return {"active_cast": states, "acoustic_log_odds": acoustic,
                "markov_log_odds": posterior_odds,
                "markov_gain": posterior_odds - acoustic,
                "transition_observations": transitions,
                "unknown_probability": float(math.exp(posterior[-1] - _logsumexp(posterior)))}
