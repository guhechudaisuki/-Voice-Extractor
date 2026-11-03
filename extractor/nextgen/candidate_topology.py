"""Preserve sentence ownership when joining adjacent certified intervals."""
from __future__ import annotations

from statistics import median


class CandidateTopology:
    def __init__(self, candidates, local, minimum):
        self.candidates = tuple(candidates)
        self.units = []
        for index, candidate in enumerate(self.candidates):
            evidence = candidate.diagnostics.get("experimental_final_island_evidence") or {}
            islands = evidence.get("islands", ()) if isinstance(evidence, dict) else ()
            parts = [row["span"] for row in islands
                     if isinstance(row, dict) and row.get("state") == "target_supported"
                     and isinstance(row.get("span"), (tuple, list)) and len(row["span"]) == 2]
            self.units.extend((index, part, float(a), float(b))
                              for part, (a, b) in enumerate(parts))
            if not parts:
                self.units.append((index, None, candidate.start, candidate.end))
        self.minimum = minimum
        widths = [float(row["span"][1] - row["span"][0])
                  for rows in (local or {}).values() if isinstance(rows, list)
                  for row in rows if isinstance(row, dict) and row.get("span")]
        # A gap larger than half an identity window has no shared observation.
        self.bridge_limit = min(minimum / 2, median(widths) / 2) if widths else minimum / 2

    def owner(self, start, end):
        center = (start + end) / 2
        matches = [unit for unit in self.units if unit[2] <= center <= unit[3]]
        if not matches:
            return None
        return min(matches, key=lambda item: (
            item[3] - item[2], not self.candidates[item[0]].accepted,
            -max(0., min(end, item[3]) - max(start, item[2]))))[:2]

    def merge(self, spans, barriers=()):
        merged, constituents = [], []
        for start, end in sorted(set((float(a), float(b)) for a, b in spans if b > a)):
            if merged:
                left, right = merged[-1]
                gap = start - right
                fragment = min(right - left, end - start) < self.minimum
                owner = self.owner(left, right)
                same_sentence = (owner is not None and self.candidates[owner[0]].accepted
                                 and owner == self.owner(start, end))
                blocked = any(min(start, b) > max(right, a) for a, b in barriers)
                if gap <= 0 or (not blocked and gap <= self.bridge_limit
                                and (fragment or same_sentence)):
                    merged[-1][1] = max(right, end)
                    constituents[-1].append((start, end))
                    continue
            merged.append([start, end])
            constituents.append([(start, end)])
        return merged, constituents
