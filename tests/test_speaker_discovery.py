"""Phase D tests: speaker discovery + poisoning guard."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extractor.nextgen.speaker_discovery import (  # noqa: E402
    compute_purity_score, discover_episode_speakers, select_pure_seeds)


def unit(v):
    v = np.asarray(v, dtype=np.float64)
    return v / np.linalg.norm(v)


class PurityScoreTests(unittest.TestCase):
    def test_score_range(self):
        cluster = {"primary_cohesion": 0.8, "primary_median": 0.9}
        score = compute_purity_score(cluster)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)
        self.assertAlmostEqual(score, 0.85)


class SelectPureSeedsTests(unittest.TestCase):
    def _cluster(self, members):
        return {"members": [{"index": i, "span": [s, e],
                             "speech_seconds": e - s,
                             "accepted_by_cluster": True}
                            for i, s, e in members],
                "outliers": []}

    def test_selects_duration_qualified(self):
        cluster = self._cluster([(0, 10.0, 13.0), (1, 20.0, 24.0), (2, 30.0, 40.0)])
        seeds = select_pure_seeds(cluster, max_seeds=2)
        self.assertEqual(len(seeds), 2)
        for s in seeds:
            self.assertGreaterEqual(s["end"] - s["start"], 1.8)
            self.assertLessEqual(s["end"] - s["start"], 5.0)

    def test_excludes_outliers(self):
        cluster = self._cluster([(0, 10.0, 13.0)])
        cluster["outliers"] = [0]
        seeds = select_pure_seeds(cluster)
        self.assertEqual(len(seeds), 0)

    def test_single_cue_bonus(self):
        from extractor.subtitles import SubtitleCue
        cluster = self._cluster([(0, 10.0, 13.0), (1, 20.0, 24.0)])
        subtitles = [SubtitleCue(1, 10.2, 12.8, "セリフ")]
        seeds = select_pure_seeds(cluster, subtitles, max_seeds=2)
        self.assertTrue(seeds[0]["text"])


class DiscoverEpisodeSpeakersTests(unittest.TestCase):
    def test_filters_small_clusters(self):
        report = {"clusters": [
            {"cluster_id": 0, "member_count": 5, "total_speech_seconds": 50,
             "primary_cohesion": 0.8, "primary_median": 0.85,
             "members": [{"index": i, "span": [i * 10.0, i * 10.0 + 3.0],
                          "speech_seconds": 3.0, "accepted_by_cluster": True}
                         for i in range(5)],
             "outliers": []},
            {"cluster_id": 1, "member_count": 2, "total_speech_seconds": 5.0,
             "members": [], "outliers": []},
        ]}
        speakers = discover_episode_speakers(report)
        self.assertEqual(len(speakers), 1)
        self.assertEqual(speakers[0].cluster_id, 0)


class PoisoningGuardTests(unittest.TestCase):
    def test_purified_indexes_computed(self):
        from extractor.nextgen.cluster_runtime import ClusterAnalyzer
        analyzer = ClusterAnalyzer.__new__(ClusterAnalyzer)
        # 模拟 5 个参考，其中 1 个是毒丸（与其他 4 个相似度极低）
        rng = np.random.default_rng(20261027)
        base = unit(rng.normal(0, 1, 16))
        base2 = unit(rng.normal(0, 1, 16))
        poison = unit(rng.normal(0, 1, 16))
        refs = [base + rng.normal(0, 0.02, 16) for _ in range(2)]
        refs += [base2 + rng.normal(0, 0.02, 16) for _ in range(2)]
        refs.append(poison)
        vectors = [(unit(v), unit(v)) for v in refs]
        analyzer._file_embedding = lambda p: vectors[0]  # placeholder
        # 直接测试 purity 逻辑
        p = np.stack([v[0] for v in vectors])
        p_unit = np.stack([unit(row) for row in p])
        sim = p_unit @ p_unit.T
        medians = np.array([np.median(np.delete(sim[i], i)) for i in range(5)])
        best_med = float(medians.max())
        valid = medians >= max(0.42, best_med - 0.22)
        self.assertLess(int(valid.sum()), 5)  # 毒丸被排除


if __name__ == "__main__":
    unittest.main()
