"""Stage 1–4 高召回改造的单元测试。

覆盖：候选句级门控与 Tier A/B 分流、连续干净子区间、短句降级、
partial 裁切导出与 GPT-SoVITS train.list、困难负例打包。
"""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
import zipfile
from dataclasses import replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extractor.nextgen.cluster_gate import (  # noqa: E402
    apply_cluster_gate, candidate_membership)
from extractor.nextgen.identity_cluster import (  # noqa: E402
    AssignResult, Cluster, TargetProfile, TurnEvidence, profile_score)
from extractor.types import CandidateSentence  # noqa: E402


def unit(vector):
    vector = np.asarray(vector, dtype=np.float64)
    return vector / np.linalg.norm(vector)


def axis(cosine, base, rng):
    other = unit(rng.normal(0.0, 1.0, size=len(base)))
    orthogonal = unit(other - (other @ base) * base)
    return unit(cosine * base + np.sqrt(max(0.0, 1 - cosine ** 2)) * orthogonal)


def make_result(turns, target_id=0, accepted_spans=None, borderline=None,
                membership=None, qualified=(), clusters=()):
    """构造一个最小可用的 AssignResult 供门控测试。"""
    membership = tuple(membership if membership is not None
                       else [True] * len(turns))
    borderline = tuple(borderline if borderline is not None
                       else [False] * len(turns))
    diagnostics = {
        "accepted_spans": accepted_spans if accepted_spans is not None
        else [[t.start, t.end] for t, m in zip(turns, membership) if m],
        "profile_floors": {"primary": 0.58, "secondary": 0.51},
        "qualified_cluster_ids": list(qualified),
    }
    return AssignResult(
        clusters=tuple(clusters) or (
            Cluster(cluster_id=target_id, members=tuple(t.index for t in turns),
                    primary_centroid=unit(np.sum([t.primary for t in turns], axis=0)),
                    secondary_centroid=unit(np.sum([t.secondary for t in turns], axis=0)),
                    primary_cohesion=0.5, primary_median=0.8,
                    total_speech_seconds=sum(t.speech_seconds for t in turns)),),
        target_cluster_id=target_id,
        membership=membership,
        cluster_of=tuple([target_id] * len(turns)),
        reason="target_cluster_found",
        diagnostics=diagnostics,
        target_cluster_ids=(target_id,),
        borderline_membership=borderline,
    )


def synthetic_turns(rng, count=4, start=0.0, duration=4.0):
    base = unit(rng.normal(0.0, 1.0, size=64))
    turns = []
    for index in range(count):
        turns.append(TurnEvidence(
            index=index, start=start + index * duration, end=start + (index + 1) * duration,
            speech_seconds=duration,
            primary=unit(base + rng.normal(0, 0.03, 64)),
            secondary=unit(base + rng.normal(0, 0.03, 64))))
    return turns


class CandidateMembershipTests(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(20261023)
        self.turns = synthetic_turns(self.rng)

    def test_candidate_covered_by_accepted_span(self):
        result = make_result(self.turns)
        candidate = CandidateSentence(0.5, 2.0, "", accepted=True)
        allowed, _ids, reason, tier = candidate_membership(
            candidate, self.turns, result, min_seconds=1.2)
        self.assertTrue(allowed)
        self.assertEqual(tier, "A")
        self.assertEqual(reason, "target_cluster")

    def test_partial_coverage_ratio_frees_candidate(self):
        # 候选 85% 以上落在已验证干净区间内（头尾轻微越界）→ 放行。
        result = make_result(self.turns, accepted_spans=[[0.1, 3.8]])
        candidate = CandidateSentence(0.0, 4.0, "", accepted=True)
        allowed, _ids, _reason, tier = candidate_membership(
            candidate, self.turns, result, min_seconds=1.2)
        self.assertTrue(allowed)
        self.assertEqual(tier, "A")

    def test_outside_spans_falls_to_borderline_tier(self):
        result = make_result(self.turns, accepted_spans=[[10.0, 12.0]],
                             borderline=[False, True, False, False],
                             membership=[True, False, True, True])
        candidate = CandidateSentence(4.2, 6.5, "", accepted=True)
        allowed, _ids, _reason, tier = candidate_membership(
            candidate, self.turns, result, min_seconds=1.2)
        self.assertFalse(allowed)
        self.assertEqual(tier, "B")

    def test_rescue_marked_weak_window_never_enters_tier_a(self):
        result = make_result(self.turns)
        candidate = CandidateSentence(0.5, 2.0, "", accepted=True,
                                      window_min_score=0.5,
                                      diagnostics={"target_locator_recovery": True,
                                                   "camplus_score": 0.9})
        allowed, _ids, reason, tier = candidate_membership(
            candidate, self.turns, result, min_seconds=1.2)
        self.assertFalse(allowed)
        self.assertEqual(tier, "B")
        self.assertEqual(reason, "rescue_strict_review")

    def test_unmarked_weak_window_goes_tier_b(self):
        result = make_result(self.turns)
        candidate = CandidateSentence(0.5, 2.0, "", accepted=True,
                                      window_min_score=0.45)
        allowed, _ids, _reason, tier = candidate_membership(
            candidate, self.turns, result, min_seconds=1.2)
        self.assertFalse(allowed)
        self.assertEqual(tier, "B")


class ApplyGateTests(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(20261023)
        self.turns = synthetic_turns(self.rng)

    def test_promotion_requires_turn_window_ratio(self):
        result = make_result(self.turns)
        local = {"0": [{"span": [0.0, 1.8], "accepted": False},
                       {"span": [0.9, 2.7], "accepted": False}]}
        candidate = CandidateSentence(0.4, 2.2, "", reject_reason="声纹匹配不足",
                                      window_min_score=0.7)
        accepted, rejected = [], [candidate]
        apply_cluster_gate(accepted, rejected, self.turns, result,
                           mode="enforce", min_seconds=1.2, local=local)
        # 低通过率（混合长句）不得整句晋升：与目标家族重叠时进 Tier B 待试听。
        self.assertFalse(candidate.accepted)
        self.assertEqual(candidate.diagnostics.get("cluster_tier"), "B")

    def test_demoted_accepted_defaults_to_tier_b(self):
        result = make_result(self.turns, accepted_spans=[[100.0, 102.0]])
        candidate = CandidateSentence(0.4, 2.2, "", accepted=True)
        accepted, rejected = [candidate], []
        apply_cluster_gate(accepted, rejected, self.turns, result,
                           mode="enforce", min_seconds=1.2)
        self.assertFalse(candidate.accepted)
        self.assertEqual(candidate.diagnostics.get("cluster_tier"), "B")

    def test_mixed_candidate_yields_cropped_child(self):
        result = make_result(self.turns, accepted_spans=[[0.2, 2.2]])
        candidate = CandidateSentence(0.0, 3.5, "", reject_reason="最终局部身份冲突，原混合片段已撤回",
                                      window_min_score=0.7)
        accepted, rejected = [], [candidate]
        changes = apply_cluster_gate(accepted, rejected, self.turns, result,
                                     mode="enforce", min_seconds=1.2)
        self.assertEqual(changes["cropped"], 1)
        self.assertEqual(len(accepted), 1)
        child = accepted[0]
        self.assertAlmostEqual(child.start, 0.2, places=6)
        self.assertAlmostEqual(child.end, 2.2, places=6)
        self.assertTrue(child.diagnostics.get("cluster_cropped"))

    def test_mixed_crop_requires_window_evidence(self):
        result = make_result(self.turns, accepted_spans=[[0.2, 2.2]])
        candidate = CandidateSentence(0.0, 3.5, "", reject_reason="疑似混合说话人，保守舍弃",
                                      window_min_score=0.0)
        accepted, rejected = [], [candidate]
        changes = apply_cluster_gate(accepted, rejected, self.turns, result,
                                     mode="enforce", min_seconds=1.2)
        # 无窗口证据的混合父段不进 Tier A。
        self.assertEqual(changes["cropped"], 0)
        self.assertEqual(changes["borderline"], 1)
        self.assertEqual(accepted, [])

    def test_classifier_conflict_goes_tier_b(self):
        result = make_result(self.turns, accepted_spans=[[0.0, 4.0]])
        candidate = CandidateSentence(0.4, 2.2, "",
                                      reject_reason="局部声学分类器确认持续非目标声音")
        accepted, rejected = [], [candidate]
        apply_cluster_gate(accepted, rejected, self.turns, result,
                           mode="enforce", min_seconds=1.2)
        self.assertEqual(candidate.diagnostics.get("cluster_tier"), "B")
        self.assertFalse(candidate.accepted)


class VerifyTemporalSpanTests(unittest.TestCase):
    def _analyzer(self):
        from extractor.nextgen.cluster_runtime import ClusterAnalyzer
        analyzer = ClusterAnalyzer.__new__(ClusterAnalyzer)
        base = np.ones(8) / np.sqrt(8)
        analyzer.profile = TargetProfile(base, base, 0.55, 0.50)
        analyzer.negatives = []
        analyzer.source_hash = "span-test"
        analyzer.progress = lambda message: None
        return analyzer

    def test_subspan_extraction_keeps_clean_run(self):
        analyzer = self._analyzer()
        turns = [TurnEvidence(0, 0.0, 9.0, 9.0,
                              np.ones(8) / np.sqrt(8), np.ones(8) / np.sqrt(8))]
        candidates = make_result(turns, accepted_spans=[[0.0, 9.0]])
        # 15 秒长轮次：前 6 秒窗口通过，后 9 秒失败 → 只保留干净子段。
        calls = {"n": 0}

        def embed(a, b):
            calls["n"] += 1
            if a < 6.0:
                return np.ones(8) / np.sqrt(8), np.ones(8) / np.sqrt(8)
            return -np.ones(8) / np.sqrt(8), -np.ones(8) / np.sqrt(8)

        analyzer.embed = embed
        result, local = analyzer.verify_temporal(turns, candidates)
        self.assertTrue(result.membership[0])
        spans = result.diagnostics["turn_clean_spans"]["0"]
        self.assertEqual(len(spans), 1)
        self.assertLessEqual(spans[0][1], 7.3)

    def test_short_weak_turn_demoted_to_borderline(self):
        analyzer = self._analyzer()
        turns = [TurnEvidence(0, 0.0, 2.5, 2.5,
                              np.ones(8) / np.sqrt(8), np.ones(8) / np.sqrt(8))]
        candidates = make_result(turns, accepted_spans=[[0.0, 2.5]])

        def embed(a, b):
            # 第一个窗口干净、第二个窗口失败：短句 + 部分窗口未过。
            if a < 0.5:
                return np.ones(8) / np.sqrt(8), np.ones(8) / np.sqrt(8)
            return -np.ones(8) / np.sqrt(8), -np.ones(8) / np.sqrt(8)

        analyzer.embed = embed
        # scores() 给出 1.0；把副通道下限抬到 0.92 使 slack=0.08 落入降级带。
        analyzer.profile = TargetProfile(np.ones(8) / np.sqrt(8),
                                         np.ones(8) / np.sqrt(8), 0.55, 0.92)
        result, _local = analyzer.verify_temporal(turns, candidates)
        self.assertFalse(result.membership[0])
        self.assertTrue(result.borderline_membership[0])
        self.assertEqual(result.diagnostics["turn_clean_spans"]["0"], [])


class ProfileScoreTests(unittest.TestCase):
    def test_multi_prototype_takes_top2_reference(self):
        rng = np.random.default_rng(1)
        base = unit(rng.normal(0, 1, 32))
        twin = axis(0.99, base, rng)
        refs = np.stack([base, base, twin])
        # 向量与特定声线参考 twin 几乎重合，与均值质心只有 0.9 左右。
        query = unit(twin + rng.normal(0, 0.01, 32))
        centroid = unit(refs.mean(axis=0))
        score = profile_score(query, centroid, refs)
        self.assertGreater(score, float(query @ centroid) - 0.01)
        self.assertGreater(score, 0.99)

    def test_single_reference_falls_back_to_centroid(self):
        rng = np.random.default_rng(2)
        base = unit(rng.normal(0, 1, 16))
        score = profile_score(base, base, None)
        self.assertAlmostEqual(score, 1.0, places=6)


class PartialExportTests(unittest.TestCase):
    def _batch(self, root: Path):
        import soundfile as sf
        batch = root / "batch"
        batch.mkdir()
        sr = 16000
        # 4 秒正弦片段（主输出）+ 4 秒片段（供 partial 裁切）
        for name in ("0001_a.wav", "0002_b.wav"):
            sf.write(batch / name, np.sin(np.linspace(0, 440 * 2 * np.pi, sr * 4)) * 0.2, sr)
        manifest = {
            "sentences": [
                {"start": 10.0, "end": 14.0, "accepted": True, "text": "こんにちは",
                 "audio_file": "0001_a.wav", "diagnostics": {}},
                {"start": 20.0, "end": 24.0, "accepted": True, "text": "ありがと",
                 "audio_file": "0002_b.wav", "diagnostics": {"cluster_removed": True}},
            ],
        }
        (batch / "batch_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")
        return batch, manifest

    def test_partial_keep_segments(self):
        from launcher.training_material import partial_keep_segments
        row = {"start": 100.0, "end": 104.0}
        segments = partial_keep_segments(row, {"partial_start": 1.0, "partial_end": 2.5})
        self.assertEqual(segments, [(0.0, 1.0), (2.5, 4.0)])
        self.assertEqual(partial_keep_segments(row, {"partial_start": 0.0, "partial_end": 4.0}), [])

    def test_reviewable_includes_tier_b_rows(self):
        from launcher.training_material import reviewable
        with tempfile.TemporaryDirectory() as directory:
            batch, manifest = self._batch(Path(directory))
            manifest["sentences"][1]["accepted"] = False
            manifest["sentences"][1]["diagnostics"] = {"cluster_tier": "B"}
            (batch / "batch_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")
            rows = reviewable(batch, manifest)
            self.assertEqual(len(rows), 2)

    def test_export_cuts_partial_and_writes_train_list(self):
        from launcher.review_export import export_review_zip
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            batch, manifest = self._batch(root)
            reviews = {
                "1": {"ordinal": 1, "filename": "0001_a.wav", "label": "correct"},
                "2": {"ordinal": 2, "filename": "0002_b.wav", "label": "partial",
                      "partial_start": 0.0, "partial_end": 1.0},
            }
            destination = root / "out.zip"
            count = export_review_zip(batch, manifest, reviews, destination,
                                      gpt_sovits_list=True, speaker="kumiko")
            self.assertEqual(count, 2)
            with zipfile.ZipFile(destination) as archive:
                names = archive.namelist()
                self.assertIn("0001_a.wav", names)
                self.assertTrue(any(name.startswith("partial/") for name in names))
                self.assertIn("train.list", names)
                lines = archive.read("train.list").decode("utf-8").strip().splitlines()
                self.assertEqual(len(lines), 2)
                self.assertTrue(all(line.split("|")[1] == "kumiko" for line in lines))
                self.assertTrue(all(line.split("|")[2] == "JA" for line in lines))
                self.assertIn("こんにちは", lines[0])

    def test_material_package_contains_hard_negative(self):
        from launcher.training_material import export_package
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            batch, manifest = self._batch(root)
            # 用流水线的 save_training_context 生成合法上下文。
            import shutil
            from extractor.training_context import save_training_context
            training = batch / "_training"
            training.mkdir()
            ref = batch / "0001_a.wav"
            shutil.copyfile(ref, training / "reference.wav")
            save_training_context(batch, ref, [training / "reference.wav"])
            reviews = {
                "1": {"ordinal": 1, "filename": "0001_a.wav", "label": "correct"},
                "2": {"ordinal": 2, "filename": "0002_b.wav", "label": "partial",
                      "partial_start": 0.5, "partial_end": 2.0},
            }
            destination = root / "material.zip"
            export_package(batch, manifest, reviews, destination)
            with zipfile.ZipFile(destination) as archive:
                data = json.loads(archive.read("material.json"))
                labels = sorted(item["review"]["label"] for item in data["items"])
                # 原片段 + hard_neg(wrong) + clean_pos(correct)
                self.assertIn("wrong", labels)
                self.assertEqual(labels.count("correct"), 2)
                derived = [item for item in data["items"] if item.get("derived_from")]
                self.assertEqual(len(derived), 2)
                for item in derived:
                    self.assertIn(item["audio"], archive.namelist())


if __name__ == "__main__":
    unittest.main()
