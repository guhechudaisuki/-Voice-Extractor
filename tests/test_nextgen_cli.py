from __future__ import annotations

from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluation.run_nextgen_scene import run_cached_scene


class CachedSceneCliTests(unittest.TestCase):
    def test_research_mode_runs_bounded_review_but_never_exports(self):
        scene = SimpleNamespace(candidates=("candidate",),
                                audio=SimpleNamespace(source=SimpleNamespace(total_samples=16000)))
        result = SimpleNamespace(selected=(), reviews=(), cancelled=False)
        bank = SimpleNamespace(entries=("target",), digest="b" * 64)
        encoder = SimpleNamespace(digest="a" * 64)
        card = SimpleNamespace(stage="research", backbone_digest=encoder.digest,
                               weights_digest="c" * 64)
        with (patch("evaluation.run_nextgen_scene.load_prepared_scene", return_value=scene),
              patch("evaluation.run_nextgen_scene.WavLMSpeakerFeatures", return_value=encoder),
              patch("evaluation.run_nextgen_scene.load_bundle", return_value=(object(), card, object())),
              patch("evaluation.run_nextgen_scene.prepare_reference_bank", return_value=(bank, ())),
              patch("evaluation.run_nextgen_scene.IdentitySession", return_value=object()),
              patch("evaluation.run_nextgen_scene.run_scene", return_value=result) as run):
            report = run_cached_scene(
                Path("scene"), Path("model"), (Path("reference"),), (), research=True,
            )
        self.assertEqual(report["status"], "research_review_only")
        self.assertEqual(report["selected_count"], 0)
        self.assertEqual(run.call_count, 1)

    def test_normal_mode_rejects_research_checkpoint_before_scene_review(self):
        scene = SimpleNamespace(audio=SimpleNamespace(source=SimpleNamespace(total_samples=16000)))
        with (patch("evaluation.run_nextgen_scene.load_prepared_scene", return_value=scene),
              patch("evaluation.run_nextgen_scene.WavLMSpeakerFeatures", return_value=object()),
              patch("evaluation.run_nextgen_scene.load_bundle",
                    return_value=(object(), SimpleNamespace(stage="research"), object())),
              patch("evaluation.run_nextgen_scene.run_scene") as run):
            with self.assertRaisesRegex(ValueError, "validated"):
                run_cached_scene(Path("scene"), Path("model"), (Path("reference"),), ())
        run.assert_not_called()

    def test_cached_full_episode_cannot_bypass_short_scene_limit(self):
        scene = SimpleNamespace(candidates=("candidate",),
                                audio=SimpleNamespace(source=SimpleNamespace(total_samples=91 * 16000)))
        with (patch("evaluation.run_nextgen_scene.load_prepared_scene", return_value=scene),
              patch("evaluation.run_nextgen_scene.WavLMSpeakerFeatures") as encoder):
            with self.assertRaisesRegex(ValueError, "90 seconds"):
                run_cached_scene(Path("scene"), Path("model"), (Path("reference"),), ())
        encoder.assert_not_called()


if __name__ == "__main__":
    unittest.main()
