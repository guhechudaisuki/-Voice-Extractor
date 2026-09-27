# Prepared-scene training dataset format

`training/build_nextgen_prepared.py` converts existing `prepared_scene` caches and explicit
human annotations into frozen safetensors examples. It does not run media preparation,
download a model, infer a speaker from subtitles, or use model predictions as labels.

## Command

```powershell
python training/build_nextgen_prepared.py dataset.json output-dataset `
  --backbone X:\local-models\wavlm-base-plus-sv --device cuda:0
```

The backbone must be a complete local checkpoint. The destination must not already exist.
The builder stages beside the destination and publishes the directory only after every
record, digest, and manifest can be read back successfully.

## Schema 1

```json
{
  "schema": 1,
  "examples": [
    {
      "id": "scene-001-target-a",
      "split": "train",
      "prepared_scene": "prepared/query-001",
      "candidate_key": "<complete immutable Candidate.key>",
      "query_source_group": "episode-001-master",
      "query_speaker_ids": ["actor-a", "actor-b"],
      "license_reference": "rights ledger entry 101",
      "use_allowed": true,
      "references": [
        {
          "role": "target",
          "prepared_scene": "prepared/reference-a",
          "source_group": "episode-004-master",
          "speaker_id": "actor-a",
          "license_reference": "rights ledger entry 205",
          "use_allowed": true
        },
        {
          "role": "background-actor",
          "prepared_scene": "prepared/reference-b",
          "source_group": "episode-007-master",
          "speaker_id": "actor-c",
          "license_reference": "rights ledger entry 311",
          "use_allowed": true
        }
      ],
      "annotation": {
        "output": {"start": 16000, "end": 32000},
        "target": [{"start": 16000, "end": 32000}],
        "other": [],
        "singing": [],
        "overlap": [],
        "change": [],
        "unknown": [],
        "observable": [{"start": 16000, "end": 32000}],
        "unobservable": [],
        "uncertain": [],
        "certain": [{"start": 16000, "end": 32000}],
        "complete_timeline": true,
        "purity": 1,
        "start_complete": 1,
        "end_complete": 1
      }
    }
  ]
}
```

All scene paths are resolved relative to the JSON file unless fully absolute. Drive-relative
and root-only Windows paths are rejected. `id` is restricted to a short portable filename.
`split` is exactly one of `train`, `development`, `calibration`, or `test`, and every split
must contain at least one example.

`candidate_key` must equal the complete cached `Candidate.key`; matching only its time span
is insufficient. Annotation intervals are absolute, half-open integer sample intervals on the
prepared 16 kHz source timeline. `annotation.output` must equal the selected candidate output.
Allowed interval fields are `target`, `other`, `singing`, `overlap`, `change`, `unknown`,
`observable`, `unobservable`, `uncertain`, and `certain`. Omitted interval fields are empty;
when `complete_timeline` is false, unlisted activity remains unknown. Labels are only
`-1`, `0`, or `1`. Unknown keys, including subtitle text and model predictions, are rejected.

At least one reference has role `target`. Any other nonempty role is an exclusion group.
`speaker_id` is the annotator's assertion that every usable clean speech island in that
reference scene belongs to that one person. A role cannot name multiple speakers, and one
speaker cannot be both the target and an exclusion. Query and reference source groups and
actual recordings must be independent.

Every query and reference requires a nonempty rights record and `use_allowed: true`, including
calibration and test assets. Before encoding, the builder rejects source-group, speaker, or
raw/stem-content reuse across splits.

## Output

The destination contains one `manifest.json` and one
`features/<split>/<id>.safetensors` shard per example. Standard manifest records contain all
source, speaker, content, feature, and backbone digests. The top-level `provenance` section
retains the exact candidate key, human annotation and its digest, per-asset role/source/speaker
metadata, rights references, and prepared-scene paths. `config_digest` fingerprints the input
JSON. All natural examples are recorded as human-labelled and non-synthetic.
