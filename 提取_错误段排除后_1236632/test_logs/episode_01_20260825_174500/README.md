# Episode 01 test log

- Batch ID: `20260825_174500_batch_d90730`
- Accepted clips: 20
- Rejected candidates: 285
- Speaker threshold: `0.68`
- Silence merge range: `0.20–0.85 s`
- Minimum standalone output: `1.20 s`
- Singing and overlapping-speaker filters: enabled
- Exclusion-reference groups: 5

Manual listening found no wrong-speaker contamination in the 20 accepted clips.
Clip `0008` has a known missed short tail after an internal pause; the tail contains
about `0.52 s` of voiced audio and was rejected during short-fragment identity review.

`batch_manifest.json` is the complete diagnostic manifest from the run. Audio,
models, references, work caches, and machine-specific paths are not included.
