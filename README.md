# Dataset Agentic Framework

This repository stores the paper data release for the agentic meshing framework.
The payload is archive-based: each dataset is packaged as `tar.zst` and split
into GitHub-safe 95 MiB parts under `archives/`.

## Contents

| Archive | Parts | Combined bytes | Combined SHA-256 |
| --- | ---: | ---: | --- |
| `vpmr_faulty_synthetic_raw_available_500` | 4 | 299380022 | `59477ff52fc9af59f029738c490f65a5b5d7204b53e424bf1bef44621cae0157` |
| `vpmr_strict_100_final_results` | 1 | 39106718 | `0fc4ddb135edc4aa01aba9906f3008ec9d10c4ae229bc503809529bd6bd79373` |
| `thingi10k_100_raw_and_audited_outputs` | 2 | 162516357 | `d281f165f59838fd4817690d8dd44a4b989c849b6a67f52955715d58542a4600` |
| `snes_v2_raw_and_final_results` | 1 | 26461699 | `47b09091eed3531a1a20ed051407bf4a726431a110009c166a42ed5ac89264e3` |
| `physiological_cads_final_results_core` | 6 | 543852856 | `23598c2c0e9afaf80e232826cdf3c647fcad5fddbca0dfe3bf8b47f9429ee536` |
| `physiological_heart_full_mesh_msh` | 52 | 5090254153 | `16e2476a5202215eb3df559da270d9c34758948e430bfcc395767d33f36c50a4` |

## Reconstructing An Archive

Use the command from the matching `manifests/*.archive_manifest.json`, for example:

```bash
cat archives/vpmr_strict_100_final_results/vpmr_strict_100_final_results.tar.zst.part-* > vpmr_strict_100_final_results.tar.zst
sha256sum vpmr_strict_100_final_results.tar.zst
tar --use-compress-program zstd -xf vpmr_strict_100_final_results.tar.zst
```

## Dataset Notes

- VPMR raw faulty synthetic source available in this workspace contains 500 rows
  (`000`-`499`) from `https://github.com/VisualGuidedMeshRepair/dataset.git` at commit `801dc65041bcadac6ade6932e491a1d6ef301b44`. The
  user-requested 874-row source was not present locally during this inventory,
  so all available raw VPMR rows were packaged and the limitation is recorded in
  the VPMR raw archive `SOURCE.json`.
- VPMR strict-100 final results follow the audited full-body gate from
  `meshability_relaxed_summary.json`: 23 accepted/caveat rows are packaged, but
  only rows with a full-body tetrahedral mesh path are labeled as functional
  full-body mesh evidence.
- Thingi10K 100 is packaged as raw inputs plus final audited repair and
  validation outputs. Its audited `accepted_count` is 0 under the full
  tetrahedral mesh gate, so these are diagnostic/failure outputs rather than
  claimed successful meshes.
- Physiological CADs are split into a core archive and a separate full Heart MSH
  archive. Duplicate VTK exports are omitted when MSH/STL/PLY equivalents are
  present.

Generated at: 2026-06-26T15:21:24.615005+00:00
