# Modeling data release 2026-04-29

Companion data for the consolidated modeling parquet. Extract to `ML/report/data/` to use the team's single ready-to-model dataset.

## Tarball

- **File:** `pontus-modeling-data-2026-04-29.tar.gz`
- **Size:** 264 MB compressed, 303 MB extracted
- **SHA-256:** `fa27aa29d3342cc1dc049dea432fdf48607ee07ad2016ed3e9a362a4ecf49c38`

## Contents

| File | Size | SHA-256 |
|------|------|---------|
| `consolidated_modeling_data.parquet` | 317.5 MB | `85f424975f8384590d8cd781d9353e2977a6cf336fa4bda01b1310f65e98cc87` |
| `consolidated_modeling_data.info.json` | 671 B | `ac5a21a31985f2554a0bab2bed493fcbcc532566950e4672898a92b24f34697d` |
| `README.md` | 2.4 KB | `72d48efe9a14753b825557f5a69d8676a2caa6781fbff7c28a7c8a8715020f66` |
| `MISSING_DATA.md` | 6.1 KB | `b4a0e901b604553741af0e7a51a748de7d4c4c9437aecbc9c8276477f39a7875` |

## Dataset summary

- 1,371,180 rows (1,114,003 train / 257,177 test, by `split` column)
- 87 columns: 5 meta/id, 70 core features, 12 wallet features
- Target: `bet_correct` (int64, binary, ~50.3% positive)
- Build date: 2026-04-29

## Pull / verify / extract

```bash
cd ML/report
curl -L -o /tmp/pontus-modeling-data-2026-04-29.tar.gz \
  https://github.com/PontusNeumann/CBS-Spring26/releases/download/pontus-modeling-data-2026-04-29/pontus-modeling-data-2026-04-29.tar.gz
echo "fa27aa29d3342cc1dc049dea432fdf48607ee07ad2016ed3e9a362a4ecf49c38  /tmp/pontus-modeling-data-2026-04-29.tar.gz" | shasum -a 256 -c
tar -xzf /tmp/pontus-modeling-data-2026-04-29.tar.gz -C data/
```

Then load with:

```python
import pandas as pd
df = pd.read_parquet("data/consolidated_modeling_data.parquet")
```

Full load instructions in `data/README.md`.
