# Data Notes

The experiments use reproducible synthetic dense-cellular snapshots generated
from the configuration files in `configs/`. The current scripts generate
snapshots in memory and save aggregate outputs under `results/`, including
metrics, summaries, sweep tables, runtime benchmarks, and figure data.

If future scripts persist full generated snapshot datasets, they should write
them under `data/generated/`. That folder is ignored by default unless the
generated data are intentionally promoted for public release.
