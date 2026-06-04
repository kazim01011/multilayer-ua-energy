# Experiment Notes

## Current Framing

The first experiment implements dense-cell user association and cell switch-off
as an independent multilayer network-science problem over simulated cellular
snapshots. The adjacent GAT user-association paper is cited as related work, but
our paper is not structured as a reproduction of that paper. The oracle
enumerates active base-station subsets and assigns UEs to minimize network
energy while preserving QoS and load constraints.

## Initial Result

The first completed run is `configs/initial.json`.

Earlier observation:

- The simulator and feasibility metrics are stable.
- Direct UE-to-BS label prediction is not enough for this paper because it
  learns association preferences but does not explicitly learn cell switch-off.

Updated two-stage result:

- The oracle switches off most cells, using about 2.55 active BSs on average.
- Conventional RSRP/SINR/load-aware heuristics keep all 7 BSs active and save
  only about 16-18% energy relative to the all-on network.
- The corrected two-stage learned controller predicts active-cell scores first
  and then applies a QoS/load-aware association repair.
- ML-GCN and attention ML-GCN keep QoS at 1.0, use about 3.25 active BSs, and
  save about 55% energy relative to the all-on network.
- This is within about 25% energy gap of the oracle while strongly outperforming
  single-shot association heuristics.

## Next Experiment

The current model uses a two-stage controller:

1. Predict active-cell scores from multilayer graph embeddings.
2. Assign UEs to the predicted active set using the QoS/load guard.

This better matches the scientific contribution: multilayer graph learning for
joint user association and cell switch-off.

Future refinements:

- Add multi-seed evaluation.
- Add load/traffic stress sweeps.
- Add ablation models for individual graph layers.
- Improve attention ML-GCN separation, since the first run gives similar
  performance for fixed and attention-weighted multilayer GCNs.

## Multi-Seed Initial Benchmark

Completed `configs/multiseed_initial.json` with seeds 17, 23, and 31.

Mean result:

- Oracle: 63.2% energy saving, 2.52 active BSs, served ratio 1.0.
- ML-GCN: 55.2% energy saving, 3.18 active BSs, served ratio 1.0.
- Attention ML-GCN: 54.7% energy saving, 3.23 active BSs, served ratio 1.0.
- Flat MLP: 54.1% energy saving, 3.27 active BSs, served ratio 1.0.
- Aggregated GCN: 54.1% energy saving, 3.30 active BSs, served ratio 1.0.
- RSRP/SINR: 18.0% energy saving, 7.0 active BSs, served ratio 0.995.
- Load-aware heuristic: 17.1% energy saving, 7.0 active BSs, served ratio 0.995.

Interpretation:

- The two-stage learned controller is stable across seeds.
- Learned controllers strongly outperform conventional user-association
  heuristics because they learn cell switch-off behavior.
- ML-GCN is currently the strongest learned method by mean energy.
- Attention ML-GCN is close, but its advantage is not yet clear. This needs
  layer-specific ablation/stress settings where relation weighting matters more.

## Single-Seed Layer Ablation

Completed `configs/ablation_initial.json` for seed 17.

Key observation:

- Full ML-GCN and attention ML-GCN save about 55.4% energy.
- Single-layer and leave-one-layer-out variants remain strong but are generally
  below the full multilayer model.
- Removing the load layer has the largest penalty in this seed, suggesting that
  traffic/load relation structure is important for energy-aware cell switch-off.

Next refinements:

- Run multi-seed ablation.
- Add high-load and low-load stress sweeps.
- Add a clearer attention setting or attention regularization if the attention
  model remains too similar to fixed ML-GCN.
