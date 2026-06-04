# Mathematical and Metrics Plan

## Positioning Decision

This paper should not be anchored on the user-association base paper. We will
cite it as adjacent graph-learning work in wireless user association, but the
paper's main framing should be independent:

> Energy-efficient user association and base-station switch-off can be modeled
> as a multilayer network decision problem in which association competition,
> interference coupling, traffic/load pressure, and temporal similarity are
> distinct relation layers.

The contribution is therefore a multilayer network-science formulation, a
layer-aware GNN decision model, and a comparative evaluation against conventional
association heuristics, single-layer graph learning, and oracle search.

## Literature Roles

### Foundational multilayer network science

- Kivela et al., "Multilayer Networks", Journal of Complex Networks, 2014:
  use for terminology and formal distinction between multiplex, multilayer, and
  aggregated networks.
- De Domenico et al., "Mathematical Formulation of Multilayer Networks",
  Physical Review X, 2013: use for tensor/supra-adjacency notation and
  generalized network descriptors such as centrality, modularity, entropy, and
  diffusion.
- Mucha et al., "Community Structure in Time-Dependent, Multiscale, and
  Multiplex Networks", Science, 2010: use for multislice/multilayer modularity
  and interslice coupling.
- Boccaletti et al., "The Structure and Dynamics of Multilayer Networks",
  Physics Reports, 2014: use as a broad review of multiplex/multilayer dynamics.
- De Domenico et al., "Structural Reducibility of Multilayer Networks", Nature
  Communications, 2015: use for layer redundancy, layer aggregation, and the
  idea that aggregation is valid only when information loss is low.
- De Domenico et al., "Ranking in Interconnected Multilayer Networks Reveals
  Versatile Nodes", Nature Communications, 2015: use for multilayer centrality
  and the warning that aggregation can mis-rank important nodes.

### Local DRA-folder papers worth using

- Dynamic-multilayer GCN monitoring paper: useful for intralayer and interlayer
  convolution language and dynamic graph snapshots.
- Attention-based dynamic multilayer GNN paper: useful for supra-adjacency,
  dynamic multilayer snapshots, and attention over graph/time representations.
- Multi-view multi-layer attention GCN paper: useful for multi-view attention
  and the Hilbert-Schmidt independence criterion idea for measuring
  relationship diversity.
- Effects of Graph Convolutions in Multi-layer Networks: useful for theoretical
  justification that graph convolutions can improve classification regimes when
  node features are coupled with graph structure.
- TNSE community-detection paper with semi-supervised joint symmetric NMF:
  useful for graph regularization and convergence-proof style.
- TNSE contrastive multilayer community-detection paper: useful for
  intra-layer/inter-layer feature fusion and layer-aware contrastive objectives.
- TNSE centrality papers: useful for network-science interpretability metrics,
  especially centrality, k-core, PageRank, relationship strength, and gravity
  centrality.
- TNSE network-dissimilarity paper: useful for layer dissimilarity/redundancy
  metrics.

## Proposed Mathematical Model

Let there be `N` UEs and `B` base stations in a cellular snapshot. The UEs are
nodes `V = {u_1, ..., u_N}` and the base stations are decision resources. We
construct a multilayer graph

```text
G = (V, X, {A^(ell)}_{ell in L}),
L = {association, interference, load, temporal}.
```

Each `A^(ell) in R^{N x N}` is nonnegative and row-normalized into a support
matrix `S^(ell)`. In the current implementation:

- `association`: UEs likely to compete for the same strongest BS.
- `interference`: UEs with similar channel-gain profiles.
- `load`: UEs with similar demand and association pressure.
- `temporal`: UEs with spatial-demand similarity modulated by the traffic hour.

For a layer-aware GCN, one hidden layer can be written as

```text
H^(ell) = phi(S^(ell) X W^(ell)),
Z = concat_{ell in L} H^(ell),
P_ua = softmax(Z W_o + b_o).
```

For the attention-weighted version,

```text
alpha_ell = exp(eta_ell) / sum_r exp(eta_r),
Z_att = concat_{ell in L} |L| alpha_ell H^(ell).
```

The current implementation uses learned global layer-attention logits. Later,
we can upgrade this to snapshot-dependent attention if the experiments need
more separation.

The active-cell decision uses model probabilities and BS-level radio/load
features:

```text
s_b = g_theta(pool_i(P_ua[i,b], Z_i, radio/load features)),
a_b = 1 if b is selected as active.
```

The final association repair solves a guarded assignment over the selected
active set and wakes additional BSs only if QoS/load constraints are violated.

## Optimization Objective

The decision-level objective should be stated independently of the supervised
training loss:

```text
min_{y,a} E(y,a)
       + lambda_q sum_i 1[QoS_i(y,a) fails]
       + lambda_l sum_b max(0, rho_b(y,a) - rho_max)

subject to y_i in {1,...,B},
           a_{y_i} = 1,
           a_b in {0,1}.
```

Energy is

```text
E(y,a) = sum_b a_b(P_fix + P_dyn rho_b) + (1-a_b)P_sleep.
```

This makes the paper about constrained energy-aware network control, not merely
classification accuracy.

## Proofs to Include

### Theorem 1: Permutation equivariance of layer-aware UE embeddings

For any UE permutation matrix `Pi`, if `X' = Pi X` and
`S'^(ell) = Pi S^(ell) Pi^T`, then the multilayer GCN embeddings satisfy
`Z' = Pi Z`.

Proof sketch: each layer satisfies
`S'^(ell) X' W^(ell) = Pi S^(ell) Pi^T Pi X W^(ell) = Pi S^(ell) X W^(ell)`.
Pointwise nonlinearities and concatenation commute with `Pi`.

Why it matters: the model does not depend on arbitrary UE indexing.

### Theorem 2: Aggregation can destroy layer-identifiable structure

Define an aggregate graph `S_bar = sum_ell omega_ell S^(ell)`. There exist two
multilayer graphs `{S_1^(ell)}` and `{S_2^(ell)}` with identical `S_bar` but
different individual layers. Any model that receives only `S_bar` must output
the same decision for both graphs, while a layer-aware model can output
different decisions by assigning different parameters to different layers.

Proof sketch: choose two layers with matrices `A` and `B`; graph 1 has
`(S^1=A, S^2=B)` and graph 2 has `(S^1=B, S^2=A)` with equal aggregation
weights. The aggregate is identical, but a layer-aware model with
`W^(1) != W^(2)` can distinguish them.

Why it matters: this is the clean mathematical reason for our multilayer
framing. It avoids overclaiming that a particular ReLU implementation strictly
contains every aggregated GCN.

### Theorem 3: Oracle is an upper-bound reference, not our contribution

Within the finite candidate class searched by the oracle, the oracle energy is
less than or equal to the energy of any other feasible candidate in that class.

Proof sketch: the oracle enumerates all active-BS subsets and evaluates the
assignment score for each subset; the minimum of a finite set is no larger than
any element of that set.

Why it matters: the oracle is a benchmark ceiling. The paper's contribution is
approaching this ceiling with learned multilayer structure.

### Theorem 4: Conditional feasibility of the QoS/load guard

If the guarded assignment routine finds an active set for which all UEs satisfy
RSRP, QoS, and load constraints, it returns a feasible assignment. If no
predicted active set is feasible, the routine progressively activates
additional BSs up to the all-on set.

Proof sketch: the return condition explicitly checks served ratio and load
limit. The active-set expansion loop is monotone: each failed iteration adds an
inactive BS and never removes active BSs.

Important wording: this is a conditional/procedural guarantee, not a proof of
global optimality of the greedy repair.

### Theorem 5: Attention weights are interpretable layer-importance parameters

Because `alpha_ell` is a softmax over layer logits, `alpha_ell >= 0` and
`sum_ell alpha_ell = 1`. Layer entropy

```text
H_alpha = - sum_ell alpha_ell log(alpha_ell)
```

measures whether the model relies broadly on all relation layers or concentrates
on a few layers.

## Metrics to Use

### Primary wireless-control metrics

- Network energy, `E_policy`.
- Energy saving versus all-on:

```text
ES = (E_all_on - E_policy) / E_all_on.
```

- Energy gap versus oracle:

```text
Gap_oracle = (E_policy - E_oracle) / E_oracle.
```

- Served ratio:

```text
SR = (1/N) sum_i 1[UE i satisfies RSRP, QoS, and load constraints].
```

- Active BS count, `sum_b a_b`.
- Maximum load, `max_b rho_b`.
- Mean active load, `mean_{b:a_b=1} rho_b`.

These are already implemented and should remain the main result tables.

### Comparative learning metrics

- Assignment accuracy against oracle labels, but only as a secondary diagnostic.
- Multi-seed mean and standard deviation.
- Stress sweeps over traffic load, number of UEs, BS density, and shadowing.

### Multilayer network-science metrics

- Layer-ablation gain:

```text
Delta_ell = ES_full - ES_without_ell.
```

- Single-layer contribution:

```text
C_ell = ES_only_ell - ES_flat.
```

- Attention entropy `H_alpha`.
- Pairwise layer redundancy, using one or more:

```text
Jaccard_ell,r = |E_ell cap E_r| / |E_ell union E_r|,
D_F(ell,r) = ||S^(ell) - S^(r)||_F / ||S^(ell)||_F,
HSIC(S^(ell), S^(r)).
```

- Optional interpretability:
  - UE multilayer participation coefficient.
  - BS centrality/versatility induced by UE association probabilities.
  - Relationship-strength or gravity-style centrality for identifying heavily
    coupled UEs or congested regions.

## What Needs Implementation Next

1. Multi-seed layer ablation table.
2. Traffic-load stress sweep.
3. Layer redundancy metrics and attention entropy export.
4. Optional snapshot-dependent attention if global attention remains too flat.
5. Overleaf methodology section with the proofs above.

## Important Papers to Download or Verify

We already have local PDFs for several TNSE multilayer papers. The papers below
should be kept in the paper's literature folder with official publisher or
author PDFs where possible:

- Kivela et al., "Multilayer Networks", Journal of Complex Networks, 2014.
- De Domenico et al., "Mathematical Formulation of Multilayer Networks",
  Physical Review X, 2013.
- Mucha et al., "Community Structure in Time-Dependent, Multiscale, and
  Multiplex Networks", Science, 2010.
- Boccaletti et al., "The Structure and Dynamics of Multilayer Networks",
  Physics Reports, 2014.
- De Domenico et al., "Structural Reducibility of Multilayer Networks", Nature
  Communications, 2015.
- De Domenico et al., "Ranking in Interconnected Multilayer Networks Reveals
  Versatile Nodes", Nature Communications, 2015.
- Eisen and Ribeiro, "Optimal Wireless Resource Allocation With Random Edge
  Graph Neural Networks", IEEE Transactions on Signal Processing, 2020.
- Shen et al., "Graph Neural Networks for Scalable Radio Resource Management",
  IEEE JSAC, 2021.

At this point I do not see a blocker that requires a new download before we
continue experiments, but official IEEE copies of the 2025/2026 TNSE papers
should be verified for final citation details.
