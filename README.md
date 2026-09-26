# MATRSS: Multi-Agent Trust and Reputation Simulation System

Decentralized trust under adversarial and non-stationary service providers. Client agents keep
private trust ledgers (no global registry, no shared reputation) and must learn which providers
to delegate to while some providers are malicious, noisy, or degrade after building trust.

> **Provenance.** This is a Python reimplementation (v2) of my 2025–26 MATRSS project, originally
> built in Java. The original codebase was lost; the design follows the project report, with
> methodological changes listed below. All results in this repository are produced by the code here.

## Model

- **Providers**: Bernoulli processes with success probability `p_j(t)`: honest (0.95), malicious
  (0.05), noisy (0.5), degrading (0.95 → 0.05 after an onset), recovering.
- **Trust models** (`src/matrss/trust.py`)
  - *Asymmetric additive* (v1): `T ← min(1, T+R)` / `max(0, T−P)`. Scores rise iff `p > P/(R+P)`.
  - *Beta reputation* with forgetting λ: posterior mean `(r+a₀)/(r+s+a₀+b₀)`, evidence discounted by λ.
- **Selection** (`src/matrss/selection.py`): greedy (random tie-break), ε-greedy, Thompson sampling.

## Methodology changes from v1

1. **Common random numbers.** Environment outcomes are pre-drawn (`U[t,c,j] < p_j(t)`) from an RNG
   stream independent of client decisions, so all conditions face the same world per seed.
2. **Multiple seeds + paired bootstrap CIs** instead of single-run point estimates.
3. **Regret against a per-round oracle** as the primary metric, alongside success rate.
4. **Principled baselines** (Beta reputation, Thompson sampling) next to the v1 heuristic.
5. **Vectorized clients.** Each trust model stores all clients' ledgers as a `(clients, providers)`
   array; row `c` is written only from client `c`'s own outcomes and read only by client `c`'s
   policy (enforced by `test_ledgers_are_private`), so decentralization is preserved while every
   round is one NumPy step. A full 30-seed experiment runs in ~5 s instead of ~34 s.

## Metrics

| Metric | Meaning |
|---|---|
| `success_rate` | share of all delegated tasks that succeeded |
| `expected_success` | mean true `p` of the chosen providers (same as above, minus outcome noise) |
| `mean_regret_per_task` | per-round oracle `max_j p_j(t)` minus `p` of the chosen provider |
| `malicious_selection_rate` | share of tasks delegated to malicious providers |
| `malicious_isolation_rounds` | rounds until clients route to malicious providers at ≤ `isolation_threshold` × the uniform-random rate, averaged over the next `isolation_window` rounds |
| `degraded_exposure_rate` | share of tasks sent to a degrading provider while its current `p(t) < 0.5` |
| `degraded_adaptation_rounds` | rounds from a degrading provider's drop below `p = 0.5` until it is isolated (same criterion) |

Isolation is measured from routing behaviour, not trust scores: a provider nobody selects keeps a
stale score, so "trust fell below X" says little about whether clients stopped using it.

## Quickstart

```powershell
.\setup_env.ps1                             # creates .venv and installs requirements.txt
.venv\Scripts\activate
pytest -q
matrss run configs/baseline.yaml            # → results/baseline/{per_seed,summary}.csv
python scripts/plot_trust.py results/baseline/trust_trajectory_asym_eps0.1_seed0.csv
```

`requirements.txt` is the library list (like Maven's `pom.xml`): it pins the tested version of
every dependency. `pyproject.toml` holds the allowed version ranges. Add new libraries to both.

## Layout

```
configs/            experiment definitions (YAML)
src/matrss/         behaviors, environment (CRN), trust, selection, engine, metrics, cli
tests/              trust invariants, ledger privacy, selection, metrics, reproducibility
scripts/            plotting
```

## Roadmap

- Gossip / shared reputation (EigenTrust-style) and Sybil-collusion attacks on it
- Sliding-window and discounted UCB for non-stationary providers
- Context-dependent (task-typed) trust

## Citation

See `CITATION.cff`.
