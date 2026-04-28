"""Multi-chain MCMC runner with mixing diagnostics.

Single-chain MCMC inference rests on a leap of faith: that the chain has
explored a representative region of the plan space. Running multiple chains
from independent random seeds and comparing them gives us a way to *test*
that assumption rather than assume it. If two chains land in completely
different parts of the plan space, R-hat for any metric will be large and
we know the single-chain answer is unreliable.

Key idea — every method on `MultiChainResult` accepts a metric function and
returns a per-chain × per-sample value array. From those arrays we derive:

- `metric_trajectories(fn)` → shape (n_chains, n_samples)
- `rhat(fn)` → R-hat across chains
- `effective_sample_size(fn)` → ESS pooled across chains
- `pooled_samples()` → all chains concatenated for ensemble-level analysis
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterable

import networkx as nx
import numpy as np

from gerrydetect import mcmc
from gerrydetect.diagnostics import effective_sample_size, gelman_rubin
from gerrydetect.partition import Partition

MetricFn = Callable[[Partition], float]


@dataclass
class MultiChainResult:
    """Output of a multi-chain MCMC run.

    Each entry of `chains` is the list of saved Partitions for one chain,
    in chronological order; `stats` holds the per-chain MCMCStats.
    """

    chains: list[list[Partition]]
    stats: list[mcmc.MCMCStats]
    seeds: list[int] = field(default_factory=list)

    @property
    def n_chains(self) -> int:
        return len(self.chains)

    @property
    def n_samples_per_chain(self) -> int:
        if not self.chains:
            return 0
        return len(self.chains[0])

    def metric_trajectories(self, fn: MetricFn) -> np.ndarray:
        """Return shape (n_chains, n_samples) array of metric values."""
        return np.array(
            [[fn(p) for p in chain] for chain in self.chains],
            dtype=float,
        )

    def rhat(self, fn: MetricFn) -> float:
        """Brooks-Gelman R-hat across chains for the given metric."""
        traj = self.metric_trajectories(fn)
        return gelman_rubin(traj)

    def effective_sample_size(self, fn: MetricFn) -> float:
        """ESS pooled across chains: ESS per chain summed."""
        traj = self.metric_trajectories(fn)
        return float(sum(effective_sample_size(traj[i]) for i in range(self.n_chains)))

    def pooled_samples(self) -> list[Partition]:
        """All chains concatenated — the ensemble used for outlier analysis."""
        out: list[Partition] = []
        for chain in self.chains:
            out.extend(chain)
        return out


def run_multichain(
    graph: nx.Graph,
    seed_assignment: dict,
    n_chains: int = 4,
    n_steps: int = 1000,
    pop_tol: float = 0.02,
    lag: int = 100,
    burn_in: int = 10_000,
    seeds: Iterable[int] | None = None,
    show_progress: bool = True,
) -> MultiChainResult:
    """Run `n_chains` independent MCMC chains and bundle the results.

    All chains start from the same `seed_assignment` (e.g. spectral bisection)
    but use different random number generator seeds, so they explore the
    plan space along independent trajectories.
    """
    if seeds is None:
        seeds = list(range(n_chains))
    seeds = list(seeds)
    if len(seeds) != n_chains:
        raise ValueError(f"seeds length {len(seeds)} != n_chains {n_chains}")

    chains: list[list[Partition]] = []
    stats: list[mcmc.MCMCStats] = []
    for chain_idx, seed in enumerate(seeds):
        samples, st = mcmc.run(
            graph,
            seed_assignment=seed_assignment,
            n_steps=n_steps,
            pop_tol=pop_tol,
            lag=lag,
            burn_in=burn_in,
            seed=seed,
            show_progress=show_progress,
        )
        chains.append(samples)
        stats.append(st)
    return MultiChainResult(chains=chains, stats=stats, seeds=seeds)
