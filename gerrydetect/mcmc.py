"""Single-flip Metropolis MCMC over the space of valid redistricting plans.

Following the proposal: at each step we sample a boundary edge uniformly at
random and propose flipping the smaller-population endpoint into the
larger-population district. The proposal is accepted iff:
1. it preserves contiguity (the source district stays connected without the
   flipped node), and
2. it preserves population balance (no district drifts farther than `pop_tol`
   from the ideal population).

After a burn-in period we record a `Partition` snapshot every `lag` steps.

This is the canonical "flip step" sampler; it is known to mix slowly compared
to recombination moves but is conceptually clean (each move is a single
swap on one boundary edge — the natural graph operation) and lets us write
the entire sampler in ~150 lines of clear Python.
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass

import networkx as nx
from tqdm import tqdm

from gerrydetect.contiguity import is_district_connected_after_flip
from gerrydetect.partition import MutablePartition, Partition

log = logging.getLogger(__name__)


@dataclass
class MCMCStats:
    proposed: int = 0
    rejected_pop: int = 0      # rejected because of population imbalance
    rejected_contig: int = 0   # rejected because of contiguity
    accepted: int = 0

    def acceptance_rate(self) -> float:
        return self.accepted / self.proposed if self.proposed else 0.0


def _ideal_pop(p: MutablePartition) -> float:
    return p.total_pop / max(p.num_districts, 1)


def _within_pop_tol(
    p: MutablePartition,
    src: int,
    dst: int,
    delta_pop: float,
    pop_tol: float,
) -> bool:
    """Check that flipping a node of weight `delta_pop` from `src` to `dst`
    leaves both districts within `pop_tol` of the ideal population.

    `delta_pop` is the population of the flipped node (positive number).
    """
    ideal = _ideal_pop(p)
    new_src = p.district_pop[src] - delta_pop
    new_dst = p.district_pop[dst] + delta_pop
    bound_lo = ideal * (1 - pop_tol)
    bound_hi = ideal * (1 + pop_tol)
    return bound_lo <= new_src <= bound_hi and bound_lo <= new_dst <= bound_hi


def run(
    graph: nx.Graph,
    seed_assignment: dict,
    n_steps: int,
    pop_tol: float = 0.02,
    lag: int = 100,
    burn_in: int = 10_000,
    seed: int = 42,
    show_progress: bool = True,
) -> tuple[list[Partition], MCMCStats]:
    """Run the single-flip Metropolis chain.

    Args:
        graph: precinct adjacency graph with node attrs `pop`, `votes_d`,
            `votes_r`.
        seed_assignment: starting partition (e.g. from spectral bisection).
        n_steps: number of *saved* samples to produce.
        pop_tol: max fractional deviation from ideal district population
            (proposal default 0.02 = ±2%).
        lag: record one sample every `lag` accepted-or-rejected attempts after
            burn-in.
        burn_in: number of warmup steps before recording.
        seed: RNG seed.
        show_progress: tqdm progress bar.

    Returns:
        (samples, stats). `samples` is a list of immutable `Partition`s of
        length `n_steps`. `stats` reports proposal acceptance counts.
    """
    rng = random.Random(seed)
    P = MutablePartition(graph, seed_assignment)
    stats = MCMCStats()

    total_steps = burn_in + n_steps * lag
    samples: list[Partition] = []
    saved = 0

    # We sample boundary edges by reservoir over P.boundary_edges. Converting
    # to a tuple every step would be expensive at PA scale; instead we keep a
    # cached list and rebuild it every `rebuild_every` accepted flips.
    boundary_list: list[frozenset] = list(P.boundary_edges)
    rebuild_every = max(50, len(graph) // 20)
    flips_since_rebuild = 0

    iterator = range(total_steps)
    if show_progress:
        iterator = tqdm(iterator, desc="MCMC", unit="step")

    for step in iterator:
        if not boundary_list:
            log.warning("No boundary edges; stopping early at step %d", step)
            break

        edge = boundary_list[rng.randrange(len(boundary_list))]
        u, v = tuple(edge)
        stats.proposed += 1

        src_u, src_v = P.assignment[u], P.assignment[v]

        # Pick which endpoint flips so that the move *transfers population
        # from the larger district to the smaller* — a mild restorative
        # bias toward balance. (The endpoint that flips is the one currently
        # in the larger district; it joins the smaller district.)
        if P.district_pop[src_u] >= P.district_pop[src_v]:
            flip_node, src, dst = u, src_u, src_v
        else:
            flip_node, src, dst = v, src_v, src_u

        node_pop = P.node_pop(flip_node)

        # Population check.
        if not _within_pop_tol(P, src, dst, node_pop, pop_tol):
            stats.rejected_pop += 1
            if step >= burn_in and ((step - burn_in) % lag == 0):
                samples.append(P.snapshot())
                saved += 1
            continue

        # Contiguity check.
        if not is_district_connected_after_flip(P, flip_node, dst):
            stats.rejected_contig += 1
            if step >= burn_in and ((step - burn_in) % lag == 0):
                samples.append(P.snapshot())
                saved += 1
            continue

        # Accept.
        P.flip(flip_node, dst)
        stats.accepted += 1
        flips_since_rebuild += 1
        if flips_since_rebuild >= rebuild_every:
            boundary_list = list(P.boundary_edges)
            flips_since_rebuild = 0

        if step >= burn_in and ((step - burn_in) % lag == 0):
            samples.append(P.snapshot())
            saved += 1

        if saved >= n_steps:
            break

    # Trim in case we recorded one extra at the boundary.
    samples = samples[:n_steps]
    return samples, stats
