"""Partition abstractions.

A `Partition` wraps a graph + node-to-district assignment and exposes
per-district aggregates (population, vote totals, boundary edges). It is
the single object every metric and every sampler operates on.

Two flavours:

- `Partition` — immutable. `flip(node, new_district)` returns a fresh
  `Partition`. Convenient and hashable; cheap to snapshot. Use this for
  ensemble samples.
- `MutablePartition` — same data, but `flip()` mutates in place and updates
  per-district aggregates incrementally in O(deg(node)). Use this inside the
  MCMC inner loop where we propose hundreds of thousands of flips.

The two are kept consistent by a single set of derivation rules; tests verify
that `MutablePartition` matches `Partition.from_assignment(...)` after every
operation on small graphs.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Hashable, Iterable

import networkx as nx

NodeId = Hashable
DistrictId = int


def _node_attr(graph: nx.Graph, node: NodeId, attr: str, default: float = 0.0) -> float:
    return float(graph.nodes[node].get(attr, default))


@dataclass(frozen=True)
class Partition:
    """Immutable partition of `graph` into districts.

    The class lazily computes and caches per-district aggregates the first
    time they are accessed. `flip()` returns a new `Partition`; the original
    is not modified.
    """

    graph: nx.Graph
    assignment: dict[NodeId, DistrictId]

    def __post_init__(self) -> None:
        # Validate: every node has an assignment.
        missing = set(self.graph.nodes) - set(self.assignment)
        if missing:
            raise ValueError(f"Partition assignment missing {len(missing)} nodes")

    # ----- lazy aggregates -----

    @property
    def districts(self) -> dict[DistrictId, set[NodeId]]:
        cached = self.__dict__.get("_districts")
        if cached is None:
            d: dict[DistrictId, set[NodeId]] = defaultdict(set)
            for node, dist in self.assignment.items():
                d[dist].add(node)
            cached = dict(d)
            object.__setattr__(self, "_districts", cached)
        return cached

    @property
    def num_districts(self) -> int:
        return len(self.districts)

    def _district_sum(self, attr: str) -> dict[DistrictId, float]:
        cache_key = f"_sum_{attr}"
        cached = self.__dict__.get(cache_key)
        if cached is None:
            totals: dict[DistrictId, float] = defaultdict(float)
            for node, dist in self.assignment.items():
                totals[dist] += _node_attr(self.graph, node, attr)
            cached = dict(totals)
            object.__setattr__(self, cache_key, cached)
        return cached

    @property
    def district_pop(self) -> dict[DistrictId, float]:
        return self._district_sum("pop")

    @property
    def district_votes_d(self) -> dict[DistrictId, float]:
        return self._district_sum("votes_d")

    @property
    def district_votes_r(self) -> dict[DistrictId, float]:
        return self._district_sum("votes_r")

    @property
    def total_pop(self) -> float:
        return sum(self.district_pop.values())

    @property
    def boundary_edges(self) -> list[tuple[NodeId, NodeId]]:
        cached = self.__dict__.get("_boundary_edges")
        if cached is None:
            cached = [
                (u, v)
                for u, v in self.graph.edges
                if self.assignment[u] != self.assignment[v]
            ]
            object.__setattr__(self, "_boundary_edges", cached)
        return cached

    @property
    def cut_size(self) -> int:
        return len(self.boundary_edges)

    # ----- operations -----

    def flip(self, node: NodeId, new_district: DistrictId) -> "Partition":
        """Return a new `Partition` with `node` reassigned. Pure."""
        if self.assignment[node] == new_district:
            return self
        new_assignment = dict(self.assignment)
        new_assignment[node] = new_district
        return Partition(self.graph, new_assignment)

    def to_array(self, node_order: Iterable[NodeId]) -> list[DistrictId]:
        """Serialize assignment as a list ordered by `node_order`."""
        return [self.assignment[n] for n in node_order]

    @classmethod
    def from_array(
        cls,
        graph: nx.Graph,
        node_order: Iterable[NodeId],
        assignment_array: Iterable[DistrictId],
    ) -> "Partition":
        """Inverse of `to_array`: rebuild a Partition from a flat array."""
        assignment = dict(zip(node_order, assignment_array, strict=True))
        return cls(graph, assignment)


class MutablePartition:
    """Same data as `Partition`, but `flip()` mutates in place.

    Maintains incremental updates to per-district aggregates and the boundary
    edge set. Designed for the MCMC inner loop where we may attempt millions
    of flips and need O(deg(node)) per accepted flip.
    """

    def __init__(self, graph: nx.Graph, assignment: dict[NodeId, DistrictId]) -> None:
        self.graph = graph
        self.assignment: dict[NodeId, DistrictId] = dict(assignment)

        # Pre-extract node attributes for hot loop speed.
        self._pop: dict[NodeId, float] = {
            n: _node_attr(graph, n, "pop") for n in graph.nodes
        }
        self._vd: dict[NodeId, float] = {
            n: _node_attr(graph, n, "votes_d") for n in graph.nodes
        }
        self._vr: dict[NodeId, float] = {
            n: _node_attr(graph, n, "votes_r") for n in graph.nodes
        }

        self.districts: dict[DistrictId, set[NodeId]] = defaultdict(set)
        self.district_pop: dict[DistrictId, float] = defaultdict(float)
        self.district_votes_d: dict[DistrictId, float] = defaultdict(float)
        self.district_votes_r: dict[DistrictId, float] = defaultdict(float)
        for node, dist in self.assignment.items():
            self.districts[dist].add(node)
            self.district_pop[dist] += self._pop[node]
            self.district_votes_d[dist] += self._vd[node]
            self.district_votes_r[dist] += self._vr[node]

        # Boundary edges as a set for O(1) add/remove. We store each edge as
        # a frozenset of its two endpoints so {u, v} == {v, u} regardless of
        # node ID type. Frozenset of two hashable items is itself hashable.
        self.boundary_edges: set[frozenset[NodeId]] = set()
        for u, v in graph.edges:
            if self.assignment[u] != self.assignment[v]:
                self.boundary_edges.add(frozenset((u, v)))

    @property
    def num_districts(self) -> int:
        return len(self.districts)

    @property
    def total_pop(self) -> float:
        return sum(self.district_pop.values())

    def node_pop(self, node: NodeId) -> float:
        return self._pop[node]

    def flip(self, node: NodeId, new_district: DistrictId) -> None:
        """Mutate in place. Caller is responsible for validity checks
        (population balance, contiguity); this method just applies the move
        and updates aggregates.
        """
        old = self.assignment[node]
        if old == new_district:
            return

        # district membership
        self.districts[old].discard(node)
        if not self.districts[old]:
            del self.districts[old]
        self.districts[new_district].add(node)

        # aggregates
        self.district_pop[old] -= self._pop[node]
        self.district_pop[new_district] += self._pop[node]
        self.district_votes_d[old] -= self._vd[node]
        self.district_votes_d[new_district] += self._vd[node]
        self.district_votes_r[old] -= self._vr[node]
        self.district_votes_r[new_district] += self._vr[node]
        if self.district_pop[old] == 0 and old not in self.districts:
            del self.district_pop[old]
            del self.district_votes_d[old]
            del self.district_votes_r[old]

        # assignment
        self.assignment[node] = new_district

        # boundary edge updates: an edge (node, nbr) is a boundary edge iff
        # the two endpoints belong to different districts.
        for nbr in self.graph.neighbors(node):
            edge = frozenset((node, nbr))
            if self.assignment[nbr] == new_district:
                self.boundary_edges.discard(edge)
            else:
                self.boundary_edges.add(edge)

    def snapshot(self) -> Partition:
        """Freeze current state into an immutable `Partition`."""
        return Partition(self.graph, dict(self.assignment))

    def cut_size(self) -> int:
        return len(self.boundary_edges)
