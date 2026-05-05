"""BFS-based district contiguity checks."""

from __future__ import annotations

from collections import deque
from typing import Hashable, Iterable

import networkx as nx

from gerrydetect.partition import MutablePartition, Partition

NodeId = Hashable


def is_district_connected(graph: nx.Graph, nodes: Iterable[NodeId]) -> bool:
    """True iff the subgraph induced on `nodes` is connected.

    O(|nodes| + |edges induced|).
    """
    nodes_set = set(nodes)
    if not nodes_set:
        return True
    start = next(iter(nodes_set))
    visited: set[NodeId] = {start}
    queue: deque[NodeId] = deque([start])
    while queue:
        n = queue.popleft()
        for nbr in graph.neighbors(n):
            if nbr in nodes_set and nbr not in visited:
                visited.add(nbr)
                queue.append(nbr)
    return len(visited) == len(nodes_set)


def is_district_connected_after_flip(
    partition: Partition | MutablePartition,
    node: NodeId,
    new_district: int,
) -> bool:
    """Would the source district remain connected if `node` were flipped to
    `new_district`?

    The source district is `partition.assignment[node]`. We need to check
    that the rest of the source district stays connected without `node`.
    The destination district stays connected automatically: `node` joins it,
    and because we only call this from MCMC after sampling a *boundary* edge,
    `node` is adjacent to at least one node already in the destination.
    """
    src = partition.assignment[node]
    if src == new_district:
        return True

    # Get the set of source-district nodes that remain.
    if isinstance(partition, MutablePartition):
        remaining = partition.districts[src] - {node}
    else:
        remaining = partition.districts[src] - {node}

    if not remaining:
        # Removing the last node from a district is allowed only if we are
        # comfortable with that district disappearing. For our k-fixed
        # redistricting pipeline this should never happen (the chain would
        # produce k-1 districts after this move), so reject it.
        return False

    return is_district_connected(partition.graph, remaining)
