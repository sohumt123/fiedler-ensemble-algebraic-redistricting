"""Partition / MutablePartition invariants on hand-checkable graphs."""

from __future__ import annotations

import pytest

from gerrydetect.partition import MutablePartition, Partition


def test_partition_district_pop_sums_to_total(four_cycle):
    p = Partition(four_cycle, {0: 0, 1: 0, 2: 1, 3: 1})
    assert p.district_pop == {0: 200.0, 1: 200.0}
    assert p.total_pop == 400.0


def test_partition_boundary_edges_count(four_cycle):
    # split 0,1 vs 2,3 — boundary edges are (1,2) and (3,0): 2 cuts
    p = Partition(four_cycle, {0: 0, 1: 0, 2: 1, 3: 1})
    assert p.cut_size == 2
    assert len(p.boundary_edges) == 2


def test_partition_flip_returns_new_object(four_cycle):
    p = Partition(four_cycle, {0: 0, 1: 0, 2: 1, 3: 1})
    p2 = p.flip(0, 1)
    assert p.assignment[0] == 0
    assert p2.assignment[0] == 1
    assert p is not p2


def test_partition_flip_to_same_district_is_noop(four_cycle):
    p = Partition(four_cycle, {0: 0, 1: 0, 2: 1, 3: 1})
    assert p.flip(0, 0) is p


def test_partition_missing_assignment_raises(four_cycle):
    with pytest.raises(ValueError):
        Partition(four_cycle, {0: 0, 1: 0, 2: 1})  # missing node 3


def test_mutable_partition_matches_immutable_after_flip(four_cycle):
    """The two implementations must agree on every aggregate after a flip."""
    initial = {0: 0, 1: 0, 2: 1, 3: 1}
    immut = Partition(four_cycle, initial).flip(0, 1)
    mut = MutablePartition(four_cycle, initial)
    mut.flip(0, 1)

    assert mut.assignment == immut.assignment
    # district populations match
    assert dict(mut.district_pop) == immut.district_pop
    # boundary edge sets — Partition uses tuples, MutablePartition frozensets;
    # compare by canonical frozenset form.
    immut_b = {frozenset(e) for e in immut.boundary_edges}
    assert mut.boundary_edges == immut_b


def test_mutable_partition_invariants_under_repeated_flips(six_grid):
    """Population + assignment invariants hold after every flip."""
    # Start: split into two halves
    initial = {0: 0, 1: 0, 2: 0, 3: 1, 4: 1, 5: 1}
    p = MutablePartition(six_grid, initial)
    initial_total_pop = p.total_pop

    # Walk through some flips that all preserve contiguity in this 2x3 grid.
    flips = [(2, 1), (1, 1), (4, 0)]
    for node, dst in flips:
        p.flip(node, dst)
        # assignment dict and districts dict stay in sync
        for d, members in p.districts.items():
            for m in members:
                assert p.assignment[m] == d
        # total population is conserved
        assert p.total_pop == initial_total_pop
        # boundary edges set has no entries that are interior
        for edge in p.boundary_edges:
            u, v = tuple(edge)
            assert p.assignment[u] != p.assignment[v]


def test_partition_to_array_roundtrip(four_cycle):
    initial = {0: 0, 1: 0, 2: 1, 3: 1}
    p = Partition(four_cycle, initial)
    order = [0, 1, 2, 3]
    arr = p.to_array(order)
    p2 = Partition.from_array(four_cycle, order, arr)
    assert p2.assignment == initial
