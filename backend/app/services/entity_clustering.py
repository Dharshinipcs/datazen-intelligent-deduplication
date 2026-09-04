from __future__ import annotations

from collections import defaultdict
from typing import Iterable


MatchEdge = tuple[int, int]


class UnionFind:
    """Disjoint-set data structure for connected-component clustering."""

    def __init__(self, record_ids: Iterable[int]) -> None:
        self.parent = {record_id: record_id for record_id in record_ids}
        self.rank = {record_id: 0 for record_id in record_ids}

    def find(self, record_id: int) -> int:
        parent = self.parent[record_id]

        if parent != record_id:
            self.parent[record_id] = self.find(parent)

        return self.parent[record_id]

    def union(self, record_a_id: int, record_b_id: int) -> None:
        root_a = self.find(record_a_id)
        root_b = self.find(record_b_id)

        if root_a == root_b:
            return

        rank_a = self.rank[root_a]
        rank_b = self.rank[root_b]

        if rank_a < rank_b:
            self.parent[root_a] = root_b
        elif rank_a > rank_b:
            self.parent[root_b] = root_a
        else:
            self.parent[root_b] = root_a
            self.rank[root_a] += 1


def build_entity_clusters(
    record_ids: Iterable[int],
    match_edges: Iterable[MatchEdge],
) -> tuple[list[list[int]], list[int]]:
    """
    Build duplicate entity clusters from confirmed match edges.

    Records connected transitively through match edges belong to
    the same entity cluster.

    Components containing only one record are considered unclustered.
    """

    normalized_record_ids = sorted(set(record_ids))

    union_find = UnionFind(normalized_record_ids)

    for record_a_id, record_b_id in match_edges:
        if record_a_id == record_b_id:
            continue

        if (
            record_a_id not in union_find.parent
            or record_b_id not in union_find.parent
        ):
            continue

        union_find.union(record_a_id, record_b_id)

    components: dict[int, list[int]] = defaultdict(list)

    for record_id in normalized_record_ids:
        root = union_find.find(record_id)
        components[root].append(record_id)

    clusters: list[list[int]] = []
    unclustered: list[int] = []

    for component in components.values():
        component.sort()

        if len(component) >= 2:
            clusters.append(component)
        else:
            unclustered.extend(component)

    # Deterministic ordering.
    clusters.sort(key=lambda cluster: cluster[0])
    unclustered.sort()

    return clusters, unclustered