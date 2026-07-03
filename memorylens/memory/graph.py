"""
GraphMemory — knowledge-graph backend built on NetworkX.

Facts are stored as (user) -[relation]-> (value) edges in a directed graph.
A fact update removes the old edge and inserts a new one, so retrieval always
serialises the current state of the graph — stale values cannot survive.

Extraction reuses the same local regex templates as EntityMemory, keeping the
backend deterministic and free of LLM calls.
"""

from typing import Dict, List

import networkx as nx

from memorylens.memory.base import BaseMemory
from memorylens.memory.entity import _extract_entity

_USER = "user"


class GraphMemory(BaseMemory):
    """
    Directed knowledge graph of user facts.

    Nodes: the user plus one node per fact value.
    Edges: (user, value) annotated with relation=fact key and the turn it was
    asserted, so the graph doubles as a temporal provenance record.

    # ponytail: single-hop user->value triples only; add entity-entity edges
    # when a scenario actually needs multi-hop retrieval.
    """

    name = "graph"

    def __init__(self) -> None:
        self.graph = nx.DiGraph()
        self.graph.add_node(_USER)

    def add_message(self, role: str, content: str, turn: int) -> None:
        if role != "user":
            return
        pair = _extract_entity(content)
        if pair is None:
            return
        key, value = pair

        for _, old_value, data in list(self.graph.out_edges(_USER, data=True)):
            if data.get("relation") == key:
                self.graph.remove_edge(_USER, old_value)
                if self.graph.degree(old_value) == 0:
                    self.graph.remove_node(old_value)

        self.graph.add_edge(_USER, value, relation=key, asserted_at=turn)

    def get_context(self, query: str, current_turn: int) -> List[Dict]:
        edges = self.graph.out_edges(_USER, data=True)
        if not edges:
            return []
        lines = [f"my {data['relation']} is {value}" for _, value, data in edges]
        return [
            {
                "role": "system",
                "content": "[Knowledge graph facts] " + "; ".join(lines) + ".",
            }
        ]

    def reset(self) -> None:
        self.graph = nx.DiGraph()
        self.graph.add_node(_USER)
