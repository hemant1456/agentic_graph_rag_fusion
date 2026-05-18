import json
from pathlib import Path

import networkx as nx

from step_04_knowledge_graph.implementation.builder import build_graph


def save_graph(g: nx.DiGraph, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = nx.node_link_data(g, edges="links")
    path.write_text(json.dumps(data, indent=2))


def load_graph(path: Path) -> nx.DiGraph:
    data = json.loads(path.read_text())
    return nx.node_link_graph(data, directed=True, multigraph=False, edges="links")


def load_or_build(corpus_path: Path, graph_path: Path) -> nx.DiGraph:
    if graph_path.exists():
        g = load_graph(graph_path)
        print(f"Loaded graph: {g.number_of_nodes()} nodes, {g.number_of_edges()} edges")
        return g
    print("Building knowledge graph from corpus CSVs…")
    g = build_graph(corpus_path)
    save_graph(g, graph_path)
    print(f"Built and saved: {g.number_of_nodes()} nodes, {g.number_of_edges()} edges → {graph_path}")
    return g
