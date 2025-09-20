import os
import sys
import socket
import random
from datetime import datetime

import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
from py2neo import Graph, GraphService, Node, Relationship
from py2neo.matching import NodeMatcher, RelationshipMatcher

NEO4J_URI = "bolt://127.0.0.1:7687"  
NEO4J_USER = "neo4j"
NEO4J_PASS = "testtest"
NEO4J_DATABASE = "sndb"


N_NODES = 1200  
BA_M = 3    
SEED = 42
BETWEENNESS_K = 300   
TOPK = 50 

random.seed(SEED)
os.makedirs("outputs", exist_ok=True)


def _port_open(host: str, port: int, timeout: float = 1.0) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(timeout)
        try:
            s.connect((host, port))
            return True
        except Exception:
            return False


def connect_graph() -> Graph:
    host = "127.0.0.1"; port = 7687
    print(f"[connect] Checking Bolt {host}:{port} ...")
    if not _port_open(host, port):
        print("[error] Bolt порт не слухає. У Neo4j Desktop база має бути RUNNING,"
              " і Connection URI відповідати bolt://127.0.0.1:7687.")
        sys.exit(1)

    try:
        print(f"[connect] Trying Graph({NEO4J_URI}, name={NEO4J_DATABASE}, routing=False)")
        g = Graph(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASS),
                  name=NEO4J_DATABASE, routing=False, secure=False)
        NodeMatcher(g).match().first()
        print("[connect] OK via Graph")
        return g
    except Exception as e:
        print(f"[connect] Graph failed: {type(e).__name__}: {e}")

    try:
        print(f"[connect] Trying GraphService({NEO4J_URI}) and selecting database '{NEO4J_DATABASE}'")
        gs = GraphService(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASS),
                          routing=False, secure=False)
        g = gs[NEO4J_DATABASE]
        NodeMatcher(g).match().first()
        print("[connect] OK via GraphService")
        return g
    except Exception as e:
        print(f"[connect] GraphService failed: {type(e).__name__}: {e}")

    print("\n[error] Не вдалося підключитися. Перевір:")
    print("  • У Desktop база RUNNING;")
    print("  • Bolt URI та порт (ми використовуємо bolt://127.0.0.1:7687);")
    print("  • Логін/пароль neo4j/testtest1!;")
    print("  • Версію Neo4j (py2neo стабільніший із 4.4.x).")
    sys.exit(1)

def seed_if_empty(graph: Graph):
    nm = NodeMatcher(graph)
    user_count = nm.match("User").count()
    if user_count > 0:
        print(f"[seed] Found existing :User nodes: {user_count}. Seeding skipped.")
        return

    print(f"[seed] No :User nodes found. Seeding synthetic social graph ({N_NODES} nodes)...")

    G = nx.barabasi_albert_graph(n=N_NODES, m=BA_M, seed=SEED)

    tx = graph.begin()

    nodes_py2neo = {}
    for n in G.nodes():
        node = Node(
            "User",
            neo_id=int(n),
            name=f"User_{n}",
            age=18 + (n % 50),
            interests=["tech", "music", "travel", "sports"][n % 4]
        )
        tx.create(node)
        nodes_py2neo[n] = node

    for u, v in G.edges():
        rel = Relationship(
            nodes_py2neo[u], "FRIENDS", nodes_py2neo[v],
            since=2023 - ((u + v) % 3)
        )
        tx.create(rel)

    graph.commit(tx)
    print("[seed] Seeding completed.")

def load_graph_from_neo4j(graph: Graph) -> nx.Graph:
    nm = NodeMatcher(graph)
    rm = RelationshipMatcher(graph)

    print("[load] Reading nodes...")
    users = nm.match("User").all()
    id_map = {} 
    attrs = {}

    for node in users:
        nid = node.get("neo_id", int(node.identity))
        id_map[node.identity] = nid
        attrs[nid] = {
            "name": node.get("name"),
            "age": node.get("age"),
            "interests": node.get("interests")
        }

    print(f"[load] Users loaded: {len(attrs)}")

    print("[load] Reading FRIENDS relationships...")
    rels = rm.match(nodes=None, r_type="FRIENDS").all()

    edges = []
    for r in rels:
        u_internal = r.start_node.identity
        v_internal = r.end_node.identity
        u = id_map.get(u_internal)
        v = id_map.get(v_internal)
        if u is None or v is None:
            continue
        edges.append((u, v))

    print(f"[load] Relationships loaded: {len(edges)}")

    G = nx.Graph()
    G.add_nodes_from(attrs.keys())
    nx.set_node_attributes(G, attrs)
    G.add_edges_from(edges)
    return G

def analyze_graph(G: nx.Graph):
    print("[analysis] Computing communities (greedy modularity)...")
    communities = list(nx.algorithms.community.greedy_modularity_communities(G))
    node_to_comm = {}
    for cid, comm in enumerate(communities):
        for n in comm:
            node_to_comm[n] = cid

    print("[analysis] Computing degree centrality...")
    deg_cent = nx.degree_centrality(G)

    print("[analysis] Computing betweenness centrality (approximate)...")
    bet_cent = nx.betweenness_centrality(G, k=BETWEENNESS_K, seed=SEED)

    print("[analysis] Computing eigenvector centrality (on GCC)...")
    gcc_nodes = max(nx.connected_components(G), key=len)
    G_gcc = G.subgraph(gcc_nodes).copy()
    eig_cent_gcc = nx.eigenvector_centrality_numpy(G_gcc)
    eig_cent = {n: eig_cent_gcc.get(n, 0.0) for n in G.nodes()}

    df = pd.DataFrame({
        "node": list(G.nodes()),
        "name": [G.nodes[n].get("name") for n in G.nodes()],
        "community_id": [node_to_comm.get(n, -1) for n in G.nodes()],
        "degree": [G.degree(n) for n in G.nodes()],
        "degree_centrality": [deg_cent[n] for n in G.nodes()],
        "betweenness": [bet_cent[n] for n in G.nodes()],
        "eigenvector": [eig_cent[n] for n in G.nodes()],
        "age": [G.nodes[n].get("age") for n in G.nodes()],
        "interests": [G.nodes[n].get("interests") for n in G.nodes()],
    })

    return df, communities

def save_tables_and_plots(G: nx.Graph, df: pd.DataFrame, communities):
    print("[save] Writing CSVs & plots...")

    df.to_csv("outputs/graph_summary.csv", index=False)

    df_ranked = df.copy()
    for col in ["degree_centrality", "betweenness", "eigenvector"]:
        df_ranked[f"rank_{col}"] = df_ranked[col].rank(ascending=False, method="min")
    df_ranked["rank_sum"] = df_ranked[["rank_degree_centrality", "rank_betweenness", "rank_eigenvector"]].sum(axis=1)
    topk = df_ranked.sort_values("rank_sum").head(TOPK)
    topk.to_csv("outputs/centrality_top50.csv", index=False)

    comm_sizes = pd.Series({i: len(c) for i, c in enumerate(communities)})
    df_comm_sizes = pd.DataFrame({"community_id": comm_sizes.index, "size": comm_sizes.values})
    df_comm_sizes.to_csv("outputs/community_sizes.csv", index=False)

    plt.figure()
    degrees = [d for _, d in G.degree()]
    plt.hist(degrees, bins=50)
    plt.title("Degree Distribution")
    plt.xlabel("Degree")
    plt.ylabel("Number of nodes")
    plt.tight_layout()
    plt.savefig("outputs/fig_degree_distribution.png")
    plt.close()

    plt.figure()
    plt.hist(df["betweenness"].values, bins=50)
    plt.title("Betweenness Centrality Distribution")
    plt.xlabel("Betweenness")
    plt.ylabel("Number of nodes")
    plt.tight_layout()
    plt.savefig("outputs/fig_betweenness_distribution.png")
    plt.close()

    plt.figure()
    plt.hist(df["eigenvector"].values, bins=50)
    plt.title("Eigenvector Centrality Distribution")
    plt.xlabel("Eigenvector centrality")
    plt.ylabel("Number of nodes")
    plt.tight_layout()
    plt.savefig("outputs/fig_eigenvector_distribution.png")
    plt.close()

    plt.figure()
    plt.bar(df_comm_sizes["community_id"].astype(int).astype(str), df_comm_sizes["size"].astype(int))
    plt.title("Community Size Distribution")
    plt.xlabel("Community ID")
    plt.ylabel("Size")
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig("outputs/fig_community_sizes.png")
    plt.close()

    largest_comm_id = int(df_comm_sizes.sort_values("size", ascending=False).iloc[0]["community_id"])
    largest_comm_nodes = list(communities[largest_comm_id])
    max_plot_nodes = 150
    if len(largest_comm_nodes) > max_plot_nodes:
        random.seed(SEED)
        largest_comm_nodes = random.sample(largest_comm_nodes, max_plot_nodes)

    G_sub = G.subgraph(largest_comm_nodes).copy()
    pos = nx.spring_layout(G_sub, seed=SEED)
    plt.figure(figsize=(8, 8))
    nx.draw_networkx_nodes(G_sub, pos, node_size=20)
    nx.draw_networkx_edges(G_sub, pos, width=0.5)
    plt.title(f"Largest Community Subgraph (sampled up to {max_plot_nodes} nodes)")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig("outputs/fig_top_community_subgraph.png")
    plt.close()

    return topk, df_comm_sizes

def main():
    print("[connect] Connecting to Neo4j...")
    graph = connect_graph()

    seed_if_empty(graph)

    print("[load] Building NetworkX graph from Neo4j...")
    G = load_graph_from_neo4j(graph)

    print("[analyze] Running analysis...")
    df, communities = analyze_graph(G)

    topk, df_comm_sizes = save_tables_and_plots(G, df, communities)

    avg_degree = sum(dict(G.degree()).values()) / float(G.number_of_nodes())
    print("\n=== SUMMARY ===")
    print(f"Nodes: {G.number_of_nodes()} | Edges: {G.number_of_edges()}")
    print(f"Communities: {len(communities)}")
    print(f"Largest community size: {int(df_comm_sizes['size'].max())}")
    print(f"Smallest community size: {int(df_comm_sizes['size'].min())}")
    print(f"Average node degree: {avg_degree:.2f}")
    print("Top-10 by composite rank:")
    print(topk[["name", "community_id", "degree"]].head(10).to_string(index=False))
    print("\nSaved to ./outputs/: graph_summary.csv, centrality_top50.csv, community_sizes.csv")
    print("Plots: fig_degree_distribution.png, fig_betweenness_distribution.png, fig_eigenvector_distribution.png, fig_community_sizes.png, fig_top_community_subgraph.png")

if __name__ == "__main__":
    main()