import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


import networkx as nx
import matplotlib.pyplot as plt

from sklearn.metrics import roc_auc_score

# PyTorch Geometric
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from torch_geometric.utils import negative_sampling
from neo4j_export_api import export_to_neo4j_api
# ----------------------------
# 1) Create a small social graph
# ----------------------------
def build_demo_graph():
    """
    Nodes = people
    Edges = friendships (undirected)
    Node features = interest one-hot (AI, Music, Sports, Gaming)
    """
    people = ["Amina", "Hadi", "Maya", "Karim", "Lina", "Omar", "Sara", "Jad"]
    interest_list = ["AI", "Music", "Sports", "Gaming"]

    # Assign interests (creative + realistic)
    interests = {
        "Amina": "AI",
        "Hadi": "AI",
        "Maya": "Music",
        "Karim": "Sports",
        "Lina": "Music",
        "Omar": "Sports",
        "Sara": "Gaming",
        "Jad": "Gaming",
    }

    # Friendships (edges)
    edges = [
        ("Amina", "Hadi"),
        ("Amina", "Maya"),
        ("Hadi", "Karim"),
        ("Maya", "Lina"),
        ("Karim", "Omar"),
        ("Sara", "Jad"),
        ("Maya", "Sara"),     # bridge between groups
        ("Omar", "Jad"),      # bridge between groups
    ]

    # Map names to ids
    node_id = {name: i for i, name in enumerate(people)}

    # Build edge_index (undirected -> add both directions)
    edge_index = []
    for u, v in edges:
        edge_index.append([node_id[u], node_id[v]])
        edge_index.append([node_id[v], node_id[u]])
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

    # Node features: interest one-hot
    feat = []
    for p in people:
        one_hot = [0] * len(interest_list)
        one_hot[interest_list.index(interests[p])] = 1
        feat.append(one_hot)
    x = torch.tensor(feat, dtype=torch.float)

    return people, interests, edge_index, x


# ----------------------------
# 2) Split edges into train/test (link prediction)
# ----------------------------
def train_test_split_edges(edge_index, test_ratio=0.25, seed=42):
    """
    edge_index contains BOTH directions already.
    We'll build a unique undirected edge list then split.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # Extract unique undirected edges (u < v)
    edges = edge_index.t().tolist()
    undirected = set()
    for u, v in edges:
        if u == v:
            continue
        a, b = (u, v) if u < v else (v, u)
        undirected.add((a, b))
    undirected = list(undirected)
    random.shuffle(undirected)

    n_test = int(len(undirected) * test_ratio)
    test_edges = undirected[:n_test]
    train_edges = undirected[n_test:]

    # Make train edge_index with both directions
    train_edge_index = []
    for u, v in train_edges:
        train_edge_index.append([u, v])
        train_edge_index.append([v, u])
    train_edge_index = torch.tensor(train_edge_index, dtype=torch.long).t().contiguous()

    return train_edges, test_edges, train_edge_index


# ----------------------------
# 3) GNN Model: GCN encoder + dot-product decoder
# ----------------------------
class GCNLinkPredictor(nn.Module):
    def __init__(self, in_dim, hidden_dim=32, out_dim=32):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, out_dim)

    def encode(self, x, edge_index):
        h = self.conv1(x, edge_index)
        h = F.relu(h)
        z = self.conv2(h, edge_index)
        return z

    def decode(self, z, edge_pairs):
        """
        edge_pairs: [2, num_edges] tensor of (u,v)
        Score = dot(z_u, z_v)
        """
        u = edge_pairs[0]
        v = edge_pairs[1]
        return (z[u] * z[v]).sum(dim=1)

    def decode_all(self, z):
        """
        Returns score matrix NxN
        """
        return z @ z.t()


# ----------------------------
# 4) Training loop
# ----------------------------
def make_edge_tensor(edge_list):
    return torch.tensor(edge_list, dtype=torch.long).t().contiguous()

def evaluate_auc(model, data, pos_edges, num_nodes):
    model.eval()
    with torch.no_grad():
        z = model.encode(data.x, data.edge_index)

        pos_edge_tensor = make_edge_tensor(pos_edges)
        pos_scores = model.decode(z, pos_edge_tensor).sigmoid().cpu().numpy()

        # Sample negative edges equal to positive
        neg_edge_index = negative_sampling(
            edge_index=data.edge_index,
            num_nodes=num_nodes,
            num_neg_samples=len(pos_edges),
            method="sparse"
        )
        neg_scores = model.decode(z, neg_edge_index).sigmoid().cpu().numpy()

        y_true = np.hstack([np.ones(len(pos_scores)), np.zeros(len(neg_scores))])
        y_pred = np.hstack([pos_scores, neg_scores])
        return roc_auc_score(y_true, y_pred)

def train(model, data, train_edges, num_nodes, epochs=200, lr=0.01):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()

    train_edge_tensor = make_edge_tensor(train_edges)

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()

        z = model.encode(data.x, data.edge_index)

        # Positive samples
        pos_logits = model.decode(z, train_edge_tensor)
        pos_labels = torch.ones(pos_logits.size(0))

        # Negative samples
        neg_edge_index = negative_sampling(
            edge_index=data.edge_index,
            num_nodes=num_nodes,
            num_neg_samples=train_edge_tensor.size(1),
            method="sparse"
        )
        neg_logits = model.decode(z, neg_edge_index)
        neg_labels = torch.zeros(neg_logits.size(0))

        logits = torch.cat([pos_logits, neg_logits], dim=0)
        labels = torch.cat([pos_labels, neg_labels], dim=0)

        loss = loss_fn(logits, labels)
        loss.backward()
        optimizer.step()

        if epoch % 25 == 0:
            auc = evaluate_auc(model, data, train_edges, num_nodes)
            print(f"Epoch {epoch:03d} | Loss: {loss.item():.4f} | Train AUC: {auc:.3f}")

    return model


# ----------------------------
# 5) Recommend new friends
# ----------------------------
def recommend_top_k(model, data, original_undirected_edges, people, k=5):
    """
    Recommend new edges (u,v) not already in graph, based on highest predicted score.
    """
    model.eval()
    num_nodes = data.num_nodes

    # Existing edges set for quick lookup
    existing = set()
    for u, v in original_undirected_edges:
        a, b = (u, v) if u < v else (v, u)
        existing.add((a, b))

    with torch.no_grad():
        z = model.encode(data.x, data.edge_index)
        score_mat = model.decode_all(z).sigmoid().cpu().numpy()

    candidates = []
    for u in range(num_nodes):
        for v in range(u + 1, num_nodes):
            if (u, v) in existing:
                continue
            candidates.append(((u, v), score_mat[u, v]))

    candidates.sort(key=lambda x: x[1], reverse=True)
    top = candidates[:k]

    print("\n=== Top Friend Recommendations ===")
    for (u, v), s in top:
        print(f"{people[u]} ↔ {people[v]}  (score={s:.3f})")

    return top


# ----------------------------
# 6) Visualize graph
# ----------------------------
def plot_graph(people, interests, undirected_edges, title="Social Graph"):
    G = nx.Graph()
    for p in people:
        G.add_node(p, interest=interests[p])

    for u, v in undirected_edges:
        G.add_edge(people[u], people[v])

    pos = nx.spring_layout(G, seed=7)

    plt.figure(figsize=(9, 6))
    nx.draw(G, pos, with_labels=True, node_size=1500, font_size=10)
    edge_labels = {(a, b): "" for a, b in G.edges()}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
    plt.title(title)
    plt.show()


def main():
    people, interests, edge_index, x = build_demo_graph()
    num_nodes = len(people)

    # Split into train/test edges
    train_edges, test_edges, train_edge_index = train_test_split_edges(edge_index, test_ratio=0.25)

    # Build Data object (use only train edges for message passing)
    data = Data(x=x, edge_index=train_edge_index)
    data.num_nodes = num_nodes

    # Visualize the original graph
    plot_graph(people, interests, train_edges + test_edges, title="Original Social Graph")

    # Train model
    model = GCNLinkPredictor(in_dim=x.size(1), hidden_dim=32, out_dim=32)
    model = train(model, data, train_edges, num_nodes, epochs=200, lr=0.01)

    # Evaluate on test edges (AUC)
    test_auc = evaluate_auc(model, data, test_edges, num_nodes)
    print(f"\nTest AUC: {test_auc:.3f}")

    # Recommend new friendships
    original_undirected = train_edges + test_edges
    top = recommend_top_k(model, data, original_undirected, people, k=6)
    # -------- Export to Neo4j Aura (NoSQL Graph DB) --------
        # -------- Export to Neo4j Aura (NoSQL Graph DB) --------
    api_url = "https://1292f305.databases.neo4j.io/db/neo4j/query/v2"
    user = "neo4j"
    password = "0Pv-iGEDHMk0bT03Ci7wuMTqe1_84QYUuZfQMGg2RNg"

    export_to_neo4j_api(
        api_url,
        user,
        password,
        people,
        interests,
        original_undirected,
        top
    )
    print("🚀 Starting export to Neo4j...")
    print("✅ Exported graph + GNN recommendations to Neo4j NoSQL DB")


    # Optional: visualize graph with recommended edges added
    # Add top 3 recommendations as dashed edges (simple)
    rec_edges = [(u, v) for (u, v), _ in top[:3]]
    plot_graph(people, interests, original_undirected + rec_edges,
               title="Graph + Top 3 Recommended Friendships (added)")


if __name__ == "__main__":
    main()
