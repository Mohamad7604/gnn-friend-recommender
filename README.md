📌 GNN Friend Recommender — Neo4j + Graph Neural Networks

A demo project that builds a social graph, trains a Graph Neural Network (GCN) for link prediction, and exports friend recommendations to a Neo4j NoSQL graph database.

🚀 Features

Build social graph with node features

Train GNN for link prediction

Generate friend recommendations

Export graph to Neo4j Aura

Query recommendations from graph DB

Optional Flask API + Web UI

🧠 Tech Stack

Python

PyTorch Geometric

NetworkX

Neo4j Aura (Graph NoSQL DB)

Flask (API layer)

📊 What the model does

The GNN learns node embeddings and predicts which pairs of people are likely to become friends based on graph structure and interests.

▶️ How to Run
pip install torch torch-geometric neo4j networkx matplotlib scikit-learn
python main.py

🗄 Neo4j Setup

Create Neo4j Aura free instance

Get URI + username + password

Put them in the code or .env file

📌 Example Output

Top Friend Recommendations:

Karim ↔ Jad

Lina ↔ Sara

Amina ↔ Karim
