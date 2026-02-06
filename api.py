from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
import requests
import os
from requests.auth import HTTPBasicAuth

app = Flask(__name__, static_folder=".", static_url_path="")
CORS(app)

# ✅ Use DB username/password
NEO4J_QUERY_URL = "https://1292f305.databases.neo4j.io/db/neo4j/query/v2"
NEO4J_USER = "neo4j"
NEO4J_PASS = "0Pv-iGEDHMk0bT03Ci7wuMTqe1_84QYUuZfQMGg2RNg"

def run_cypher(cypher, params=None):
    payload = {
        "statement": cypher,
        "parameters": params or {}
    }

    try:
        r = requests.post(
            NEO4J_QUERY_URL,
            json=payload,
            auth=HTTPBasicAuth(NEO4J_USER, NEO4J_PASS),
            headers={"Content-Type": "application/json"},
            timeout=30
        )
        r.raise_for_status()
        resp = r.json()
        
        data = resp.get("data")

        # ✅ Handle different API response formats
        if isinstance(data, dict) and "values" in data:
            return data["values"]
        if isinstance(data, list):
            return data
            
        return []
    except Exception as e:
        print(f"Database Error: {e}")
        return []

@app.route("/")
def home():
    return send_from_directory(".", "index.html")

@app.route("/people")
def people():
    data = run_cypher("MATCH (p:Person) RETURN p.name AS name ORDER BY name")
    # Handle case where data might be empty or formatted differently
    if not data:
        return jsonify([])
    names = [row[0] for row in data]
    return jsonify(names)

@app.route("/recommendations")
def recommendations():
    person = request.args.get("name")
    if not person:
        return jsonify({"error": "Missing ?name="}), 400

    # 1. GNN Attempt (AI)
    # Note: We now COLLECT the names of mutual friends
    gnn_query = """
        MATCH (:Person {name:$name})-[r:RECOMMENDED_FRIEND]->(p:Person)
        MATCH (:Person {name:$name})-[:FRIEND]-(mutual)-[:FRIEND]-(p)
        RETURN p.name, r.score, COLLECT(mutual.name) as reasons
        ORDER BY r.score DESC
    """
    rows = run_cypher(gnn_query, {"name": person})

    # 2. Topology Fallback (Real-Time Graph Traversal)
    if not rows:
        print(f"Switching to Topology for {person}...")
        fallback_query = """
            MATCH (me:Person {name:$name})-[:FRIEND]-(friend)-[:FRIEND]-(fof:Person)
            WHERE NOT (me)-[:FRIEND]-(fof) AND me <> fof
            RETURN fof.name, 
                   0.5 + (COUNT(friend) * 0.1) as score, 
                   COLLECT(friend.name) as reasons
            ORDER BY score DESC
            LIMIT 5
        """
        rows = run_cypher(fallback_query, {"name": person})

    # Format Output
    out = []
    if rows:
        for row in rows:
            # 'reasons' is now a list like ['Karim', 'Sara']
            out.append({
                "name": row[0], 
                "score": float(row[1]) if row[1] is not None else 0.0,
                "reasons": row[2] if len(row) > 2 else [] 
            })
            
    return jsonify(out)
@app.route("/graph")
def graph():
    nodes = run_cypher("MATCH (p:Person) RETURN p.name, p.interest ORDER BY p.name")
    rels = run_cypher("""
        MATCH (a:Person)-[r]->(b:Person)
        RETURN a.name, type(r), b.name, coalesce(r.score, 0)
    """)

    return jsonify({
        "nodes": [{"name": n[0], "interest": n[1]} for n in nodes] if nodes else [],
        "rels": [{"from": r[0], "type": r[1], "to": r[2], "score": float(r[3])} for r in rels] if rels else [],
    })

# --- NEW ROUTES ADDED BELOW FOR YOUR INTERFACE ---

@app.route('/add_node', methods=['POST'])
def add_node():
    data = request.json
    name = data.get('name')
    
    if not name:
        return jsonify({"error": "No name provided"}), 400

    # Using your existing run_cypher function
    run_cypher("MERGE (p:Person {name: $name}) RETURN p", {"name": name})
    
    return jsonify({"status": "success", "message": f"Added {name}"}), 201

@app.route('/add_link', methods=['POST'])
def add_link():
    data = request.json
    p1 = data.get('person1')
    p2 = data.get('person2')
    
    if not p1 or not p2:
         return jsonify({"error": "Missing names"}), 400

    # Using your existing run_cypher function
    run_cypher("""
        MATCH (a:Person {name: $p1}), (b:Person {name: $p2})
        MERGE (a)-[:FRIEND]-(b)
        RETURN a, b
    """, {"p1": p1, "p2": p2})
    
    return jsonify({"status": "success"}), 201

if __name__ == "__main__":
    app.run(port=5000, debug=True)