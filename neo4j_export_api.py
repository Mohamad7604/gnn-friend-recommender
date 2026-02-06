import requests

def cypher(api_url, user, password, query, params=None):
    payload = {"statement": query, "parameters": params or {}}
    r = requests.post(api_url, auth=(user, password), json=payload)
    r.raise_for_status()
    return r.json()

def export_to_neo4j_api(api_url, user, password, people, interests, undirected_edges, recommendations):
    # Clear DB
    cypher(api_url, user, password, "MATCH (n) DETACH DELETE n")

    # Nodes
    for name in people:
        cypher(
            api_url, user, password,
            "MERGE (p:Person {name:$name}) SET p.interest=$interest",
            {"name": name, "interest": interests[name]}
        )

    # FRIEND relationships (both directions)
    for u, v in undirected_edges:
        cypher(api_url, user, password, """
        MATCH (a:Person {name:$a}), (b:Person {name:$b})
        MERGE (a)-[:FRIEND]->(b)
        MERGE (b)-[:FRIEND]->(a)
        """, {"a": people[u], "b": people[v]})

    # RECOMMENDED_FRIEND relationships
    for (u, v), score in recommendations:
        cypher(api_url, user, password, """
        MATCH (a:Person {name:$a}), (b:Person {name:$b})
        MERGE (a)-[r:RECOMMENDED_FRIEND]->(b)
        SET r.score = $score
        """, {"a": people[u], "b": people[v], "score": float(score)})

    print("✅ Exported to Neo4j via Query API (HTTPS)")
