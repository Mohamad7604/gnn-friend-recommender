from neo4j import GraphDatabase

def export_to_neo4j(uri, user, password, database, people, interests, undirected_edges, recommendations):
    driver = GraphDatabase.driver(uri, auth=(user, password))

    with driver.session(database=database) as session:
        session.run("MATCH (n) DETACH DELETE n")

        # Nodes
        for name in people:
            session.run(
                "MERGE (p:Person {name:$name}) SET p.interest=$interest",
                name=name,
                interest=interests[name]
            )

        # FRIEND edges (both directions)
        for u, v in undirected_edges:
            session.run("""
            MATCH (a:Person {name:$a}), (b:Person {name:$b})
            MERGE (a)-[:FRIEND]->(b)
            MERGE (b)-[:FRIEND]->(a)
            """, a=people[u], b=people[v])

        # RECOMMENDED edges (store score)
        for (u, v), score in recommendations:
            session.run("""
            MATCH (a:Person {name:$a}), (b:Person {name:$b})
            MERGE (a)-[r:RECOMMENDED_FRIEND]->(b)
            SET r.score = $score
            """, a=people[u], b=people[v], score=float(score))

    driver.close()
