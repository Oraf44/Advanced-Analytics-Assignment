// Basic Epstein ego-network sample
MATCH (e:Entity {name: "Jeffrey Epstein"})-[r:RELATED_TO]->(other:Entity)
RETURN other.name, r.action, r.doc_id
LIMIT 50;