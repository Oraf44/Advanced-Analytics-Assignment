// 15_unknownA_doccategory_nodes.cql
// Export nodes for the alter–category graph
// Nodes include all alters connected to Unknown Person A by >= 10 docs in any category,
// plus the corresponding Category nodes

WITH "
MATCH (u:Entity)
WHERE toLower(u.name) CONTAINS 'unknown person a'

MATCH (d:Document)-[:MENTIONS]->(u)
MATCH (d)-[:MENTIONS]->(other:Entity)
WHERE NOT toLower(other.name) CONTAINS 'unknown person a'
MATCH (d)-[:IN_CATEGORY]->(c:Category)

// same aggregation / threshold as in 14_unknownA_doccategory_edges.cql
WITH other, c, count(DISTINCT d) AS weight
WHERE weight >= 10

WITH collect(DISTINCT other) AS alters,
     collect(DISTINCT c)     AS cats
UNWIND (alters + cats) AS n

RETURN
  CASE
    WHEN n:Entity   THEN n.name
    WHEN n:Category THEN n.name
  END              AS id,
  CASE
    WHEN n:Entity   THEN n.name
    WHEN n:Category THEN n.name
  END              AS label,
  labels(n)[0]     AS node_type
" AS query
CALL export_util.csv_query(
  query,
  '/var/lib/memgraph/unknownA_doccategory_nodes.csv',
  TRUE
)
YIELD file_path
RETURN file_path;