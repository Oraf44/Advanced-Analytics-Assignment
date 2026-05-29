// 13_unknownA_ego_nodes.cql
// Export node table for the Unknown Person A ego-network
// Nodes: all Unknown Person A variants + their alters

WITH "
MATCH (u:Entity)
WHERE toLower(u.name) CONTAINS 'unknown person a'

MATCH (u)-[r:RELATED_TO]-(other:Entity)
WHERE NOT toLower(other.name) CONTAINS 'unknown person a'

// collect all ego nodes (Unknown Person A variants + neighbors)
WITH collect(DISTINCT u) + collect(DISTINCT other) AS nodes
UNWIND nodes AS n

RETURN
  n.name        AS id,
  n.name        AS label,
  labels(n)[0]  AS node_type
" AS query
CALL export_util.csv_query(
  query,
  '/var/lib/memgraph/unknownA_ego_nodes.csv',
  TRUE
)
YIELD file_path
RETURN file_path;