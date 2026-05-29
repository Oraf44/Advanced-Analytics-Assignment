// 12_unknownA_ego_edges.cql
// Export a focused ego-network around "Unknown person A"
// Edges: Unknown person A (merged variants) -> alters, weighted by interaction count and doc count

WITH "
MATCH (u:Entity)
WHERE toLower(u.name) CONTAINS 'unknown person a'

// ego edges from any Unknown Person A variant to its neighbors
MATCH (u)-[r:RELATED_TO]-(other:Entity)
WHERE NOT toLower(other.name) CONTAINS 'unknown person a'

// aggregate multiple edges and documents
WITH u, other,
     count(r) AS weight,
     count(DISTINCT r.doc_id) AS docs

// threshold to avoid a hairball – adjust if needed
WHERE weight >= 10

RETURN
  u.name     AS source,
  other.name AS target,
  weight,
  docs
ORDER BY weight DESC
" AS query
CALL export_util.csv_query(
  query,
  '/var/lib/memgraph/unknownA_ego_edges.csv',
  TRUE
)
YIELD file_path
RETURN file_path;