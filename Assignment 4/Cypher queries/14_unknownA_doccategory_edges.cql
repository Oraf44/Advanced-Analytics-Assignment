// 14_unknownA_doccategory_edges.cql
// Export alter–category edges for documents that mention Unknown Person A
// Each edge connects a co-actor (Entity) to a document Category, weighted by doc count

WITH "
MATCH (u:Entity)
WHERE toLower(u.name) CONTAINS 'unknown person a'

// documents that mention any Unknown Person A variant
MATCH (d:Document)-[:MENTIONS]->(u)

// other entities in those documents
MATCH (d)-[:MENTIONS]->(other:Entity)
WHERE NOT toLower(other.name) CONTAINS 'unknown person a'

// document categories
MATCH (d)-[:IN_CATEGORY]->(c:Category)

// aggregate by alter and category
WITH other, c, count(DISTINCT d) AS weight

// threshold to keep the graph readable – adjust if you want denser/sparser
WHERE weight >= 10

RETURN
  other.name AS source,   // alter
  c.name     AS target,   // document category
  weight
ORDER BY weight DESC, source, target
" AS query
CALL export_util.csv_query(
  query,
  '/var/lib/memgraph/unknownA_doccategory_edges.csv',
  TRUE
)
YIELD file_path
RETURN file_path;