// 04_ego_export_edges.cypher
// Purpose: export edges for Epstein ego graph (for Gephi)

WITH "
MATCH (a:Entity {name: 'Jeffrey Epstein'})-[r:RELATED_TO]->(b:Entity)
RETURN a.name AS source,
       b.name AS target,
       r.action AS action,
       r.doc_id AS doc_id
" AS query
CALL export_util.csv_query(
  query,
  '/var/lib/memgraph/epstein_edges.csv',
  TRUE
)
YIELD file_path
RETURN file_path;