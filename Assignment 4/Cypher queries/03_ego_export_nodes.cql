// 03_ego_export_nodes.cypher
// Purpose: export nodes for Epstein ego graph (for Gephi)

WITH "
MATCH (e:Entity)
WHERE e.name = 'Jeffrey Epstein'
   OR EXISTS {
     MATCH (:Entity {name: 'Jeffrey Epstein'})-[:RELATED_TO]->(e)
   }
RETURN e.name AS id,
       e.name AS label,
       e.entity_type AS entity_type
" AS query
CALL export_util.csv_query(
  query,
  '/var/lib/memgraph/epstein_nodes.csv',
  TRUE
)
YIELD file_path
RETURN file_path;