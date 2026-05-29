// 06_interactions_export.cypher
// Purpose: export extended interactions table to CSV for analysis

WITH "
WITH
  ['met','attended','traveled','socialized','visited','communicated','dined','stayed'] AS social_actions,
  ['accused','investigated','charged','testified','represented','indicted','sued']    AS legal_actions

MATCH (e:Entity {name: 'Jeffrey Epstein'})-[r:RELATED_TO]->(other:Entity)
WITH other, r, social_actions, legal_actions
WITH other,
     count(*) AS total_interactions,
     sum(CASE WHEN r.action IN social_actions THEN 1 ELSE 0 END) AS social_interactions,
     sum(CASE WHEN r.action IN legal_actions THEN 1 ELSE 0 END) AS legal_interactions
RETURN other.name AS alter,
       total_interactions,
       social_interactions,
       legal_interactions,
       CASE
         WHEN total_interactions = 0 THEN 0.0
         ELSE 1.0 * legal_interactions / total_interactions
       END AS legal_share
ORDER BY total_interactions DESC
LIMIT 500
" AS query
CALL export_util.csv_query(
  query,
  '/var/lib/memgraph/epstein_interactions.csv',
  TRUE
)
YIELD file_path
RETURN file_path;