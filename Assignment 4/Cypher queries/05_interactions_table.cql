// 05_interactions_table.cypher
// Purpose: compute interaction counts and social/legal split per alter

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
LIMIT 100;