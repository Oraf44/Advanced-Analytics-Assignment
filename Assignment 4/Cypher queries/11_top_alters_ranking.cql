MATCH (:Entity {name: 'Jeffrey Epstein'})-[r]->(alter:Entity)
WITH type(r) AS relType, alter, COUNT(r) AS interactions
RETURN alter.name AS alter, relType, interactions
ORDER BY interactions DESC
LIMIT 10;