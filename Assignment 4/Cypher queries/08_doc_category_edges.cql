MATCH (ep:Entity {name: "Jeffrey Epstein"})
MATCH (other:Entity)
WHERE other.name IN ["unknown person A", "Donald Trump", "Reid Weingarten"]

MATCH (d:Document)-[:IN_CATEGORY]->(cat:Category)
MATCH (d)-[:HAS_CLAIM]->(c:Claim)
MATCH (c)-[:ACTOR|TARGET]->(ep)
MATCH (c)-[:ACTOR|TARGET]->(other)
WHERE ep <> other
  AND cat.name IN ["email", "transcript", "report", "court_filing", "media_article"]

WITH other.name AS source,
     cat.name AS target,
     count(DISTINCT d) AS weight
WHERE weight > 0

RETURN source, target, weight
ORDER BY source, weight DESC;