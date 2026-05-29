MATCH (ep:Entity {name: "Jeffrey Epstein"})
MATCH (other:Entity)
WHERE other.name IN ["Unknown Person A", "Donald Trump", "Reid Weingarten"]

MATCH (d:Document)-[:IN_CATEGORY]->(cat:Category)
MATCH (d)-[:HAS_CLAIM]->(c:Claim)
MATCH (c)-[:ACTOR|TARGET]->(ep)
MATCH (c)-[:ACTOR|TARGET]->(other)
WHERE ep <> other
  AND cat.name IN ["email", "transcript", "report", "court_filing", "media_article"]

RETURN other.name AS alter,
       cat.name AS category,
       count(DISTINCT d) AS doc_count
ORDER BY alter, doc_count DESC;