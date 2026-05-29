// Count documents by category
MATCH (d:Document)-[:IN_CATEGORY]->(c:Category)
RETURN c.name, count(d) AS n
ORDER BY n DESC;