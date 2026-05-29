UNWIND [
  "Jeffrey Epstein",
  "unknown person A",
  "Donald Trump",
  "Reid Weingarten",
  "email",
  "transcript",
  "report",
  "court_filing",
  "media_article"
] AS name
RETURN
  name AS id,
  name AS label,
  CASE
    WHEN name IN ["Jeffrey Epstein", "unknown person A", "Donald Trump", "Reid Weingarten"] THEN "Entity"
    ELSE "Category"
  END AS type
ORDER BY type, label;