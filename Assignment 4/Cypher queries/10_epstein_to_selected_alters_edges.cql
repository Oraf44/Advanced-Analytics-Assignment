UNWIND [
  "unknown person A",
  "Donald Trump",
  "Reid Weingarten"
] AS alter
RETURN
  "Jeffrey Epstein" AS source,
  alter AS target,
  1 AS weight,
  "ego_link" AS edge_type
ORDER BY target;