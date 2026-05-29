# Assignment 4

This repository contains the materials for Assignment 4 on network analysis using Memgraph and Gephi. The project analyses an Epstein-centered ego-network and then narrows the focus to key alters, especially Unknown Person A, to compare how different actors appear across document categories.

## Folder structure

- `Cypher queries/` – Memgraph Cypher files used for querying and exporting results.
- `Data files/` – exported CSV and related data files.
- `Gephi plots/` – network visualizations created in Gephi.
- `images/` – supporting screenshots and figure images.

## Workflow

1. Build and inspect the Epstein ego-network in Memgraph.
2. Export nodes and edges for Gephi.
3. Identify key alters through interaction counts.
4. Create focused follow-up graphs, including Unknown Person A ego and document-category graphs.
5. Use the figures and tables in the final report.

## Tools used

- Memgraph / Memgraph Lab
- Cypher
- Gephi

## Notes

The analysis treats `Unknown Person A` as an anonymized central alter by merging document-specific variants of the same label for network exploration. The goal is interpretive network analysis, not real-world identification.
