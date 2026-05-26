from __future__ import annotations

import os
import platform
import sys

# platform.uname() on Windows calls a WMI query that can hang indefinitely on
# some systems. ollama.Client() builds a user-agent string using platform.machine()
# at import time, so we pre-populate the cache with env-var values (no WMI needed)
# before importing ollama.
if sys.platform == "win32" and not getattr(platform, "_uname_cache", None):
    platform._uname_cache = platform.uname_result(
        "Windows",
        os.environ.get("COMPUTERNAME", "localhost"),
        "",
        "",
        os.environ.get("PROCESSOR_ARCHITEW6432") or os.environ.get("PROCESSOR_ARCHITECTURE", "AMD64"),
    )

import json
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import chromadb
import ollama
from sentence_transformers import SentenceTransformer

from steam_sqlite import load_games_from_sqlite

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = Path(os.environ.get("RAGLOOKER_DB_PATH", BASE_DIR / "steam_games_reviews_25.sqlite"))
CHROMA_PATH = BASE_DIR / "chroma_db"
COLLECTION_NAME = "steam_games"
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
OLLAMA_HOST = "http://localhost:11434"
OLLAMA_MODEL = "gemma3:1b"
MAX_GAMES = 5000
RETRIEVAL_COUNT = 20   # candidates fetched from ChromaDB
DEFAULT_MATCH_COUNT = 5  # final results returned to the user
INDEX_BATCH_SIZE = 500


def create_search_engine() -> "GameSearchEngine":
    return GameSearchEngine(DB_PATH)


# ---------------------------------------------------------------------------
# Data model — keep GameRecord and to_result() exactly as provided
# ---------------------------------------------------------------------------

@dataclass
class GameRecord:
    app_id: str
    raw: dict[str, Any]

    @property
    def name(self) -> str:
        return self.raw.get("name", "Unknown title")

    @property
    def short_description(self) -> str:
        return self.raw.get("short_description", "")

    def to_result(self, score: float) -> dict[str, Any]:
        return {
            "app_id": self.app_id,
            "name": self.name,
            "score": round(score, 4),
            "short_description": self.short_description,
            "genres": self.raw.get("genres", []),
            "tags": self._normalize_tags(self.raw.get("tags")),
            "price": self.raw.get("price"),
            "release_date": self.raw.get("release_date"),
            "header_image": self.raw.get("header_image"),
            "store_page": f"https://store.steampowered.com/app/{self.app_id}",
            "platforms": {
                "windows": bool(self.raw.get("windows")),
                "mac": bool(self.raw.get("mac")),
                "linux": bool(self.raw.get("linux")),
            },
        }

    @staticmethod
    def _normalize_tags(tags: Any) -> list[str]:
        if isinstance(tags, dict):
            return list(tags.keys())[:8]
        if isinstance(tags, list):
            return tags[:8]
        return []


# ---------------------------------------------------------------------------
# Helper: build a text document for a game to use as the embedding input.
# Combines name, short description, genres, and tags into one string so the
# vector captures multiple facets of the game.
# ---------------------------------------------------------------------------

def _build_document(record: GameRecord) -> str:
    genres = ", ".join(record.raw.get("genres", []))
    tags = ", ".join(GameRecord._normalize_tags(record.raw.get("tags")))
    parts: list[str] = [record.name]
    if record.short_description:
        parts.append(record.short_description)
    if genres:
        parts.append(f"Genres: {genres}")
    if tags:
        parts.append(f"Tags: {tags}")
    return ". ".join(parts)


# ---------------------------------------------------------------------------
# Search engine
# ---------------------------------------------------------------------------

class GameSearchEngine:
    """
    LLM-driven recommender system for Steam games.

    Pipeline:
      1. retrieve_candidates  — embed query → ChromaDB cosine search → top-20
      2. rank_candidates      — LLM (gemma3:4b) reranks → top-5 app_ids
      3. generate_answer      — LLM writes a RAG paragraph using game info + reviews
    """

    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self.records = self.load_records()

        # Dict for O(1) app_id lookups during retrieval
        self.records_by_id: dict[str, GameRecord] = {r.app_id: r for r in self.records}

        # Populated by retrieve_candidates so rank_candidates can reuse them
        self._last_scores: dict[str, float] = {}

        # Step 1 — load sentence-transformer embedding model (cached by HuggingFace locally)
        print(f"[init] Loading embedding model '{EMBED_MODEL_NAME}'…")
        self.embed_model = SentenceTransformer(EMBED_MODEL_NAME)

        # Step 2 — connect to (or create) the persisted ChromaDB collection.
        # metadata hnsw:space=cosine means distances returned are cosine distances
        # (distance = 1 - similarity), so similarity = 1 - distance.
        self.chroma_client = chromadb.PersistentClient(path=str(CHROMA_PATH))
        self.collection = self.chroma_client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )

        # Build the index only on the first run; subsequent startups skip this
        if self.collection.count() == 0:
            print(f"[init] Building ChromaDB index for {len(self.records)} games…")
            self._build_index()
            print("[init] Index build complete.")
        else:
            print(f"[init] Loaded existing index ({self.collection.count()} documents).")

        # Ollama client — shared across ranking and answer generation calls
        self.ollama_client = ollama.Client(host=OLLAMA_HOST)

    # ------------------------------------------------------------------
    # Index construction (runs once, then persisted to ./chroma_db/)
    # ------------------------------------------------------------------

    def _build_index(self) -> None:
        """Encode all game documents and upsert into ChromaDB in batches."""
        documents = [_build_document(r) for r in self.records]
        app_ids = [r.app_id for r in self.records]
        metadatas = [{"name": r.name} for r in self.records]

        # Encode all at once; sentence-transformers handles batching internally
        embeddings: list[list[float]] = self.embed_model.encode(
            documents, batch_size=64, show_progress_bar=True
        ).tolist()

        for start in range(0, len(app_ids), INDEX_BATCH_SIZE):
            end = start + INDEX_BATCH_SIZE
            self.collection.upsert(
                ids=app_ids[start:end],
                embeddings=embeddings[start:end],
                documents=documents[start:end],
                metadatas=metadatas[start:end],
            )

    # ------------------------------------------------------------------
    # Record loading
    # ------------------------------------------------------------------

    def load_records(self) -> list[GameRecord]:
        records: list[GameRecord] = []
        for app_id, raw in load_games_from_sqlite(self.db_path, MAX_GAMES):
            records.append(GameRecord(app_id=app_id, raw=raw))
        return records

    # ------------------------------------------------------------------
    # Public search entry point (shape must remain stable for Flask app)
    # ------------------------------------------------------------------

    def search(self, query: str) -> dict[str, Any]:
        candidates = self.retrieve_candidates(query)
        ranked_matches = self.rank_candidates(query, candidates)
        results = [record.to_result(score) for record, score in ranked_matches]

        return {
            "matches": results,
            "answer": self.generate_answer(query, ranked_matches),
            "meta": {
                "indexed_games": len(self.records),
                "retrieval_mode": "embeddings+llm",
                "note": (
                    "Retrieval: sentence-transformers (all-MiniLM-L6-v2) + ChromaDB cosine search. "
                    "Reranking & answer: gemma3:4b via Ollama."
                ),
            },
        }

    # ------------------------------------------------------------------
    # Step 1 — Embedding-based retrieval
    # ------------------------------------------------------------------

    def retrieve_candidates(self, query: str) -> list[GameRecord]:
        """
        Embed the query with the same model used at index time, then query
        ChromaDB for the top-20 most similar game vectors.

        Side-effect: populates self._last_scores so rank_candidates can
        attach cosine similarity scores to the final ranked list.
        """
        # Encode the query — encode() returns (1, dim) array; take first row
        query_vec: list[float] = self.embed_model.encode([query])[0].tolist()

        n = min(RETRIEVAL_COUNT, self.collection.count())
        results = self.collection.query(query_embeddings=[query_vec], n_results=n)

        app_ids: list[str] = results["ids"][0]
        # Cosine distance ∈ [0, 2]; cosine similarity = 1 - distance (clamped to [0, 1])
        distances: list[float] = results["distances"][0]

        self._last_scores = {}
        candidates: list[GameRecord] = []
        for app_id, dist in zip(app_ids, distances):
            similarity = max(0.0, 1.0 - dist)
            self._last_scores[app_id] = similarity
            if app_id in self.records_by_id:
                candidates.append(self.records_by_id[app_id])

        return candidates

    # ------------------------------------------------------------------
    # Step 2 — LLM reranking
    # ------------------------------------------------------------------

    def rank_candidates(
        self, query: str, candidates: list[GameRecord]
    ) -> list[tuple[GameRecord, float]]:
        """
        Ask gemma3:4b to select the 5 most relevant app_ids from the 20
        candidates, then return those records paired with their cosine
        similarity scores from the retrieval step.

        Falls back to top-5 by embedding score if Ollama is unavailable
        or the LLM response cannot be parsed.
        """
        if not candidates:
            return []

        # Build a compact list the LLM can reason over
        lines: list[str] = []
        for rec in candidates:
            tags = ", ".join(GameRecord._normalize_tags(rec.raw.get("tags")))
            desc = rec.short_description[:150].replace("\n", " ")
            lines.append(f'app_id={rec.app_id} | {rec.name} | {desc} | tags: {tags}')
        game_list = "\n".join(lines)

        prompt = (
            f'You are a Steam game recommendation assistant.\n'
            f'User query: "{query}"\n\n'
            f'Candidate games (app_id | name | description | tags):\n'
            f'{game_list}\n\n'
            f'Return ONLY a JSON array (no explanation, no markdown) containing the app_ids '
            f'of the 5 most relevant games, ordered from best to least relevant.\n'
            f'Example format: ["123456", "789012", "345678", "901234", "567890"]'
        )

        try:
            response = self.ollama_client.chat(
                model=OLLAMA_MODEL,
                messages=[{"role": "user", "content": prompt}],
            )
            raw_text: str = response.message.content

            # Extract the JSON array — handles cases where the LLM wraps it in prose
            match = re.search(r'\[.*?\]', raw_text, re.DOTALL)
            if not match:
                raise ValueError(f"No JSON array in LLM response: {raw_text[:200]}")
            top_ids: list[str] = [str(x) for x in json.loads(match.group())]

            # Build the ranked list in LLM order, attach embedding scores
            seen: set[str] = set()
            ranked: list[tuple[GameRecord, float]] = []
            for app_id in top_ids:
                if app_id in self.records_by_id and app_id not in seen:
                    seen.add(app_id)
                    ranked.append((self.records_by_id[app_id], self._last_scores.get(app_id, 0.0)))

            # Pad to DEFAULT_MATCH_COUNT if the LLM returned fewer than 5 valid ids
            if len(ranked) < DEFAULT_MATCH_COUNT:
                for rec in sorted(candidates, key=lambda r: self._last_scores.get(r.app_id, 0.0), reverse=True):
                    if rec.app_id not in seen:
                        ranked.append((rec, self._last_scores.get(rec.app_id, 0.0)))
                        seen.add(rec.app_id)
                        if len(ranked) >= DEFAULT_MATCH_COUNT:
                            break

            return ranked[:DEFAULT_MATCH_COUNT]

        except Exception as exc:
            # Graceful fallback: return top-5 by cosine similarity without LLM
            print(f"[rank_candidates] LLM unavailable or parse error — using embedding scores. ({exc})")
            sorted_cands = sorted(
                candidates,
                key=lambda r: self._last_scores.get(r.app_id, 0.0),
                reverse=True,
            )
            return [(r, self._last_scores.get(r.app_id, 0.0)) for r in sorted_cands[:DEFAULT_MATCH_COUNT]]

    # ------------------------------------------------------------------
    # Step 3 — RAG answer generation
    # ------------------------------------------------------------------

    def _fetch_reviews(self, app_id: str, n: int = 3) -> list[str]:
        """Return up to n English player reviews for a game, ordered by helpfulness."""
        sql = """
            SELECT review FROM reviews
            WHERE appid = ?
              AND language = 'english'
              AND review IS NOT NULL
              AND TRIM(review) != ''
            ORDER BY votes_up DESC
            LIMIT ?
        """
        with sqlite3.connect(self.db_path) as con:
            rows = con.execute(sql, (int(app_id), n)).fetchall()
        return [row[0].strip() for row in rows if row[0]]

    def generate_answer(self, query: str, matches: list[tuple[GameRecord, float]]) -> str:
        """
        Build a RAG prompt with the top-5 games (description + player reviews)
        and ask gemma3:4b to write a friendly 3-4 sentence recommendation.

        Falls back to a plain-text summary if Ollama is unavailable.
        """
        if not matches:
            return "No games were found matching your query."

        # Assemble context: game metadata + top-3 player reviews per game
        context_blocks: list[str] = []
        for rec, _score in matches:
            reviews = self._fetch_reviews(rec.app_id, n=3)
            review_text = (
                " | ".join(f'"{r[:200]}"' for r in reviews)
                if reviews
                else "No reviews available."
            )
            genres = ", ".join(rec.raw.get("genres", []))
            tags = ", ".join(GameRecord._normalize_tags(rec.raw.get("tags")))
            context_blocks.append(
                f"Game: {rec.name}\n"
                f"Description: {rec.short_description[:300]}\n"
                f"Genres: {genres} | Tags: {tags}\n"
                f"Player reviews: {review_text}"
            )

        context = "\n\n".join(context_blocks)
        prompt = (
            f'You are a helpful Steam game recommendation assistant.\n'
            f'A user is looking for: "{query}"\n\n'
            f'Top matching games with player reviews:\n\n'
            f'{context}\n\n'
            f'Write a friendly 3-4 sentence recommendation paragraph. '
            f'Mention the best-matching games by name and explain why they suit the user\'s request. '
            f'Be concise and enthusiastic.'
        )

        try:
            response = self.ollama_client.chat(
                model=OLLAMA_MODEL,
                messages=[{"role": "user", "content": prompt}],
            )
            return response.message.content.strip()

        except Exception as exc:
            # Graceful fallback: plain summary without LLM
            print(f"[generate_answer] LLM unavailable — returning plain summary. ({exc})")
            names = ", ".join(rec.name for rec, _ in matches[:3])
            return (
                f'Based on your search for "{query}", the closest matches are: {names}. '
                f"(LLM answer unavailable — is Ollama running at {OLLAMA_HOST}?)"
            )
