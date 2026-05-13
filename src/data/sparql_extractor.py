"""SPARQL extraction from Wikidata with SQLite caching and retry logic.

This module handles all communication with the Wikidata SPARQL endpoint.
It implements Gap #3 requirements:
  - SQLite caching keyed by SHA256 of query string
  - 7-day cache TTL (configurable)
  - Exponential backoff retry (max 5 retries, 2^n seconds)
  - Pagination with LIMIT/OFFSET for full mode
  - --force-refresh flag to bypass cache
  - Graceful degradation when endpoint is down (serve from cache)

Why this works:
    Wikidata's public SPARQL endpoint throttles heavy users aggressively.
    Without caching, every pipeline run re-queries and risks getting blocked.
    The SQLite cache stores results keyed by query hash, so identical queries
    return instantly from disk. Exponential backoff handles transient failures.
"""

import hashlib
import json
import sqlite3
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Optional

from loguru import logger

try:
    from SPARQLWrapper import SPARQLWrapper, JSON as SPARQL_JSON
    from SPARQLWrapper.SPARQLExceptions import (
        EndPointNotFound,
        EndPointInternalError,
        QueryBadFormed,
    )
except ImportError:
    logger.error("SPARQLWrapper not installed. Run: pip install SPARQLWrapper==2.0.0")
    raise

from src.utils.text_utils import extract_qid_from_uri, clean_entity_label


# ---------------------------------------------------------------------------
# SPARQL query templates for the tech company domain
# {LANG} is replaced with the language code (de, ja, nl, en)
# {LIMIT} and {OFFSET} are replaced for pagination
# ---------------------------------------------------------------------------

QUERY_TECH_COMPANIES = """
SELECT DISTINCT ?company ?companyLabel ?country ?countryLabel
       ?industry ?industryLabel WHERE {{
  ?company wdt:P31/wdt:P279* wd:Q783794.
  ?company wdt:P452 ?industry.
  FILTER(?industry IN (wd:Q11661, wd:Q21091, wd:Q1371849))
  OPTIONAL {{ ?company wdt:P17 ?country }}
  SERVICE wikibase:label {{
    bd:serviceParam wikibase:language "{lang},en".
  }}
}}
LIMIT {limit} OFFSET {offset}
"""

QUERY_ACQUISITIONS = """
SELECT ?acquirer ?acquirerLabel ?target ?targetLabel WHERE {{
  ?acquirer wdt:P31/wdt:P279* wd:Q783794.
  ?acquirer wdt:P452 ?industry.
  FILTER(?industry IN (wd:Q11661, wd:Q21091, wd:Q1371849))
  ?acquirer wdt:P355 ?target.
  SERVICE wikibase:label {{
    bd:serviceParam wikibase:language "{lang},en".
  }}
}}
LIMIT {limit} OFFSET {offset}
"""

QUERY_FOUNDERS = """
SELECT ?company ?companyLabel ?founder ?founderLabel WHERE {{
  ?company wdt:P31/wdt:P279* wd:Q783794.
  ?company wdt:P452 ?industry.
  FILTER(?industry IN (wd:Q11661, wd:Q21091, wd:Q1371849))
  ?company wdt:P112 ?founder.
  SERVICE wikibase:label {{
    bd:serviceParam wikibase:language "{lang},en".
  }}
}}
LIMIT {limit} OFFSET {offset}
"""

QUERY_HEADQUARTERS = """
SELECT ?company ?companyLabel ?hq ?hqLabel ?country ?countryLabel WHERE {{
  ?company wdt:P31/wdt:P279* wd:Q783794.
  ?company wdt:P452 ?industry.
  FILTER(?industry IN (wd:Q11661, wd:Q21091, wd:Q1371849))
  ?company wdt:P159 ?hq.
  ?hq wdt:P17 ?country.
  SERVICE wikibase:label {{
    bd:serviceParam wikibase:language "{lang},en".
  }}
}}
LIMIT {limit} OFFSET {offset}
"""

# Maps query type names to (template, relation_name, head_fields, tail_fields)
# head_fields/tail_fields: (uri_key, label_key) in SPARQL results
QUERY_REGISTRY: dict[str, tuple[str, str, tuple[str, str], tuple[str, str]]] = {
    "tech_companies_country": (
        QUERY_TECH_COMPANIES, "located_in",
        ("company", "companyLabel"), ("country", "countryLabel"),
    ),
    "tech_companies_industry": (
        QUERY_TECH_COMPANIES, "industry_of",
        ("company", "companyLabel"), ("industry", "industryLabel"),
    ),
    "acquisitions": (
        QUERY_ACQUISITIONS, "subsidiary_of",
        ("acquirer", "acquirerLabel"), ("target", "targetLabel"),
    ),
    "founders": (
        QUERY_FOUNDERS, "founded_by",
        ("company", "companyLabel"), ("founder", "founderLabel"),
    ),
    "headquarters": (
        QUERY_HEADQUARTERS, "headquartered_in",
        ("company", "companyLabel"), ("hq", "hqLabel"),
    ),
}


class WikidataSPARQLExtractor:
    """Extracts structured triples from Wikidata's SPARQL endpoint.

    Handles query execution, SQLite caching, exponential backoff retry,
    and pagination. Returns triples in a standardized format.

    Args:
        config: Full configuration dictionary (from config.yaml).
        force_refresh: If True, bypass cache and re-query everything.

    Example:
        >>> import yaml
        >>> config = yaml.safe_load(open("config.yaml"))
        >>> extractor = WikidataSPARQLExtractor(config)
        >>> triples = extractor.extract_all_languages(sample_mode=True)
        >>> len(triples["de"])  # Number of German triples
    """

    def __init__(self, config: dict, force_refresh: bool = False) -> None:
        data_cfg = config["data"]
        self.endpoint: str = data_cfg["sparql_endpoint"]
        self.max_retries: int = data_cfg["sparql_max_retries"]
        self.retry_base: int = data_cfg["sparql_retry_base_seconds"]
        self.cache_ttl_days: int = data_cfg["cache_ttl_days"]
        self.sample_limit: int = data_cfg["sample_limit"]
        self.full_page_size: int = data_cfg["full_page_size"]
        self.languages: list[str] = data_cfg["languages"]
        self.force_refresh: bool = force_refresh

        # Set up SQLite cache directory and database
        cache_dir = Path(data_cfg["cache_dir"])
        cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_db_path: Path = cache_dir / data_cfg["cache_db_name"]
        self._init_cache_db()

        logger.info(
            f"Extractor initialized | endpoint={self.endpoint} "
            f"| cache={self.cache_db_path} | force_refresh={force_refresh}"
        )

    # ------------------------------------------------------------------
    # SQLite cache management
    # ------------------------------------------------------------------

    def _init_cache_db(self) -> None:
        """Create the cache table if it doesn't exist."""
        with sqlite3.connect(str(self.cache_db_path)) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS sparql_cache (
                    query_hash    TEXT PRIMARY KEY,
                    query_text    TEXT NOT NULL,
                    results       TEXT NOT NULL,
                    language      TEXT NOT NULL,
                    query_type    TEXT NOT NULL,
                    result_count  INTEGER NOT NULL,
                    created_at    TEXT NOT NULL,
                    expires_at    TEXT NOT NULL
                )
            """)
            conn.commit()

    @staticmethod
    def _get_cache_key(query: str) -> str:
        """SHA256 hash of the query string, used as cache lookup key.

        Args:
            query: Full SPARQL query string.

        Returns:
            Hex-encoded SHA256 hash string.
        """
        return hashlib.sha256(query.encode("utf-8")).hexdigest()

    def _check_cache(self, query_hash: str) -> Optional[list[dict]]:
        """Check if unexpired cached results exist for this query hash.

        Args:
            query_hash: SHA256 hash of the query.

        Returns:
            List of result dicts if cache hit and not expired, else None.
        """
        with sqlite3.connect(str(self.cache_db_path)) as conn:
            cursor = conn.execute(
                "SELECT results, expires_at FROM sparql_cache WHERE query_hash = ?",
                (query_hash,),
            )
            row = cursor.fetchone()

        if row is None:
            return None

        results_json, expires_at_str = row
        expires_at = datetime.fromisoformat(expires_at_str)

        if expires_at < datetime.now():
            logger.debug(f"Cache expired | hash={query_hash[:16]}...")
            return None

        return json.loads(results_json)

    def _store_cache(
        self,
        query_hash: str,
        query: str,
        results: list[dict],
        language: str,
        query_type: str,
    ) -> None:
        """Store query results in SQLite cache.

        Args:
            query_hash: SHA256 hash of the query.
            query: Original SPARQL query text (stored for debugging).
            results: List of result dicts to cache.
            language: Language code (de/ja/nl/en).
            query_type: Query type name from QUERY_REGISTRY.
        """
        expires_at = datetime.now() + timedelta(days=self.cache_ttl_days)
        with sqlite3.connect(str(self.cache_db_path)) as conn:
            conn.execute(
                """INSERT OR REPLACE INTO sparql_cache
                   (query_hash, query_text, results, language, query_type,
                    result_count, created_at, expires_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    query_hash,
                    query,
                    json.dumps(results, ensure_ascii=False),
                    language,
                    query_type,
                    len(results),
                    datetime.now().isoformat(),
                    expires_at.isoformat(),
                ),
            )
            conn.commit()
        logger.debug(
            f"Cache stored | type={query_type} | lang={language} "
            f"| count={len(results)} | hash={query_hash[:16]}"
        )

    # ------------------------------------------------------------------
    # SPARQL query execution with retry
    # ------------------------------------------------------------------

    def _execute_sparql(self, query: str) -> list[dict]:
        """Execute a SPARQL query with exponential backoff retry.

        Args:
            query: Complete SPARQL query string.

        Returns:
            List of result binding dicts from the SPARQL endpoint.

        Raises:
            RuntimeError: If all retries are exhausted.
        """
        sparql = SPARQLWrapper(self.endpoint)
        sparql.setQuery(query)
        sparql.setReturnFormat(SPARQL_JSON)
        # Polite user-agent so Wikidata admins can contact us if needed
        sparql.addCustomHttpHeader(
            "User-Agent",
            "CrossLingualKG/0.1 (Research Project; Python SPARQLWrapper)",
        )

        last_error: Optional[Exception] = None
        for attempt in range(self.max_retries):
            try:
                results = sparql.query().convert()
                bindings = results.get("results", {}).get("bindings", [])
                return bindings

            except (EndPointNotFound, EndPointInternalError) as e:
                last_error = e
                wait_time = self.retry_base ** (attempt + 1)
                logger.warning(
                    f"SPARQL endpoint error | attempt={attempt + 1}/{self.max_retries} "
                    f"| error={type(e).__name__}: {str(e)[:100]} "
                    f"| retry_in={wait_time}s"
                )
                time.sleep(wait_time)

            except QueryBadFormed as e:
                # Query syntax errors should not be retried
                logger.error(f"SPARQL query malformed | error={str(e)[:200]}")
                raise RuntimeError(f"Malformed SPARQL query: {e}") from e

            except Exception as e:
                last_error = e
                wait_time = self.retry_base ** (attempt + 1)
                logger.warning(
                    f"SPARQL query failed | attempt={attempt + 1}/{self.max_retries} "
                    f"| error={type(e).__name__}: {str(e)[:100]} "
                    f"| retry_in={wait_time}s"
                )
                if attempt < self.max_retries - 1:
                    time.sleep(wait_time)

        raise RuntimeError(
            f"SPARQL query failed after {self.max_retries} retries. "
            f"Last error: {last_error}"
        )

    # ------------------------------------------------------------------
    # Query execution with caching layer
    # ------------------------------------------------------------------

    def _run_cached_query(
        self,
        query: str,
        language: str,
        query_type: str,
    ) -> list[dict]:
        """Execute a query, checking cache first.

        Logs truncated query string on cache miss (per Parineeta's request).

        Args:
            query: Complete SPARQL query string.
            language: Language code.
            query_type: Query type name.

        Returns:
            List of result binding dicts.
        """
        query_hash = self._get_cache_key(query)

        # Check cache (unless force refresh)
        if not self.force_refresh:
            cached = self._check_cache(query_hash)
            if cached is not None:
                logger.info(
                    f"Cache hit | type={query_type} | lang={language} "
                    f"| results={len(cached)} | hash={query_hash[:16]}"
                )
                return cached

        # Cache miss — log the query for debugging
        truncated_query = query.replace("\n", " ")[:200]
        logger.info(
            f"Cache miss | type={query_type} | lang={language} "
            f"| hash={query_hash[:16]} | query={truncated_query}"
        )

        try:
            bindings = self._execute_sparql(query)
        except RuntimeError:
            # Endpoint is down — try to serve stale cache
            logger.warning(
                f"Endpoint unreachable, checking stale cache | type={query_type} "
                f"| lang={language}"
            )
            stale = self._check_stale_cache(query_hash)
            if stale is not None:
                logger.warning(
                    f"Serving STALE cache (expired) | type={query_type} "
                    f"| lang={language} | results={len(stale)}"
                )
                return stale
            raise

        # Store in cache
        self._store_cache(query_hash, query, bindings, language, query_type)
        logger.info(
            f"Query complete | type={query_type} | lang={language} "
            f"| results={len(bindings)}"
        )
        return bindings

    def _check_stale_cache(self, query_hash: str) -> Optional[list[dict]]:
        """Check for expired cache entries (fallback when endpoint is down).

        Args:
            query_hash: SHA256 hash of the query.

        Returns:
            Cached results regardless of expiry, or None if not found.
        """
        with sqlite3.connect(str(self.cache_db_path)) as conn:
            cursor = conn.execute(
                "SELECT results FROM sparql_cache WHERE query_hash = ?",
                (query_hash,),
            )
            row = cursor.fetchone()

        if row is None:
            return None
        return json.loads(row[0])

    # ------------------------------------------------------------------
    # Triple extraction from SPARQL bindings
    # ------------------------------------------------------------------

    @staticmethod
    def _get_binding_value(binding: dict, key: str) -> Optional[str]:
        """Safely extract a value from a SPARQL result binding.

        Args:
            binding: Single result row from SPARQL query.
            key: Key name to look up.

        Returns:
            The value string, or None if the key is missing.
        """
        if key in binding and "value" in binding[key]:
            return binding[key]["value"]
        return None

    def _bindings_to_triples(
        self,
        bindings: list[dict],
        language: str,
        relation: str,
        head_uri_key: str,
        head_label_key: str,
        tail_uri_key: str,
        tail_label_key: str,
    ) -> list[dict[str, str]]:
        """Convert raw SPARQL bindings into standardized triple dicts.

        Each triple has the format:
            {
                "head": "SAP SE",
                "head_id": "Q80994",
                "relation": "headquartered_in",
                "tail": "Walldorf",
                "tail_id": "Q3955",
                "language": "de"
            }

        Args:
            bindings: Raw SPARQL result bindings.
            language: Language code.
            relation: Relation name for these triples.
            head_uri_key: Key for head entity URI in bindings.
            head_label_key: Key for head entity label in bindings.
            tail_uri_key: Key for tail entity URI in bindings.
            tail_label_key: Key for tail entity label in bindings.

        Returns:
            List of triple dicts with head, head_id, relation, tail, tail_id, language.
        """
        triples: list[dict[str, str]] = []

        for binding in bindings:
            head_uri = self._get_binding_value(binding, head_uri_key)
            head_label = self._get_binding_value(binding, head_label_key)
            tail_uri = self._get_binding_value(binding, tail_uri_key)
            tail_label = self._get_binding_value(binding, tail_label_key)

            # Skip if any required field is missing
            if not all([head_uri, head_label, tail_uri, tail_label]):
                continue

            head_id = extract_qid_from_uri(head_uri)
            tail_id = extract_qid_from_uri(tail_uri)

            if not head_id or not tail_id:
                continue

            triples.append({
                "head": clean_entity_label(head_label, language),
                "head_id": head_id,
                "relation": relation,
                "tail": clean_entity_label(tail_label, language),
                "tail_id": tail_id,
                "language": language,
            })

        return triples

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract_for_language(
        self,
        language: str,
        sample_mode: bool = True,
    ) -> list[dict[str, str]]:
        """Extract all triples for a single language.

        Runs all query types (companies, acquisitions, founders, headquarters)
        and collects triples. In sample mode, uses a single page per query.
        In full mode, paginates until results are exhausted.

        Args:
            language: Language code (de, ja, nl, en).
            sample_mode: If True (default), limit each query to sample_limit.
                If False, paginate through all available results.

        Returns:
            List of triple dicts for this language.

        Example:
            >>> triples = extractor.extract_for_language("de", sample_mode=True)
            >>> len(triples)
            150
        """
        all_triples: list[dict[str, str]] = []
        # Track which query templates we've already run to avoid duplicate queries
        # (tech_companies_country and tech_companies_industry use the same template)
        executed_templates: dict[str, list[dict]] = {}

        for query_type, (template, relation, head_fields, tail_fields) in QUERY_REGISTRY.items():
            template_key = self._get_cache_key(template)

            if sample_mode:
                # Sample mode: single page with sample_limit
                query = template.format(
                    lang=language, limit=self.sample_limit, offset=0
                )

                # Check if we already executed this exact template
                if template_key in executed_templates:
                    bindings = executed_templates[template_key]
                else:
                    bindings = self._run_cached_query(query, language, query_type)
                    executed_templates[template_key] = bindings

                triples = self._bindings_to_triples(
                    bindings, language, relation,
                    head_fields[0], head_fields[1],
                    tail_fields[0], tail_fields[1],
                )
                all_triples.extend(triples)

            else:
                # Full mode: paginate until no more results
                offset = 0
                page_num = 0
                while True:
                    query = template.format(
                        lang=language, limit=self.full_page_size, offset=offset
                    )

                    if template_key in executed_templates and offset == 0:
                        # Only reuse cache for first page of same template
                        bindings = executed_templates[template_key]
                    else:
                        bindings = self._run_cached_query(
                            query, language, f"{query_type}_p{page_num}"
                        )
                        if offset == 0:
                            executed_templates[template_key] = bindings

                    if not bindings:
                        logger.debug(
                            f"Pagination complete | type={query_type} "
                            f"| lang={language} | pages={page_num}"
                        )
                        break

                    triples = self._bindings_to_triples(
                        bindings, language, relation,
                        head_fields[0], head_fields[1],
                        tail_fields[0], tail_fields[1],
                    )
                    all_triples.extend(triples)

                    # If we got fewer results than the page size, we're done
                    if len(bindings) < self.full_page_size:
                        break

                    offset += self.full_page_size
                    page_num += 1

                    # Polite delay between pages to avoid throttling
                    time.sleep(1.0)

        logger.info(
            f"Extraction complete | lang={language} | total_triples={len(all_triples)} "
            f"| mode={'sample' if sample_mode else 'full'}"
        )
        return all_triples

    def extract_all_languages(
        self,
        sample_mode: bool = True,
    ) -> dict[str, list[dict[str, str]]]:
        """Extract triples for all configured languages.

        Args:
            sample_mode: If True (default), use sample_limit per query.
                If False, paginate through all results.

        Returns:
            Dict mapping language codes to lists of triple dicts.
            Example: {"de": [...], "ja": [...], "nl": [...], "en": [...]}
        """
        results: dict[str, list[dict[str, str]]] = {}

        for lang in self.languages:
            logger.info(f"--- Extracting language: {lang.upper()} ---")
            triples = self.extract_for_language(lang, sample_mode=sample_mode)
            results[lang] = triples

            # Polite delay between languages to avoid throttling
            time.sleep(2.0)

        # Summary stats
        total = sum(len(t) for t in results.values())
        per_lang = {lang: len(t) for lang, t in results.items()}
        logger.info(f"All extractions complete | total={total} | per_lang={per_lang}")

        return results

    def get_cache_stats(self) -> dict[str, Any]:
        """Return statistics about the current cache state.

        Returns:
            Dict with cache statistics (total entries, per language, etc.)

        Example:
            >>> extractor.get_cache_stats()
            {'total_entries': 16, 'per_language': {'de': 4, 'ja': 4, ...}}
        """
        with sqlite3.connect(str(self.cache_db_path)) as conn:
            cursor = conn.execute("SELECT COUNT(*) FROM sparql_cache")
            total = cursor.fetchone()[0]

            cursor = conn.execute(
                "SELECT language, COUNT(*), SUM(result_count) "
                "FROM sparql_cache GROUP BY language"
            )
            per_lang = {
                row[0]: {"queries": row[1], "total_results": row[2]}
                for row in cursor.fetchall()
            }

        return {"total_entries": total, "per_language": per_lang}
