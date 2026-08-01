"""
Entity Harvester - Web search driven entity pool construction

Core approach:
  For each entity table in each database, auto-generate differentiated search queries,
  obtain real data via Web Search, LLM extracts structured entities from search results,
  build entity pool for LLMDataGenerator use.

Diversity guarantee:
  Not via temperature, but via query dimension combinations (discipline x region x time x institution type).
  Each search round uses different queries -> search engine returns different results -> naturally non-repeating.

Workflow:
  1. LLM analyzes DDL -> identifies entity tables + generates N sets of differentiated search queries
  2. Search per query -> LLM extracts structured entities from search results
  3. Deduplicate and merge -> entity pool JSON
  4. Cache locally (no need to re-search next time)
"""

import json
import http.client
import sys
import time
import random
import re
import concurrent.futures
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import yaml

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from data_synthesis.schema_parser import SchemaInfo, TableInfo, ColumnInfo
from llm.client import LLMClient
from utils.logging_config import get_logger

logger = get_logger(__name__)

# Load search config
def _load_search_config() -> Dict[str, Any]:
    """Load search config from data_synthesis.yaml."""
    config_path = PROJECT_ROOT / "config" / "data_synthesis.yaml"
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        return config.get("entity_harvester", {}).get("search", {})
    except Exception as e:
        logger.warning(f"Failed to load search config: {e}，will use default config")
        return {}

_SEARCH_CONFIG = _load_search_config()


class _SerperKeyManager:
    """Serper API Key rotation manager(thread-safe)"""
    
    def __init__(self, config: Dict[str, Any]):
        import threading
        self._lock = threading.Lock()
        # Compatible with old config serper_api_key (single) and new config serper_api_keys (list)
        keys = config.get("serper_api_keys", [])
        if not keys:
            single_key = config.get("serper_api_key", "")
            keys = [single_key] if single_key else []
        self._keys = [k for k in keys if k]  # Filter empty values
        self._current_idx = 0
        self._exhausted = set()  # Exhausted key indices
    
    @property
    def available(self) -> bool:
        return len(self._keys) > 0 and len(self._exhausted) < len(self._keys)
    
    @property
    def current_key(self) -> str:
        with self._lock:
            if not self.available:
                return ""
            return self._keys[self._current_idx]
    
    def mark_exhausted(self, key: str) -> str:
        """Mark key as exhausted, switch to next available key. Returns new key or empty string."""
        with self._lock:
            # Find this key index and mark
            for i, k in enumerate(self._keys):
                if k == key:
                    self._exhausted.add(i)
                    logger.warning(f"Serper API key ...{key[-6:]} quota exhausted，"
                                   f"used {len(self._exhausted)}/{len(self._keys)} keys")
                    break
            # Find next available key
            for offset in range(1, len(self._keys) + 1):
                next_idx = (self._current_idx + offset) % len(self._keys)
                if next_idx not in self._exhausted:
                    self._current_idx = next_idx
                    logger.info(f"Switched to Serper API key ...{self._keys[next_idx][-6:]}")
                    return self._keys[next_idx]
            logger.error("All Serper API key quotas exhausted!")
            return ""


_SERPER_KEY_MGR = _SerperKeyManager(_SEARCH_CONFIG)

# Try to import DuckDuckGo search library (as fallback)
try:
    from ddgs import DDGS
    HAS_DDGS = True
except ImportError:
    try:
        from duckduckgo_search import DDGS
        HAS_DDGS = True
    except ImportError:
        HAS_DDGS = False

# Try to import web scraping library
try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

try:
    from bs4 import BeautifulSoup
    HAS_BS4 = True
except ImportError:
    HAS_BS4 = False


# =============================================================================
# Web Search wrapper - supports Serper (paid) / DuckDuckGo (free fallback)
# =============================================================================

def _serper_search(query: str, api_key: str, max_results: int = 12) -> List[Dict[str, str]]:
    """
    Use Serper.dev API to execute Google search

    Returns: [{"title": ..., "body": ..., "href": ...}, ...]
    Raises: SerperQuotaExhausted when key quota is exhausted
    """
    conn = http.client.HTTPSConnection("google.serper.dev", timeout=15)
    payload = json.dumps({"q": query, "num": max_results})
    headers = {
        'X-API-KEY': api_key,
        'Content-Type': 'application/json'
    }
    try:
        conn.request("POST", "/search", payload, headers)
        res = conn.getresponse()
        status_code = res.status
        body = res.read().decode("utf-8")
        
        # Detect quota exhausted: HTTP 400 + "Not enough credits"
        if status_code == 400:
            try:
                data = json.loads(body)
                msg = data.get("message", "")
                if "credit" in msg.lower() or "quota" in msg.lower() or "limit" in msg.lower():
                    raise _SerperQuotaExhausted(f"HTTP 400: {msg}")
            except _SerperQuotaExhausted:
                raise
            except Exception:
                pass
        if status_code in (403, 429):
            raise _SerperQuotaExhausted(f"HTTP {status_code}: {body[:200]}")
        if status_code != 200:
            raise Exception(f"Serper API HTTP {status_code}: {body[:200]}")
        
        data = json.loads(body)
        
        # Sometimes returns 200 but body has error message
        if "message" in data and ("limit" in data["message"].lower() or 
                                   "quota" in data["message"].lower() or
                                   "credit" in data["message"].lower()):
            raise _SerperQuotaExhausted(f"API message: {data['message']}")
        
        results = []
        for item in data.get("organic", []):
            results.append({
                "title": item.get("title", ""),
                "body": item.get("snippet", ""),
                "href": item.get("link", "")
            })
        return results
    except _SerperQuotaExhausted:
        raise
    except Exception as e:
        raise e
    finally:
        conn.close()


class _SerperQuotaExhausted(Exception):
    """Serper API key quota exhausted exception"""
    pass


def _ddgs_search(query: str, max_results: int = 8) -> List[Dict[str, str]]:
    """
    Use DuckDuckGo free search (as fallback)
    """
    if not HAS_DDGS:
        return []
    ddgs = DDGS(timeout=15)
    results = list(ddgs.text(query, region='wt-wt', max_results=max_results))
    return [{"title": r.get("title", ""), "body": r.get("body", ""), "href": r.get("href", "")} for r in results]


def web_search(query: str, max_results: int = 8) -> List[Dict[str, str]]:
    """
    Execute web search, return result list

    Each result: {"title": ..., "body": ..., "href": ...}

    Search strategy:
      1. Prefer Serper.dev API (Google search, fast, stable)
         - supports multi-key rotation: auto-switch when a key quota is exhausted
      2. If all Serper keys exhausted or not configured, fall back to DuckDuckGo
    """
    provider = _SEARCH_CONFIG.get("provider", "serper")

    # --- Serper (priority, supports multi-key rotation) ---
    if provider == "serper" and _SERPER_KEY_MGR.available:
        for attempt in range(len(_SERPER_KEY_MGR._keys) + 1):  # Try all keys at most
            api_key = _SERPER_KEY_MGR.current_key
            if not api_key:
                break  # All keys exhausted
            try:
                results = _serper_search(query, api_key, max_results)
                return results
            except _SerperQuotaExhausted:
                # quota exhausted, switch to next key and retry
                new_key = _SERPER_KEY_MGR.mark_exhausted(api_key)
                if not new_key:
                    break  # All keys exhausted
                continue  # Retry with new key
            except Exception as e:
                # Non-quota error, retry once
                if attempt == 0:
                    logger.debug(f"Serper search retry: {e}")
                    time.sleep(0.5)
                else:
                    logger.warning(f"Serper search failed: {e}")
                    break
        
        # All Serper failed, try DuckDuckGo fallback
        if HAS_DDGS:
            logger.info("Serper unavailable, falling back to DuckDuckGo search...")
            try:
                return _ddgs_search(query, max_results)
            except Exception as e2:
                logger.warning(f"DuckDuckGo fallback also failed: {e2}")
            return []

    # --- DuckDuckGo (fallback) ---
    if HAS_DDGS:
        for attempt in range(2):
            try:
                return _ddgs_search(query, max_results)
            except Exception as e:
                if attempt == 0:
                    logger.debug(f"DuckDuckGo search retry: {e}")
                    time.sleep(1)
                else:
                    logger.warning(f"DuckDuckGo search failed (after 2 retries): {e}")
                    return []
    else:
        logger.warning("No Serper API key configured and DuckDuckGo search library not installed")
        return []
    return []


# =============================================================================
# Web page content extraction
# =============================================================================

# Domains not worth scraping (login walls, strict anti-scraping, etc.)
_SKIP_DOMAINS = {
    'youtube.com', 'twitter.com', 'x.com', 'facebook.com', 'instagram.com',
    'linkedin.com', 'tiktok.com', 'reddit.com', 'pinterest.com',
    'amazon.com', 'ebay.com', 'google.com', 'apple.com',
}

_REQUEST_HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
    'Accept-Language': 'en-US,en;q=0.9',
}

# HTML tags to remove
_REMOVE_TAGS = {'script', 'style', 'nav', 'header', 'footer', 'aside',
                'form', 'noscript', 'iframe', 'svg', 'button', 'menu'}


def _extract_text_from_html(html: str, max_chars: int = 6000) -> str:
    """
    Extract body text from HTML

    Strategy: prefer extracting from <article>/<main>, otherwise from <body>,
    remove navigation, footer and other noise tags.
    """
    if not HAS_BS4:
        # Without BeautifulSoup, use regex for simple extraction
        text = re.sub(r'<script[^>]*>.*?</script>', '', html, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r'<style[^>]*>.*?</style>', '', html, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r'<[^>]+>', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text[:max_chars]

    soup = BeautifulSoup(html, 'html.parser')

    # Remove noise tags
    for tag_name in _REMOVE_TAGS:
        for tag in soup.find_all(tag_name):
            tag.decompose()

    # Prefer extracting from article / main
    content_node = soup.find('article') or soup.find('main')
    if content_node is None:
        content_node = soup.find('body')
    if content_node is None:
        content_node = soup

    # Extract text, join block elements with newlines
    lines = []
    for element in content_node.stripped_strings:
        lines.append(element)

    text = '\n'.join(lines)

    # Compress consecutive empty lines
    text = re.sub(r'\n{3,}', '\n\n', text)

    return text[:max_chars]


def fetch_page_content(url: str, timeout: int = 8, max_chars: int = 6000) -> Optional[str]:
    """
    Scrape single web page content

    Args:
        url: Web page URL
        timeout: Request timeout seconds
        max_chars: Max characters of returned content

    Returns:
        Extracted body text, or None on failure
    """
    if not HAS_REQUESTS:
        return None

    # Skip domains not suitable for scraping
    try:
        from urllib.parse import urlparse
        domain = urlparse(url).netloc.lower()
        for skip in _SKIP_DOMAINS:
            if skip in domain:
                return None
    except Exception:
        pass

    try:
        resp = requests.get(url, headers=_REQUEST_HEADERS, timeout=timeout,
                            allow_redirects=True, verify=False)
        resp.raise_for_status()

        # Only process HTML
        content_type = resp.headers.get('Content-Type', '')
        if 'text/html' not in content_type and 'application/xhtml' not in content_type:
            return None

        # Try with correct encoding
        resp.encoding = resp.apparent_encoding or 'utf-8'
        html = resp.text

        text = _extract_text_from_html(html, max_chars=max_chars)

        # If extracted text is too short (< 100 chars), may be anti-scraping page
        if len(text) < 100:
            return None

        return text

    except Exception as e:
        logger.debug(f"Web page scraping failed ({url}): {e}")
        return None


def fetch_pages_concurrent(
    urls: List[str],
    max_workers: int = 4,
    timeout: int = 8,
    max_chars_per_page: int = 6000,
) -> Dict[str, str]:
    """
    Concurrently scrape multiple web page contents

    Args:
        urls: URL list
        max_workers: Max concurrency
        timeout: Timeout seconds per request
        max_chars_per_page: Max characters returned per page

    Returns:
        {url: extracted_text} only includes successfully scraped
    """
    results: Dict[str, str] = {}

    if not urls or not HAS_REQUESTS:
        return results

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_url = {
            executor.submit(fetch_page_content, url, timeout, max_chars_per_page): url
            for url in urls
        }
        for future in concurrent.futures.as_completed(future_to_url):
            url = future_to_url[future]
            try:
                text = future.result()
                if text:
                    results[url] = text
            except Exception:
                pass

    return results


# =============================================================================
# System Prompt - Search Query Generation
# =============================================================================

QUERY_GEN_SYSTEM_PROMPT = """You are a data collection expert. Your task is to generate multiple sets of differentiated search engine queries for database tables,
to collect real data from the internet to populate the table.

Key requirements:
1. Each set of queries must cover different dimension combinations (e.g., different sub-domains, regions, time periods, institution types, etc.)
2. Queries should be specific enough to find result pages containing actual data (names, values, etc.)
3. Write queries in English (richer search results)
4. Output only a JSON array, each element is a search query string"""


# =============================================================================
# System Prompt - Entity Extraction
# =============================================================================

EXTRACT_SYSTEM_PROMPT = """You are a data extraction expert. Your task is to extract structured data from web search results to populate database tables.

Key requirements:
1. Extract as many entity records as possible from search results
2. Even if information is incomplete (some columns have no values), still extract - set missing columns to null
3. If search results mention entity names (people, organizations, conferences, etc.) without complete info, still extract fields you can determine
4. You can also supplement with relevant real information you know based on clues in search results (e.g., if results mention CVPR, you can add its full name and website)
5. Output a JSON array, each element is an object with keys as table column names
6. Skip primary key ID columns (will be auto-assigned later)
7. Aim to extract 15-30 non-duplicate records
8. Output only a JSON array, nothing else"""


# =============================================================================
# EntityHarvester
# =============================================================================

class EntityHarvester:
    """
    Web search driven entity pool builder

    For each entity table:
      1. LLM analyzes DDL -> generates differentiated search queries
      2. Search -> LLM extract -> deduplicate and merge
      3. Cache to output/entity_pools/{db_name}.json
    """

    def __init__(
        self,
        llm_client: Optional[LLMClient] = None,
        provider: str = "aliyun",
        model: str = "qwen3.5-flash",
        queries_per_table: int = 4,
        entities_per_query: int = 20,
        concurrent_tables: int = 4,
        concurrent_queries: int = 3,
        cache_dir: Optional[str] = None,
    ):
        """
        Args:
            llm_client: LLM client (reuse existing)
            provider: LLM provider
            model: LLM model
            queries_per_table: How many search query sets per entity table
            entities_per_query: How many entities expected per search
            concurrent_tables: Multi-table concurrent collection count
            concurrent_queries: Multi-query concurrent search count within single table
            cache_dir: Cache directory
        """
        if llm_client is not None:
            self.llm_client = llm_client
        else:
            self.llm_client = LLMClient(provider=provider, model=model)

        self.queries_per_table = queries_per_table
        self.entities_per_query = entities_per_query
        self.concurrent_tables = concurrent_tables
        self.concurrent_queries = concurrent_queries
        self.cache_dir = Path(cache_dir or str(PROJECT_ROOT / "output" / "entity_pools"))
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # token/cost stats (thread-safe accumulation)
        import threading
        self._stats_lock = threading.Lock()
        self.stats = {"total_tokens": 0, "total_cost": 0.0, "llm_calls": 0}
    
    def _track_response(self, response):
        """Thread-safe accumulation of LLM call stats."""
        with self._stats_lock:
            self.stats["total_tokens"] += getattr(response, 'total_tokens', 0)
            self.stats["total_cost"] += getattr(response, 'cost', 0.0)
            self.stats["llm_calls"] += 1

    # =========================================================================
    # Main entry
    # =========================================================================

    def harvest(
        self,
        schema: SchemaInfo,
        target_counts: Optional[Dict[str, int]] = None,
        use_cache: bool = True,
    ) -> Dict[str, List[Dict]]:
        """
        Collect entity pool for entire database

        Args:
            schema: Database schema info
            target_counts: {table_name: target entity count}, None defaults to 100 per table
            use_cache: Whether to use local cache

        Returns:
            {table_name: [entity_dict, ...]}
        """
        db_name = schema.database_name

        # Check cache
        cache_file = self.cache_dir / f"{db_name}.json"
        if use_cache and cache_file.exists():
            logger.info(f"📦 Loading cached entity pool: {cache_file}")
            with open(cache_file, 'r', encoding='utf-8') as f:
                pool = json.load(f)
            total = sum(len(v) for v in pool.values())
            logger.info(f"   Cache hit: {len(pool)} tables, {total} entities")
            return pool

        logger.info(f"🌐 Start collecting entity pool: {db_name}")

        # Identify entity tables (exclude bridge tables)
        entity_tables = self._identify_entity_tables(schema)
        logger.info(f"   Entity tables: {[t.name for t in entity_tables]}")

        # Concurrently collect multiple entity tables
        pool: Dict[str, List[Dict]] = {}
        
        def _harvest_one(table):
            target = (target_counts or {}).get(table.name, 100)
            try:
                entities = self._harvest_table(schema, table, target_count=target)
                return (table.name, entities)
            except Exception as e:
                logger.error(f"   {table.name} collection failed: {e}")
                return (table.name, [])
        
        max_workers = min(self.concurrent_tables, len(entity_tables))
        if max_workers <= 1:
            # Only 1 table, serial
            for table in entity_tables:
                name, entities = _harvest_one(table)
                if entities:
                    pool[name] = entities
                    logger.info(f"   ✅ {name}: collected {len(entities)} entities")
                else:
                    logger.warning(f"   ⚠️ {name}: no entities collected")
        else:
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {executor.submit(_harvest_one, t): t.name for t in entity_tables}
                for future in concurrent.futures.as_completed(futures):
                    name, entities = future.result()
                    if entities:
                        pool[name] = entities
                        logger.info(f"   ✅ {name}: collected {len(entities)} entities")
                    else:
                        logger.warning(f"   ⚠️ {name}: no entities collected")

        # Save cache
        with open(cache_file, 'w', encoding='utf-8') as f:
            json.dump(pool, f, ensure_ascii=False, indent=2)
        logger.info(f"   💾 Cache saved: {cache_file}")

        total = sum(len(v) for v in pool.values())
        logger.info(
            f"🌐 Entity pool collection complete: {len(pool)} tables, {total} entities, "
            f"{self.stats['llm_calls']} LLM calls, "
            f"{self.stats['total_tokens']:,} tokens, "
            f"${self.stats['total_cost']:.4f}"
        )
        return pool

    # =========================================================================
    # Identify entity tables
    # =========================================================================

    def _identify_entity_tables(self, schema: SchemaInfo) -> List[TableInfo]:
        """
        Identify entity tables to collect (exclude bridge tables and pure numeric tables)

        Entity tables = have TEXT/VARCHAR type non-FK columns and are not pure bridge tables
        """
        entity_tables = []
        for table in schema.tables:
            # Check if it is a bridge table (all columns are PK or FK)
            pk_cols = set(table.primary_key)
            fk_cols = set()
            for fk in table.foreign_keys:
                fk_cols.update(fk.columns)

            non_pk_fk_cols = [c for c in table.columns if c.name not in pk_cols and c.name not in fk_cols]

            # Bridge table: no non-PK/FK columns, or very few
            if len(non_pk_fk_cols) == 0:
                continue

            # Check if it has text columns (columns needing real data)
            has_text_col = any(
                c.is_string_type() and c.name not in fk_cols
                for c in table.columns
            )

            if has_text_col:
                entity_tables.append(table)

        return entity_tables

    # =========================================================================
    # Single table collection
    # =========================================================================

    def _harvest_table(
        self,
        schema: SchemaInfo,
        table: TableInfo,
        target_count: int = 100,
    ) -> List[Dict]:
        """
        Collect entities for a single table

        1. Generate differentiated search queries
        2. Search each + LLM extract
        3. Deduplicate and merge
        """
        db_name = schema.database_name

        # Step 1: Generate search queries
        queries = self._generate_search_queries(schema, table)
        if not queries:
            logger.warning(f"   Failed to generate search queries: {table.name}")
            return []

        logger.info(f"   🔍 {table.name}: generated {len(queries)} search queries")

        # Step 2: Concurrent search + web scraping + LLM extraction
        all_entities: List[Dict] = []
        seen_keys: set = set()
        dedup_fields = self._get_dedup_fields(table)
        fallback_counter = 0

        def _search_and_extract(query_idx_and_query):
            idx, query = query_idx_and_query
            logger.info(f"     {table.name} query {idx+1}/{len(queries)}: searching... | {query[:60]}")
            
            # DuckDuckGo high concurrency causes rate limiting, add brief interval
            time.sleep(idx * 0.5)
            
            search_results = web_search(query, max_results=8)
            if not search_results:
                logger.info(f"     {table.name} query {idx+1}: no search results")
                return (idx, query, [])
            
            logger.info(f"     {table.name} query {idx+1}: found {len(search_results)} results, LLM extracting...")
            
            # Use search snippets (title + body) for LLM extraction, no full page scraping
            # Search snippets are sufficient for extracting entity names, org names, etc.
            entities = self._extract_entities(table, search_results, existing=None)
            
            logger.info(f"     {table.name} query {idx+1}: extracted {len(entities)} entities")
            return (idx, query, entities)

        # Concurrently execute all queries (max 3 concurrent, avoid search API rate limiting)
        query_tasks = list(enumerate(queries))
        search_workers = min(self.concurrent_queries, len(query_tasks))
        
        if search_workers <= 1:
            results = [_search_and_extract(t) for t in query_tasks]
        else:
            results = []
            with concurrent.futures.ThreadPoolExecutor(max_workers=search_workers) as executor:
                futures = [executor.submit(_search_and_extract, t) for t in query_tasks]
                for future in concurrent.futures.as_completed(futures):
                    results.append(future.result())
        
        # Sort by original order then deduplicate and merge
        results.sort(key=lambda x: x[0])
        
        for idx, query, entities in results:
            if len(all_entities) >= target_count:
                break
            
            new_count = 0
            for entity in entities:
                key = self._entity_key(entity, dedup_fields)
                if key is None:
                    fallback_counter += 1
                    key = f"__fallback_{fallback_counter}__"
                if key not in seen_keys:
                    seen_keys.add(key)
                    all_entities.append(entity)
                    new_count += 1

            logger.info(
                f"     query {idx+1}/{len(queries)}: "
                f"extracted {len(entities)} → new {new_count} "
                f"(total {len(all_entities)}) | {query}"
            )

        return all_entities[:target_count]

    # =========================================================================
    # Search Query Generation
    # =========================================================================

    def _generate_search_queries(
        self,
        schema: SchemaInfo,
        table: TableInfo,
    ) -> List[str]:
        """LLM analyzes DDL -> generates differentiated search queries"""

        # Build table description
        col_desc = ", ".join(
            f"{c.name}({c.data_type})" for c in table.columns
            if c.is_string_type() or c.name.lower() in ('name', 'title')
        )

        # Build foreign key description
        fk_desc = ""
        if table.foreign_keys:
            fk_parts = [f"{fk.columns}→{fk.ref_table}" for fk in table.foreign_keys]
            fk_desc = f"\nForeign key relationships: {', '.join(fk_parts)}"

        user_prompt = f"""Database: {schema.database_name}
Table name: {table.name}
Text columns: {col_desc}
{fk_desc}

All tables in DDL: {', '.join(t.name for t in schema.tables)}

Please generate for this table {self.queries_per_table} differentiated English search queries for collecting real {table.name} data from the internet.

Requirements:
- Each query covers different sub-domain/region/time/type dimensions
- Queries should be specific enough to find pages with extractable real values for table columns
- Broad coverage, do not focus on one domain

Output JSON array, each element is a query string."""

        try:
            response = self.llm_client.complete(
                system_prompt=QUERY_GEN_SYSTEM_PROMPT,
                user_prompt=user_prompt,
                temperature=0.7,
                max_tokens=2048,
            )
            self._track_response(response)
            queries = self._parse_json_array(response.content)
            if isinstance(queries, list) and all(isinstance(q, str) for q in queries):
                return queries[:self.queries_per_table]
        except Exception as e:
            logger.error(f"Failed to generate search queries: {e}")

        return []

    # =========================================================================
    # Entity extraction
    # =========================================================================

    def _extract_entities(
        self,
        table: TableInfo,
        search_results: List[Dict[str, str]],
        existing: Optional[List[Dict]] = None,
    ) -> List[Dict]:
        """Extract structured entities from search results (supports web page content)"""

        # Build search result text (including web page content)
        results_text = ""
        total_chars = 0
        max_total_chars = 24000  # Control total text to avoid exceeding token limit

        for i, r in enumerate(search_results):
            if total_chars >= max_total_chars:
                break

            entry = f"\n[{i+1}] {r['title']}\nSummary: {r['body']}\n"

            # If web page content available, append key parts
            page_content = r.get('page_content', '')
            if page_content:
                # Calculate available content character budget for this entry
                remaining = max_total_chars - total_chars - len(entry) - 100
                if remaining > 500:
                    truncated = page_content[:min(remaining, 4000)]
                    entry += f"Web page content:\n{truncated}\n"

            results_text += entry
            total_chars += len(entry)

        # Build column description (exclude auto-increment PK)
        columns_desc = []
        for col in table.columns:
            if col.is_auto_increment:
                continue
            if col.is_primary_key and col.is_integer_type():
                continue  # Skip integer PK, auto-assigned later
            columns_desc.append(f"- {col.name}: {col.data_type}")

        columns_text = "\n".join(columns_desc)

        # Existing entity summary (prevent duplicates)
        existing_hint = ""
        if existing and len(existing) > 0:
            dedup_fields = self._get_dedup_fields(table)
            if dedup_fields:
                existing_values = set()
                for e in existing:
                    for f in dedup_fields:
                        v = e.get(f)
                        if v and isinstance(v, str):
                            existing_values.add(v)
                if existing_values:
                    sample = sorted(existing_values)[:30]
                    existing_hint = (
                        f"\n\nAlready collected values (do not extract duplicates):\n{', '.join(sample)}"
                    )

        user_prompt = f"""Table name: {table.name}
Columns to extract:
{columns_text}

Search results:
{results_text}
{existing_hint}

Please extract as many {table.name} records as possible from the above search results.
Output JSON array, each record contains above column names as keys.
Skip primary key ID columns. Set null for columns with no value found.
Output only JSON array."""

        try:
            response = self.llm_client.complete(
                system_prompt=EXTRACT_SYSTEM_PROMPT,
                user_prompt=user_prompt,
                temperature=0.3,  # Use low temperature for extraction, ensure accuracy
                max_tokens=8192,
            )
            self._track_response(response)
            entities = self._parse_json_array(response.content)
            if isinstance(entities, list):
                return [e for e in entities if isinstance(e, dict)]
        except Exception as e:
            logger.error(f"Entity extraction failed: {e}")

        return []

    # =========================================================================
    # Utility methods
    # =========================================================================

    def _get_dedup_fields(self, table: TableInfo) -> List[str]:
        """Get fields used for deduplication."""
        # Exact match dedup field names
        dedup_names = {'name', 'shortname', 'short_name', 'title', 'fullname',
                       'full_name', 'superhero_name', 'label', 'tag_name',
                       'community_area_name', 'district_name', 'city_name'}
        fields = []
        for col in table.columns:
            if col.name.lower() in dedup_names and col.is_string_type():
                fields.append(col.name)
        
        # If exact match not found, try fuzzy match (column name contains name/title/school/label keywords)
        if not fields:
            fuzzy_keywords = ['name', 'title', 'school', 'label', 'description']
            for col in table.columns:
                if col.is_string_type() and not col.is_primary_key:
                    col_lower = col.name.lower()
                    if any(kw in col_lower for kw in fuzzy_keywords):
                        fields.append(col.name)
        
        # If still not found, use first text column
        if not fields:
            for col in table.columns:
                if col.is_string_type() and not col.is_primary_key:
                    fields.append(col.name)
                    break
        return fields

    def _entity_key(self, entity: Dict, dedup_fields: List[str]) -> Optional[str]:
        """Generate dedup key for entity"""
        parts = []
        for f in dedup_fields:
            v = entity.get(f)
            if v and isinstance(v, str):
                parts.append(v.strip().lower())
        return "|".join(parts) if parts else None

    def _parse_json_array(self, content: str) -> Any:
        """Parse JSON array (with error tolerance)."""
        text = content.strip()

        # Remove markdown code block
        if '```json' in text:
            text = text.split('```json')[1].split('```')[0]
        elif '```' in text:
            parts = text.split('```')
            if len(parts) >= 3:
                text = parts[1]

        text = text.strip()

        try:
            return json.loads(text)
        except json.JSONDecodeError:
            # Try to find first [ and last ]
            start = text.find('[')
            end = text.rfind(']')
            if start != -1 and end != -1 and end > start:
                try:
                    return json.loads(text[start:end+1])
                except json.JSONDecodeError:
                    pass
        return []


# =============================================================================
# CLI entry
# =============================================================================

def main():
    """CLI entry - run entity collection standalone."""
    import argparse

    parser = argparse.ArgumentParser(description="Entity harvester")
    parser.add_argument("--db", type=str, required=True, help="Database name")
    parser.add_argument("--database-dir", default=str(PROJECT_ROOT / "database"), help="Database directory")
    parser.add_argument("--queries-per-table", type=int, default=8, help="Search queries per table")
    parser.add_argument("--no-cache", action="store_true", help="Do not use cache")
    parser.add_argument("--provider", default="aliyun", help="LLM provider")
    parser.add_argument("--model", default="qwen3.5-flash", help="LLM model")

    args = parser.parse_args()

    from data_synthesis.schema_parser import parse_all_dual_schemas

    # Parse schema
    schemas = parse_all_dual_schemas(args.database_dir)
    if args.db not in schemas:
        logger.error(f"Database not found: {args.db}")
        return

    dual_schema = schemas[args.db]
    schema = dual_schema.mysql_schema or dual_schema.pg_schema

    # Collect
    harvester = EntityHarvester(
        provider=args.provider,
        model=args.model,
        queries_per_table=args.queries_per_table,
    )
    pool = harvester.harvest(schema, use_cache=not args.no_cache)

    # Print results
    for table_name, entities in pool.items():
        print(f"\n{table_name}: {len(entities)} entities")
        for e in entities[:3]:
            print(f"  {e}")
        if len(entities) > 3:
            print(f"  ... ({len(entities) - 3} more)")


if __name__ == "__main__":
    main()
