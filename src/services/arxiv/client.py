import asyncio
import logging
import time
import xml.etree.ElementTree as ET
from functools import cached_property
from pathlib import Path
from typing import Dict, List, Optional
from urllib.parse import quote, urlencode

import httpx
from src.config import ArxivSettings
from src.exceptions import ArxivAPIException, ArxivAPITimeoutError, ArxivParseError, PDFDownloadException, PDFDownloadTimeoutError
from src.schemas.arxiv.paper import ArxivPaper

logger = logging.getLogger(__name__)


class ArxivClient:
    """Async client for the arXiv public API.

    Handles three responsibilities:
    1. Querying arXiv for paper metadata (by category, custom query, or ID).
    2. Parsing arXiv's Atom XML response into typed ArxivPaper objects.
    3. Downloading the PDFs of those papers to a local cache.

    All HTTP traffic goes through httpx.AsyncClient with a shared rate-limit
    gate (arXiv's terms of service ask for ~3s between requests). The client
    is stateful only for rate limiting (_last_request_time) — everything else
    is derived from the injected ArxivSettings.

    Errors are normalized into the project's typed exception hierarchy
    (ArxivAPIException, ArxivAPITimeoutError, ArxivParseError, etc.) so
    callers can handle failure modes without depending on httpx internals.
    """

    def __init__(self, settings: ArxivSettings):
        """Initialize the client.

        Args:
            settings: Configuration object holding base URL, rate limit,
                timeout, default page size, search category, and XML
                namespace map. All settings are exposed via @property
                accessors so the rest of the class never touches the
                settings object directly.
        """
        self._settings = settings
        self._last_request_time: Optional[float] = None

    ## use of @cached_property for pdf_cache_dir allows lazy initialization of the cache directory on first access, ensuring that we don't touch 
    ## the filesystem until we actually need to download a PDF. This is a performance optimization that avoids unnecessary directory creation if 
    ## the client is only used for metadata fetching. The property will create the directory if it doesn't exist and then memoize the Path object for 
    ## future accesses, so we only pay the cost of checking/creating the directory once per client instance.
    @cached_property
    def pdf_cache_dir(self) -> Path:
        """Local directory where downloaded PDFs are cached.

        Created lazily on first access and memoized for the lifetime of the
        client instance. Using @cached_property avoids touching the
        filesystem until a download is actually requested.
        """
        cache_dir = Path(self._settings.pdf_cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        return cache_dir

    ## The following @property methods provide clean access to the various configuration settings defined in ArxivSettings. 
    ## This encapsulation allows the rest of the class to refer to these settings as simple attributes (e.g. self.base_url) 
    ## without needing to know about the underlying settings object. It also provides a single place to add any additional logic related to these settings 
    ## in the future (e.g. validation, transformation) without changing the rest of the code that uses them.
    @property
    def base_url(self) -> str:
        """arXiv API base URL (e.g. https://export.arxiv.org/api/query)."""
        return self._settings.base_url

    @property
    def namespaces(self) -> dict:
        """XML namespace map used by ElementTree to resolve atom: prefixes."""
        return self._settings.namespaces

    @property
    def rate_limit_delay(self) -> float:
        """Minimum seconds between API calls (arXiv recommends 3s)."""
        return self._settings.rate_limit_delay

    @property
    def timeout_seconds(self) -> int:
        """HTTP request timeout for metadata calls."""
        return self._settings.timeout_seconds

    @property
    def max_results(self) -> int:
        """Default page size when the caller does not pass max_results."""
        return self._settings.max_results

    @property
    def search_category(self) -> str:
        """Default arXiv category (e.g. cs.AI) used by fetch_papers."""
        return self._settings.search_category

    ## why using async def for fetch_papers and other methods?
    ## Because these methods involve making HTTP requests to the arXiv API, which can be slow and may involve waiting for network responses. 
    ## By defining these methods as async, we can use the   asyncio library to allow other tasks to run concurrently while waiting for the HTTP responses, 
    ## improving the overall efficiency and responsiveness of the application. This is especially important if the client is used in a context where 
    ## multiple requests are made or if the client is part of a larger application that needs to remain responsive while waiting for external API   
    async def fetch_papers(
        self,
        max_results: Optional[int] = None,
        start: int = 0,
        sort_by: str = "submittedDate",
        sort_order: str = "descending",
        from_date: Optional[str] = None,
        to_date: Optional[str] = None,
    ) -> List[ArxivPaper]:
        """Fetch papers for the configured arXiv category.

        Builds a category-scoped search query, optionally narrowed by a
        submission date range, and returns parsed ArxivPaper objects. This
        is the most common entry point for batch ingestion (e.g. "give me
        the latest cs.AI papers from yesterday").

        Args:
            max_results: Page size. Falls back to settings.max_results when
                None. Hard-capped at 2000 (arXiv's per-request limit).
            start: Pagination offset.
            sort_by: One of "submittedDate", "lastUpdatedDate", "relevance".
            sort_order: "ascending" or "descending".
            from_date: Inclusive lower bound (YYYYMMDD). Combined with
                "0000" to mean start of day.
            to_date: Inclusive upper bound (YYYYMMDD). Combined with
                "2359" to mean end of day.

        Returns:
            List of ArxivPaper objects (may be empty).

        Raises:
            ArxivAPITimeoutError: HTTP timeout from arXiv.
            ArxivAPIException: Non-2xx response or any other request failure.
            ArxivParseError: Response was received but XML could not be parsed.
        """
        if max_results is None:
            max_results = self.max_results

        # Build search query
        search_query = f"cat:{self.search_category}"

        # Add date filtering if provided
        if from_date or to_date:
            # Convert dates to arXiv format (YYYYMMDDHHMM) - use 0000 for start of day, 2359 for end
            date_from = f"{from_date}0000" if from_date else "*"
            date_to = f"{to_date}2359" if to_date else "*"
            # Use correct arXiv API syntax with + symbols
            search_query += f" AND submittedDate:[{date_from}+TO+{date_to}]"

        params = {
            "search_query": search_query,
            "start": start,
            "max_results": min(max_results, 2000),
            "sortBy": sort_by,
            "sortOrder": sort_order,
        }

        safe = ":+[]"  # Don't encode :, +, [, ] characters needed for arXiv queries
        url = f"{self.base_url}?{urlencode(params, quote_via=quote, safe=safe)}"

        try:
            logger.info(f"Fetching {max_results} {self.search_category} papers from arXiv")

            # Add rate limiting delay between all requests (arXiv recommends 3 seconds)
            if self._last_request_time is not None:
                time_since_last = time.time() - self._last_request_time
                if time_since_last < self.rate_limit_delay:
                    sleep_time = self.rate_limit_delay - time_since_last
                    ## Using await asyncio.sleep(sleep_time) allows the event loop to run other tasks while waiting, rather than blocking the entire thread. 
                    ## This is crucial for maintaining responsiveness in an async application, especially if multiple requests are being made concurrently 
                    ## or if the client is part of a larger system that needs to remain responsive while waiting for external API calls. 
                    ## By yielding control back to the event loop,  other operations can proceed without delay, improving the overall efficiency of the application.
                    await asyncio.sleep(sleep_time)

            self._last_request_time = time.time()

            async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
                response = await client.get(url)
                response.raise_for_status()
                xml_data = response.text

            papers = self._parse_response(xml_data)
            logger.info(f"Fetched {len(papers)} papers")

            return papers

        except httpx.TimeoutException as e:
            logger.error(f"arXiv API timeout: {e}")
            raise ArxivAPITimeoutError(f"arXiv API request timed out: {e}")
        except httpx.HTTPStatusError as e:
            logger.error(f"arXiv API HTTP error: {e}")
            raise ArxivAPIException(f"arXiv API returned error {e.response.status_code}: {e}")
        except Exception as e:
            logger.error(f"Failed to fetch papers from arXiv: {e}")
            raise ArxivAPIException(f"Unexpected error fetching papers from arXiv: {e}")

    async def fetch_papers_with_query(
        self,
        search_query: str,
        max_results: Optional[int] = None,
        start: int = 0,
        sort_by: str = "submittedDate",
        sort_order: str = "descending",
    ) -> List[ArxivPaper]:
        """Fetch papers using a raw arXiv search query string.

        Escape hatch for queries beyond what fetch_papers supports — author
        searches (au:), title keyword searches (ti:), boolean combinations,
        and so on. The query string is passed through to arXiv almost as-is
        (only URL-encoded with arXiv-specific safe characters preserved).

        Args:
            search_query: Raw arXiv query (e.g. "au:LeCun AND cat:cs.AI").
            max_results: Page size, capped at 2000.
            start: Pagination offset.
            sort_by: Sort key.
            sort_order: Sort direction.

        Returns:
            List of ArxivPaper objects matching the query.

        Raises:
            ArxivAPITimeoutError, ArxivAPIException, ArxivParseError —
            same semantics as fetch_papers.

        Examples:
            "cat:cs.AI AND submittedDate:[20240101 TO *]"  # since 2024-01-01
            "au:LeCun AND cat:cs.AI"                       # by author
            "ti:transformer AND cat:cs.AI"                 # title keyword
        """
        if max_results is None:
            max_results = self.max_results

        params = {
            "search_query": search_query,
            "start": start,
            "max_results": min(max_results, 2000),
            "sortBy": sort_by,
            "sortOrder": sort_order,
        }

        safe = ":+[]*"  # Don't encode :, +, [, ], *, characters needed for arXiv queries
        url = f"{self.base_url}?{urlencode(params, quote_via=quote, safe=safe)}"

        try:
            # Add rate limiting delay between all requests (arXiv recommends 3 seconds)
            if self._last_request_time is not None:
                time_since_last = time.time() - self._last_request_time
                if time_since_last < self.rate_limit_delay:
                    sleep_time = self.rate_limit_delay - time_since_last
                    await asyncio.sleep(sleep_time)

            self._last_request_time = time.time()

            async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
                response = await client.get(url)
                response.raise_for_status()
                xml_data = response.text

            papers = self._parse_response(xml_data)
            logger.info(f"Query returned {len(papers)} papers")

            return papers

        except httpx.TimeoutException as e:
            logger.error(f"arXiv API timeout: {e}")
            raise ArxivAPITimeoutError(f"arXiv API request timed out: {e}")
        except httpx.HTTPStatusError as e:
            logger.error(f"arXiv API HTTP error: {e}")
            raise ArxivAPIException(f"arXiv API returned error {e.response.status_code}: {e}")
        except Exception as e:
            logger.error(f"Failed to fetch papers from arXiv: {e}")
            raise ArxivAPIException(f"Unexpected error fetching papers from arXiv: {e}")

    async def fetch_paper_by_id(self, arxiv_id: str) -> Optional[ArxivPaper]:
        """Fetch a single paper by its arXiv ID.

        Uses the id_list parameter rather than a search query for direct
        lookup. Strips the version suffix (e.g. "2507.17748v1" -> "2507.17748")
        because arXiv's id_list expects the un-versioned form.

        Args:
            arxiv_id: arXiv paper ID, with or without version suffix.

        Returns:
            ArxivPaper if found, None if arXiv returned an empty result.

        Raises:
            ArxivAPITimeoutError, ArxivAPIException, ArxivParseError.
        """
        # Clean the arXiv ID (remove version if needed for search)
        clean_id = arxiv_id.split("v")[0] if "v" in arxiv_id else arxiv_id
        params = {"id_list": clean_id, "max_results": 1}

        safe = ":+[]*"  # Don't encode :, +, [, ], *, characters needed for arXiv queries
        url = f"{self.base_url}?{urlencode(params, quote_via=quote, safe=safe)}"

        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(url)
                response.raise_for_status()
                xml_data = response.text

            papers = self._parse_response(xml_data)

            if papers:
                return papers[0]
            else:
                logger.warning(f"Paper {arxiv_id} not found")
                return None

        except httpx.TimeoutException as e:
            logger.error(f"arXiv API timeout for paper {arxiv_id}: {e}")
            raise ArxivAPITimeoutError(f"arXiv API request timed out for paper {arxiv_id}: {e}")
        except httpx.HTTPStatusError as e:
            logger.error(f"arXiv API HTTP error for paper {arxiv_id}: {e}")
            raise ArxivAPIException(f"arXiv API returned error {e.response.status_code} for paper {arxiv_id}: {e}")
        except Exception as e:
            logger.error(f"Failed to fetch paper {arxiv_id} from arXiv: {e}")
            raise ArxivAPIException(f"Unexpected error fetching paper {arxiv_id} from arXiv: {e}")

    def _parse_response(self, xml_data: str) -> List[ArxivPaper]:
        """Parse a raw arXiv Atom XML response into ArxivPaper objects.

        Top-level parsing entry point shared by all three fetch methods.
        Locates every <atom:entry> in the feed and delegates per-entry
        extraction to _parse_single_entry. Entries that fail to parse are
        skipped (logged) rather than aborting the whole batch.

        Args:
            xml_data: Raw XML body from arXiv.

        Returns:
            List of successfully parsed ArxivPaper objects.

        Raises:
            ArxivParseError: XML is malformed or unparseable at the root level.
        """
        try:
            root = ET.fromstring(xml_data)
            entries = root.findall("atom:entry", self.namespaces)

            papers = []
            for entry in entries:
                paper = self._parse_single_entry(entry)
                if paper:
                    papers.append(paper)

            return papers

        except ET.ParseError as e:
            logger.error(f"Failed to parse arXiv XML response: {e}")
            raise ArxivParseError(f"Failed to parse arXiv XML response: {e}")
        except Exception as e:
            logger.error(f"Unexpected error parsing arXiv response: {e}")
            raise ArxivParseError(f"Unexpected error parsing arXiv response: {e}")

    def _parse_single_entry(self, entry: ET.Element) -> Optional[ArxivPaper]:
        """Build an ArxivPaper from a single <atom:entry> element.

        Orchestrates the field-level extractors (_get_arxiv_id, _get_text,
        _get_authors, etc.) and assembles the result. Returns None — rather
        than raising — when an individual entry is malformed, so that one
        bad record does not poison an otherwise-good batch.

        Args:
            entry: <atom:entry> XML element.

        Returns:
            ArxivPaper, or None if the entry was unusable (e.g. missing ID).
        """
        try:
            # Extract basic metadata
            arxiv_id = self._get_arxiv_id(entry)
            if not arxiv_id:
                return None

            title = self._get_text(entry, "atom:title", clean_newlines=True)
            authors = self._get_authors(entry)
            abstract = self._get_text(entry, "atom:summary", clean_newlines=True)
            published = self._get_text(entry, "atom:published")
            categories = self._get_categories(entry)
            pdf_url = self._get_pdf_url(entry)

            return ArxivPaper(
                arxiv_id=arxiv_id,
                title=title,
                authors=authors,
                abstract=abstract,
                published_date=published,
                categories=categories,
                pdf_url=pdf_url,
            )

        except Exception as e:
            logger.error(f"Failed to parse entry: {e}")
            return None

    def _get_text(self, element: ET.Element, path: str, clean_newlines: bool = False) -> str:
        """Generic safe text extractor for any namespaced XML child element.

        Returns an empty string instead of raising when the child or its
        text is missing — keeps the parser resilient to incomplete entries.
        clean_newlines collapses embedded newlines into spaces, used for
        title/abstract fields where arXiv inserts line breaks for display.

        Args:
            element: Parent XML element.
            path: Namespaced XPath (e.g. "atom:title").
            clean_newlines: If True, replace newlines with single spaces.

        Returns:
            Stripped text content, or empty string when not present.
        """
        elem = element.find(path, self.namespaces)
        if elem is None or elem.text is None:
            return ""

        text = elem.text.strip()
        return text.replace("\n", " ") if clean_newlines else text

    def _get_arxiv_id(self, entry: ET.Element) -> Optional[str]:
        """Extract the bare arXiv ID from an entry's <atom:id> element.

        The <atom:id> field contains a full URL like
        "http://arxiv.org/abs/2507.17748v1" — only the trailing segment
        after the last "/" is the actual paper ID. Returns None when the
        element is missing so _parse_single_entry can skip the entry.

        Args:
            entry: <atom:entry> XML element.

        Returns:
            The trailing path segment (with version), or None.
        """
        id_elem = entry.find("atom:id", self.namespaces)
        if id_elem is None or id_elem.text is None:
            return None
        return id_elem.text.split("/")[-1]

    def _get_authors(self, entry: ET.Element) -> List[str]:
        """Extract author display names from an entry.

        Iterates every <atom:author> child and reads each <atom:name>.
        Empty names are skipped silently so callers always get a clean list.

        Args:
            entry: <atom:entry> XML element.

        Returns:
            Ordered list of author names (may be empty).
        """
        authors = []
        for author in entry.findall("atom:author", self.namespaces):
            name = self._get_text(author, "atom:name")
            if name:
                authors.append(name)
        return authors

    def _get_categories(self, entry: ET.Element) -> List[str]:
        """Extract arXiv category tags from an entry.

        Categories live in <atom:category term="..."> elements. The term
        attribute is the canonical tag (e.g. "cs.AI"), which is what we
        store. Categories without a term are skipped.

        Args:
            entry: <atom:entry> XML element.

        Returns:
            List of category terms (e.g. ["cs.AI", "cs.LG"]).
        """
        categories = []
        for category in entry.findall("atom:category", self.namespaces):
            term = category.get("term")
            if term:
                categories.append(term)
        return categories

    def _get_pdf_url(self, entry: ET.Element) -> str:
        """Extract the PDF URL from an entry's links.

        Each entry has multiple <atom:link> elements (abstract page, DOI,
        etc.); we pick the one with type="application/pdf". arXiv sometimes
        returns http:// URLs — those are normalized to https:// so the
        downloader never falls back to plaintext HTTP.

        Args:
            entry: <atom:entry> XML element.

        Returns:
            HTTPS PDF URL, or empty string if no PDF link is present.
        """
        for link in entry.findall("atom:link", self.namespaces):
            if link.get("type") == "application/pdf":
                url = link.get("href", "")
                # Convert HTTP to HTTPS for arXiv URLs
                if url.startswith("http://arxiv.org/"):
                    url = url.replace("http://arxiv.org/", "https://arxiv.org/")
                return url
        return ""

    async def download_pdf(self, paper: ArxivPaper, force_download: bool = False) -> Optional[Path]:
        """Download a paper's PDF to the local cache.

        Public entry point for PDF retrieval. Implements a cache-first
        strategy: if the file already exists on disk and force_download is
        False, the cached path is returned without making a network call.
        Otherwise the download is delegated to _download_with_retry.

        Args:
            paper: ArxivPaper whose pdf_url will be fetched.
            force_download: If True, bypass cache and always re-download.

        Returns:
            Path to the local PDF, or None if the paper had no PDF URL or
            the download ultimately failed without raising.

        Raises:
            PDFDownloadTimeoutError, PDFDownloadException — see
            _download_with_retry for the failure conditions.
        """
        if not paper.pdf_url:
            logger.error(f"No PDF URL for paper {paper.arxiv_id}")
            return None

        pdf_path = self._get_pdf_path(paper.arxiv_id)

        # Return cached PDF if exists
        if pdf_path.exists() and not force_download:
            logger.info(f"Using cached PDF: {pdf_path.name}")
            return pdf_path

        # Download with retry
        ## await usage here allows the event loop to run other tasks while waiting for the download to complete, 
        ## rather than blocking the entire thread.
        if await self._download_with_retry(paper.pdf_url, pdf_path):
            return pdf_path
        else:
            return None

    def _get_pdf_path(self, arxiv_id: str) -> Path:
        """Compute the on-disk path for a given arXiv ID's cached PDF.

        Sanitizes the ID by replacing "/" with "_" so older-style IDs like
        "math.GT/0309136" produce safe filenames on every filesystem.

        Args:
            arxiv_id: arXiv paper ID.

        Returns:
            Absolute Path inside pdf_cache_dir.
        """
        safe_filename = arxiv_id.replace("/", "_") + ".pdf"
        return self.pdf_cache_dir / safe_filename

    async def _download_with_retry(self, url: str, path: Path, max_retries: int = 3) -> bool:
        """Stream a PDF to disk with retry-on-failure.

        Streams the body in chunks via response.aiter_bytes() so memory
        usage stays flat regardless of file size. Each retry uses linear
        backoff (5s, 10s, 15s). Timeout and HTTP errors retry; any other
        exception aborts immediately. After the final retry exhausts, the
        original error is re-raised as a typed PDFDownloadException so
        callers know the download is permanently failed (not in flight).

        Args:
            url: PDF URL to fetch.
            path: Destination file path. Partial files are removed if the
                whole sequence fails.
            max_retries: Number of attempts before giving up (default 3).

        Returns:
            True on success. (False is unreachable in practice — exhausted
            retries raise; the explicit return False is a fallthrough guard.)

        Raises:
            PDFDownloadTimeoutError: All attempts timed out.
            PDFDownloadException: All attempts failed with HTTP errors, or
                an unexpected exception was hit at any point.
        """
        logger.info(f"Downloading PDF from {url}")

        # Respect rate limits
        await asyncio.sleep(self.rate_limit_delay)

        for attempt in range(max_retries):
            try:
                async with httpx.AsyncClient(timeout=30.0) as client:
                    async with client.stream("GET", url) as response:
                        response.raise_for_status()
                        with open(path, "wb") as f:
                            async for chunk in response.aiter_bytes():
                                f.write(chunk)
                logger.info(f"Successfully downloaded to {path.name}")
                return True

            except httpx.TimeoutException as e:
                if attempt < max_retries - 1:
                    wait_time = 5 * (attempt + 1)
                    logger.warning(f"PDF download timeout (attempt {attempt + 1}/{max_retries}): {e}")
                    logger.info(f"Retrying in {wait_time}s...")
                    await asyncio.sleep(wait_time)
                else:
                    logger.error(f"PDF download failed after {max_retries} attempts due to timeout: {e}")
                    raise PDFDownloadTimeoutError(f"PDF download timed out after {max_retries} attempts: {e}")
            except httpx.HTTPError as e:
                if attempt < max_retries - 1:
                    wait_time = 5 * (attempt + 1)  # Exponential backoff
                    logger.warning(f"Download failed (attempt {attempt + 1}/{max_retries}): {e}")
                    logger.info(f"Retrying in {wait_time}s...")
                    await asyncio.sleep(wait_time)
                else:
                    logger.error(f"Failed after {max_retries} attempts: {e}")
                    raise PDFDownloadException(f"PDF download failed after {max_retries} attempts: {e}")
            except Exception as e:
                logger.error(f"Unexpected download error: {e}")
                raise PDFDownloadException(f"Unexpected error during PDF download: {e}")

        # Clean up partial download
        if path.exists():
            path.unlink()

        return False
