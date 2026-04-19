"""
Grafana Loki log fetcher.

Queries Loki via the Grafana HTTP API (no MCP server needed).
The Grafana MCP server (grafana/mcp-grafana) can be used as an
alternative — see README for setup instructions.

How it works:
  1. Build a time range: [now - duration_minutes, now] in nanoseconds (Loki's required format)
  2. Send a LogQL query to Loki via the Grafana datasource proxy endpoint
  3. Filter the returned lines to keep only ERROR/WARN/EXCEPTION entries

Why Loki instead of direct log files?
  - Loki aggregates logs from all pod replicas in one place (k8s environment)
  - Supports label-based filtering — easy to scope to a single service
  - No SSH or kubectl access needed

Graceful degradation:
  If GRAFANA_ENABLED=false or credentials are missing, all methods return
  empty/default values — the bot works normally without logs.
"""
import time
import httpx
import config


class GrafanaTool:
    def __init__(self):
        # Strip trailing slash so we can safely append paths with a leading slash
        self.base_url = config.GRAFANA_URL.rstrip("/")
        self.api_key = config.GRAFANA_API_KEY
        # Loki datasource UID — found in Grafana → Connections → Data sources → Loki → UID field
        self.loki_uid = config.GRAFANA_LOKI_UID
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",  # Grafana service account token
            "Content-Type": "application/json",
        }

    def _is_configured(self) -> bool:
        """
        Check whether Grafana integration is active and fully configured.

        Returns True only if:
          - GRAFANA_ENABLED=true in .env
          - GRAFANA_API_KEY is set (non-empty)
          - GRAFANA_LOKI_UID is set (non-empty)

        This is checked at the start of every method so callers never need
        to guard against Grafana being disabled — they just get back empty results.
        """
        return config.GRAFANA_ENABLED and bool(self.api_key and self.loki_uid)

    def fetch_logs(
        self,
        service_name: str,
        duration_minutes: int = 15,
        limit: int = 50,
    ) -> str:
        """
        Fetch recent Loki logs for a service.
        Returns formatted log lines or an empty string if Grafana is not configured.

        How the time range works:
          Loki uses Unix timestamps in nanoseconds (not seconds).
          now_ns  = current time in nanoseconds
          start_ns = now_ns minus the requested duration in nanoseconds

        LogQL query format:
          {service="name"} | logfmt
          - {service="name"} : selects all log streams tagged with this service label
          - | logfmt          : parses log lines as logfmt key=value pairs (structured logging)

        Args:
            service_name     : Loki service label value (e.g. "lead-service", "rupeek-bazaar")
            duration_minutes : how far back to look (default: last 15 minutes)
            limit            : max number of log lines to return (default: 50)

        Returns:
            Newline-joined log lines (filtered to ERROR/WARN only), or empty string.
        """
        if not self._is_configured():
            return ""

        # Loki requires nanosecond timestamps — multiply Unix seconds by 1e9
        now_ns = int(time.time() * 1e9)
        start_ns = now_ns - duration_minutes * 60 * int(1e9)

        # LogQL query - adjust label selector to match your Loki setup
        query = f'{{service="{service_name}"}} | logfmt'

        try:
            # Grafana proxies Loki queries through its datasource proxy endpoint
            # URL format: /api/datasources/proxy/uid/{loki_uid}/loki/api/v1/query_range
            resp = httpx.get(
                f"{self.base_url}/api/datasources/proxy/uid/{self.loki_uid}/loki/api/v1/query_range",
                headers=self.headers,
                params={
                    "query": query,
                    "start": str(start_ns),
                    "end": str(now_ns),
                    "limit": limit,
                    "direction": "backward",  # newest logs first
                },
                timeout=15,
            )
            resp.raise_for_status()
            data = resp.json()

            # Loki response structure:
            # { "data": { "result": [ { "stream": {...}, "values": [ [timestamp_ns, log_line], ... ] } ] } }
            lines = []
            for stream in data.get("data", {}).get("result", []):
                for ts, log_line in stream.get("values", []):
                    lines.append(log_line)

            if not lines:
                return f"No logs found for service '{service_name}' in the last {duration_minutes} minutes."

            # Filter to keep only error-level lines (falls back to all if none match)
            return "\n".join(self._filter_logs(lines)[:limit])

        except httpx.HTTPStatusError as e:
            return f"Grafana error ({e.response.status_code}): {e.response.text[:200]}"
        except Exception as e:
            return f"Could not fetch logs: {e}"

    def _filter_logs(self, lines: list[str]) -> list[str]:
        """
        Keep only ERROR/WARN/EXCEPTION lines to reduce noise.

        Filters by checking if any error-indicating keyword appears anywhere
        in the lowercased log line. Falls back to returning all lines if none
        match — so we always return something useful rather than nothing.

        Keywords checked: error, exception, fatal, warn, failed, traceback, caused by
        """
        keywords = {"error", "exception", "fatal", "warn", "failed", "traceback", "caused by"}
        filtered = [l for l in lines if any(kw in l.lower() for kw in keywords)]
        # Fall back to all lines if none are error-level (e.g. service has only INFO logs)
        return filtered if filtered else lines

    def search_dashboards(self, query: str) -> list[dict]:
        """
        Find Grafana dashboards by keyword.

        Uses the Grafana Search API to look up dashboards by name or tag.
        Returns an empty list if Grafana is not configured or if the search fails.

        Args:
            query : search string (dashboard name, tag, or keyword)

        Returns:
            List of dashboard metadata dicts (uid, title, url, etc.) or []
        """
        if not self._is_configured():
            return []
        try:
            resp = httpx.get(
                f"{self.base_url}/api/search",
                headers=self.headers,
                params={"query": query, "type": "dash-db"},  # dash-db = standard dashboards
                timeout=10,
            )
            resp.raise_for_status()
            return resp.json()
        except Exception:
            return []
