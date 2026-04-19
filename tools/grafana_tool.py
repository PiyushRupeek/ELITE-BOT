"""
Grafana Loki log fetcher.

Queries Loki via the Grafana HTTP API (no MCP server needed).
The Grafana MCP server (grafana/mcp-grafana) can be used as an
alternative — see README for setup instructions.
"""
import time
import httpx
import config


class GrafanaTool:
    def __init__(self):
        self.base_url = config.GRAFANA_URL.rstrip("/")
        self.api_key = config.GRAFANA_API_KEY
        self.loki_uid = config.GRAFANA_LOKI_UID
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def _is_configured(self) -> bool:
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
        """
        if not self._is_configured():
            return ""

        now_ns = int(time.time() * 1e9)
        start_ns = now_ns - duration_minutes * 60 * int(1e9)

        # LogQL query - adjust label selector to match your Loki setup
        query = f'{{service="{service_name}"}} | logfmt'

        try:
            resp = httpx.get(
                f"{self.base_url}/api/datasources/proxy/uid/{self.loki_uid}/loki/api/v1/query_range",
                headers=self.headers,
                params={
                    "query": query,
                    "start": str(start_ns),
                    "end": str(now_ns),
                    "limit": limit,
                    "direction": "backward",
                },
                timeout=15,
            )
            resp.raise_for_status()
            data = resp.json()

            lines = []
            for stream in data.get("data", {}).get("result", []):
                for ts, log_line in stream.get("values", []):
                    lines.append(log_line)

            if not lines:
                return f"No logs found for service '{service_name}' in the last {duration_minutes} minutes."

            return "\n".join(self._filter_logs(lines)[:limit])

        except httpx.HTTPStatusError as e:
            return f"Grafana error ({e.response.status_code}): {e.response.text[:200]}"
        except Exception as e:
            return f"Could not fetch logs: {e}"

    def _filter_logs(self, lines: list[str]) -> list[str]:
        """Keep only ERROR/WARN/EXCEPTION lines. Falls back to all lines if none match."""
        keywords = {"error", "exception", "fatal", "warn", "failed", "traceback", "caused by"}
        filtered = [l for l in lines if any(kw in l.lower() for kw in keywords)]
        return filtered if filtered else lines

    def search_dashboards(self, query: str) -> list[dict]:
        """Find dashboards by keyword."""
        if not self._is_configured():
            return []
        try:
            resp = httpx.get(
                f"{self.base_url}/api/search",
                headers=self.headers,
                params={"query": query, "type": "dash-db"},
                timeout=10,
            )
            resp.raise_for_status()
            return resp.json()
        except Exception:
            return []
