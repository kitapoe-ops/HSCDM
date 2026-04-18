"""
MTC API Clients — MarineTraffic + News Feed
"""
from __future__ import annotations

import os
import time
import requests
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from datetime import datetime, timedelta


# ─── MarineTraffic Client ─────────────────────────────────────────

@dataclass
class AISSnapshot:
    """單一海域的 AIS 快照"""
    timestamp: datetime
    total_vessels: int
    avg_speed: float          # knots
    throughput_1h: float      # 過去1小時通過船隻數
    anchor_count: int          # 拋錨等待船隻
    deviation_index: float    # 偏離海峽比例 [0,1]
    vessels: List[Dict] = field(default_factory=list)


class MarineTrafficClient:
    """
    MarineTraffic API v3 client.

    端點參考：
    GET https://services.marinetraffic.com/api/exportvessel/vessel/{vessel_mmsi}
    GET https://services.marinetraffic.com/api/exportvessel/{port_id}/timespan:{minutes}

    實際使用需要 API Key，見 MARINETRAFFIC_API_KEY 環境變數。
    模擬模式（無 key）自動開啟。
    """

    BASE_URL = "https://services.marinetraffic.com/api"

    def __init__(self, api_key: Optional[str] = None, timeout: int = 15):
        self.api_key = api_key or os.getenv('MARINETRAFFIC_API_KEY', '')
        self.timeout = timeout
        self._mock = not self.api_key

    @property
    def is_mock(self) -> bool:
        return self._mock

    def get_hormuz_snapshot(self, timespan: int = 60) -> AISSnapshot:
        """
        獲取霍爾木茲海峽當前 AIS 快照

        timespan: 歷史窗口（分鐘）
        """
        if self._mock:
            return self._mock_snapshot()

        # Real API: use vessel database endpoint
        # For Hormuz region, port_id for Fujairah (major anchorage before strait)
        # Port 331 = Fujairah, AE
        try:
            url = f"{self.BASE_URL}/exportvessel/331/timespan:{timespan}/protocol:json"
            resp = requests.get(
                url,
                params={'api_key': self.api_key},
                timeout=self.timeout,
            )
            resp.raise_for_status()
            data = resp.json()
            return self._parse_response(data)

        except requests.exceptions.RequestException as e:
            print(f"⚠️ MarineTraffic API error: {e}")
            return self._mock_snapshot()

    def get_region_vessels(self, region_id: int = 7, timespan: int = 60) -> AISSnapshot:
        """
        獲取特定區域（region_id=7 = Persian Gulf approach）的船隻數據
        """
        if self._mock:
            return self._mock_snapshot()

        try:
            url = f"{self.BASE_URL}/exportvessel/{region_id}/timespan:{timespan}/protocol:json"
            resp = requests.get(
                url,
                params={'api_key': self.api_key},
                timeout=self.timeout,
            )
            resp.raise_for_status()
            data = resp.json()
            return self._parse_response(data)

        except requests.exceptions.RequestException as e:
            print(f"⚠️ MarineTraffic API error: {e}")
            return self._mock_snapshot()

    def _parse_response(self, data: List[Dict]) -> AISSnapshot:
        """解析 MarineTraffic API 響應"""
        if not data:
            return self._mock_snapshot()

        speeds = [float(v.get('SOG', 0)) for v in data if v.get('SOG')]
        anchorage = [v for v in data if float(v.get('SOG', 99)) < 1.0]

        # Deviation: vessels not on typical through-route
        # Simple proxy: vessels with heading significantly different from strait heading (~45°)
        through_vessels = [v for v in data if 30 <= float(v.get('COG', 0)) <= 60]

        return AISSnapshot(
            timestamp=datetime.utcnow(),
            total_vessels=len(data),
            avg_speed=float(sum(speeds) / len(speeds)) if speeds else 0,
            throughput_1h=len(data) / 4,  # rough proxy
            anchor_count=len(anchorage),
            deviation_index=1 - (len(through_vessels) / max(len(data), 1)),
            vessels=data,
        )

    def _mock_snapshot(self) -> AISSnapshot:
        """模擬數據（無 API key 或 API 失敗時使用）"""
        import random
        # Simulate some realistic fluctuation
        now = datetime.utcnow()
        # Slightly correlated with time-of-day (Fujairah anchorage patterns)
        hour_factor = abs(12 - (now.hour % 24)) / 12
        base_traffic = 20 + 8 * (1 - hour_factor)

        return AISSnapshot(
            timestamp=now,
            total_vessels=int(base_traffic + random.uniform(-5, 5)),
            avg_speed=round(random.uniform(8, 14), 1),
            throughput_1h=round(random.uniform(15, 25), 1),
            anchor_count=int(random.uniform(2, 10)),
            deviation_index=round(random.uniform(0.01, 0.08), 3),
            vessels=[],
        )


# ─── News Feed Aggregator ─────────────────────────────────────────

class NewsFeedAggregator:
    """
    統一的新聞獲取介面
    支持：RSS feeds, NewsAPI, custom endpoints
    """

    def __init__(self, sources: Optional[List[str]] = None):
        self.sources = sources or []

    def fetch_all(self) -> List[Dict]:
        """從所有配置的來源抓取"""
        all_articles = []
        for source in self.sources:
            try:
                if source.endswith('.xml') or 'rss' in source.lower():
                    articles = self._fetch_rss(source)
                elif 'newsapi' in source.lower():
                    articles = self._fetch_newsapi(source)
                else:
                    articles = self._fetch_generic(source)
                all_articles.extend(articles)
            except Exception as e:
                print(f"⚠️ Source fetch failed for {source}: {e}")
        return all_articles

    def _fetch_rss(self, url: str) -> List[Dict]:
        """RSS feed → article list"""
        try:
            import feedparser
            feed = feedparser.parse(url)
            articles = []
            for entry in feed.entries[:20]:
                articles.append({
                    'title': entry.get('title', ''),
                    'summary': entry.get('summary', entry.get('description', '')),
                    'url': entry.get('link', ''),
                    'published': entry.get('published'),
                    'source': self._source_name(url),
                })
            return articles
        except Exception as e:
            print(f"⚠️ RSS error {url}: {e}")
            return []

    def _fetch_newsapi(self, url: str) -> List[Dict]:
        """NewsAPI → article list"""
        try:
            resp = requests.get(url, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            articles = []
            for item in data.get('articles', [])[:20]:
                articles.append({
                    'title': item.get('title', ''),
                    'summary': item.get('description', ''),
                    'url': item.get('url', ''),
                    'published': item.get('publishedAt'),
                    'source': item.get('source', {}).get('name', 'newsapi'),
                })
            return articles
        except Exception as e:
            print(f"⚠️ NewsAPI error: {e}")
            return []

    def _fetch_generic(self, url: str) -> List[Dict]:
        """Generic JSON endpoint"""
        try:
            resp = requests.get(url, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            if isinstance(data, list):
                return data[:20]
            return []
        except Exception:
            return []

    @staticmethod
    def _source_name(url: str) -> str:
        try:
            from urllib.parse import urlparse
            netloc = urlparse(url).netloc
            return netloc.replace('feeds.', '').replace('www.', '').split('.')[0]
        except Exception:
            return 'unknown'
