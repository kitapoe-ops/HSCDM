"""
MTC NLP — News Sentiment Analyzer
關鍵詞驅動嘅海峽相關新聞情緒分析器
"""
from __future__ import annotations

import re
import hashlib
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from datetime import datetime, timedelta
from collections import defaultdict
import feedparser
import requests

try:
    from newspaper import Article
    HAS_NEWSPAPER = True
except ImportError:
    HAS_NEWSPAPER = False


@dataclass
class SentimentResult:
    danger_score: float      # 危險關鍵詞加權分數
    calming_score: float     # 緩和關鍵詞加權分數
    danger_hits: List[str]  # 命中的危險關鍵詞
    calming_hits: List[str] # 命中的緩和關鍵詞
    article_count: int       # 總文章數
    avg_reliability: float   # 平均信源可靠度
    source_breakdown: Dict[str, int]  # 各來源文章數
    time_window: str         # 時間窗口描述


class NewsSentimentAnalyzer:
    """
    NLP Sentiment Engine for Hormuz Strait crisis detection.

    兩層分析：
    1. Keyword Frequency — 計算危險/緩和關鍵詞出現頻率
    2. Source Reliability Weighting — 按來源可信度加權

    輸出：
        danger_score, calming_score, source_reliability
    """

    # 來源可信度基線（可擴展）
    SOURCE_RELIABILITY = {
        'bbc': 0.95,
        'reuters': 0.92,
        'ap': 0.90,
        'afp': 0.88,
        'nyt': 0.85,
        'ft': 0.85,
        'wsj': 0.83,
        'bloomberg': 0.82,
        'guardian': 0.78,
        'aljazeera': 0.75,
        'rt': 0.60,
        'sputnik': 0.55,
        'tweet': 0.40,
        'unknown': 0.50,
    }

    # 危險關鍵詞 + 權重（越高越危險）
    DANGER_TERMS = {
        # 封鎖/軍事衝突
        'hormuz': 2.0, 'strait of hormuz': 2.5,
        'closed': 3.0, 'blockade': 3.5,
        'attacked': 3.5, 'attack': 2.5,
        'attacked': 3.5, 'fired': 3.0,
        'missile': 3.0, 'ballistic missile': 4.0,
        'drone': 2.5, 'uav': 2.5,
        'warship': 2.0, 'navy': 1.5, 'warships': 2.0,
        'military': 1.5, 'iran': 1.0, 'iranian': 1.0,
        'revolutionary guard': 2.5, 'irgc': 2.5,
        'seized': 3.5, 'detained': 3.0,
        'tension': 2.0, 'escalation': 3.0,
        'conflict': 3.0, 'war': 3.5,
        'iran navy': 2.0, 'us navy': 2.0,
        'oil tanker': 2.5, 'tanker': 2.0,
        'sanctions': 1.5, 'sanction': 1.0,
        'threat': 2.0, 'warning': 1.5,
        'emergency': 2.0, 'crisis': 2.5,
        'shots': 3.0, 'firing': 3.0,
        'intercepted': 2.5, 'seized': 3.5,
        # 航行警告
        '航行警告': 3.0, '禁航': 3.5,
        '海峽封鎖': 4.0, '海峽緊張': 2.5,
        '軍事演習': 1.5, '實彈射擊': 3.0,
        '伊朗革命衛隊': 2.5,
    }

    # 緩和關鍵詞 + 權重（越高越安全）
    CALMING_TERMS = {
        'reopened': 3.0, 'reopen': 3.0,
        'de-escalation': 3.5, 'deescalation': 3.5,
        'talks': 2.0, 'negotiation': 2.5,
        'agreement': 3.0, 'deal': 2.5,
        'peace': 3.0, 'peaceful': 2.5,
        'normalized': 2.5, 'normalise': 2.5,
        'cleared': 2.5, 'clear': 2.0,
        'open': 1.5, 'safe': 2.0,
        'resume': 2.0, 'resumed': 2.0,
        'cooperation': 2.0, 'cooperate': 2.0,
        'diplomatic': 1.5, 'diplomacy': 2.0,
        'eased': 2.5, 'easing': 2.5,
        '降級': 3.0, '恢復': 2.5,
        '談判': 2.0, '和解': 3.0,
        '開放': 1.5, '安全': 2.0,
    }

    def __init__(
        self,
        danger_keywords: Optional[tuple] = None,
        calming_keywords: Optional[tuple] = None,
        news_sources: Optional[tuple] = None,
    ):
        self.news_sources = list(news_sources) if news_sources else []

        # Build custom keyword dicts from tuples if provided
        self.danger_kw = dict(self.DANGER_TERMS)
        self.calming_kw = dict(self.CALMING_TERMS)

        if danger_keywords:
            for kw in danger_keywords:
                if kw.lower() not in self.danger_kw:
                    self.danger_kw[kw.lower()] = 2.0

        if calming_keywords:
            for kw in calming_keywords:
                if kw.lower() not in self.calming_kw:
                    self.calming_kw[kw.lower()] = 2.0

        # In-memory article cache
        self._cache: Dict[str, dict] = {}
        self._last_fetch: Optional[datetime] = None
        self._fetch_interval = timedelta(minutes=15)

    # ── Core NLP ─────────────────────────────────────────────────

    def score_text(self, text: str, source_url: str = '') -> Dict:
        """
        對單篇文章進行情緒分析
        返回: {danger_score, calming_score, danger_hits, calming_hits, reliability}
        """
        text_lower = text.lower()

        # Detect source reliability
        reliability = self._get_source_reliability(source_url)

        # Count danger keywords
        danger_hits = []
        danger_total = 0.0
        for term, weight in self.danger_kw.items():
            count = len(re.findall(r'\b' + re.escape(term) + r'\b', text_lower))
            if count > 0:
                danger_hits.append(f"{term}×{count}")
                danger_total += count * weight

        # Count calming keywords
        calming_hits = []
        calming_total = 0.0
        for term, weight in self.calming_kw.items():
            count = len(re.findall(r'\b' + re.escape(term) + r'\b', text_lower))
            if count > 0:
                calming_hits.append(f"{term}×{count}")
                calming_total += count * weight

        return {
            'danger_score': danger_total,
            'calming_score': calming_total,
            'danger_hits': danger_hits,
            'calming_hits': calming_hits,
            'reliability': reliability,
        }

    def score_batch(self, texts: List[Dict]) -> SentimentResult:
        """
        對一批新聞進行批量分析

        texts: List[{
            'title': str,
            'summary': str,
            'url': str,
            'published': datetime,
            'source': str,
        }]
        """
        if not texts:
            return SentimentResult(
                danger_score=0, calming_score=0,
                danger_hits=[], calming_hits=[],
                article_count=0, avg_reliability=0.5,
                source_breakdown={}, time_window='N/A',
            )

        total_danger = 0.0
        total_calming = 0.0
        all_danger_hits: List[str] = []
        all_calming_hits: List[str] = []
        total_reliability = 0.0
        source_counts: Dict[str, int] = defaultdict(int)

        for item in texts:
            text = f"{item.get('title', '')} {item.get('summary', '')}"
            url = item.get('url', '')
            source = item.get('source', 'unknown')

            result = self.score_text(text, url)
            reliability = result['reliability']

            total_danger += result['danger_score'] * reliability
            total_calming += result['calming_score'] * reliability
            all_danger_hits.extend(result['danger_hits'])
            all_calming_hits.extend(result['calming_hits'])
            total_reliability += reliability
            source_counts[source] += 1

        n = len(texts)
        avg_rel = total_reliability / n if n > 0 else 0.5

        # Normalize scores (scale to ~0-10 range)
        norm_danger = min(total_danger / max(n, 1), 15.0)
        norm_calming = min(total_calming / max(n, 1), 15.0)

        times = [t['published'] for t in texts if t.get('published')]
        if times:
            t_min, t_max = min(times), max(times)
            window_desc = f"{t_min.strftime('%m-%d %H:%M')} ~ {t_max.strftime('%m-%d %H:%M')}"
        else:
            window_desc = 'N/A'

        return SentimentResult(
            danger_score=norm_danger,
            calming_score=norm_calming,
            danger_hits=all_danger_hits[:50],
            calming_hits=all_calming_hits[:50],
            article_count=n,
            avg_reliability=avg_rel,
            source_breakdown=dict(source_counts),
            time_window=window_desc,
        )

    # ── News Fetching ──────────────────────────────────────────────

    def fetch_news(self, hours: int = 4, force: bool = False) -> List[Dict]:
        """
        從配置的 RSS 源抓取最近 N 小時的新聞

        返回: List[{
            'title', 'summary', 'url', 'published', 'source'
        }]
        """
        now = datetime.utcnow()
        if not force and self._last_fetch:
            if now - self._last_fetch < self._fetch_interval:
                return list(self._cache.values())

        articles = []
        seen_hashes = set()

        for feed_url in self.news_sources:
            try:
                feed = feedparser.parse(feed_url)
                source_name = self._extract_source_name(feed_url)

                for entry in feed.entries[:30]:
                    # Dedupe by title hash
                    h = hashlib.md5(entry.get('title', '').encode()).hexdigest()
                    if h in seen_hashes:
                        continue
                    seen_hashes.add(h)

                    published = self._parse_date(entry.get('published') or entry.get('updated'))
                    if published and (now - published).total_seconds() > hours * 3600:
                        continue

                    article = {
                        'title': entry.get('title', ''),
                        'summary': entry.get('summary', entry.get('description', '')),
                        'url': entry.get('link', ''),
                        'published': published,
                        'source': source_name,
                    }
                    articles.append(article)
                    self._cache[h] = article

            except Exception as e:
                print(f"⚠️ RSS fetch failed for {feed_url}: {e}")
                continue

        self._last_fetch = now
        return articles

    # ── Full pipeline ──────────────────────────────────────────────

    def analyze(self, hours: int = 4, force_fetch: bool = False) -> SentimentResult:
        """
        完整 pipeline: fetch → score → aggregate
        """
        articles = self.fetch_news(hours=hours, force=force_fetch)
        return self.score_batch(articles)

    # ── Helpers ───────────────────────────────────────────────────

    def _get_source_reliability(self, url: str) -> float:
        """根據 URL 估算信源可靠度"""
        if not url:
            return 0.5
        url_lower = url.lower()
        for src, rel in self.SOURCE_RELIABILITY.items():
            if src in url_lower:
                return rel
        return 0.5

    def _extract_source_name(self, url: str) -> str:
        """從 URL 提取來源名"""
        import urllib.parse
        try:
            netloc = urllib.parse.urlparse(url).netloc
            parts = netloc.replace('feeds.', '').replace('www.', '').split('.')
            return parts[0] if parts else 'unknown'
        except Exception:
            return 'unknown'

    def _parse_date(self, date_str: Optional[str]) -> Optional[datetime]:
        """嘗試解析多種日期格式"""
        if not date_str:
            return None
        from email.utils import parsedate_to_datetime
        try:
            return parsedate_to_datetime(date_str).replace(tzinfo=None)
        except Exception:
            try:
                from dateutil import parser
                return parser.parse(date_str)
            except Exception:
                return None
