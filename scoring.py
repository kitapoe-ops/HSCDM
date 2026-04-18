"""
MTC Scoring Engine — 實時評分 + Scanbot 集成介面
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from typing import Dict, Optional, List

from .model import HSCDM, STATE_EMOJI
from .config import HSCDMConfig
from .nlp import NewsSentimentAnalyzer
from .api_clients import MarineTrafficClient, AISSnapshot


@dataclass
class MTCSnapshot:
    """完整快照 — 包含所有模型輸入"""
    timestamp: datetime
    # Maritime
    throughput: float
    avg_speed: float
    deviation_index: float
    anchor_count: float
    # NLP
    danger_score: float
    calming_score: float
    source_reliability: float
    # Meta
    article_count: int
    nlp_hits: int
    is_mock_maritime: bool = False
    is_mock_nlp: bool = False

    def to_dict(self) -> Dict:
        d = asdict(self)
        d['timestamp'] = self.timestamp.isoformat()
        return d


class MTCScorer:
    """
    MTC 實時評分引擎
    整合：MarineTraffic API + News NLP + HSCDM Model

    使用方式：
        scorer = MTCScorer()
        scorer.calibrate(historical_df)      # 一次性基線校準
        result = scorer.score_now()           # 每次調用 = 一次評分
        print(result['HCS'], result['State'])
    """

    def __init__(
        self,
        mt_client: Optional[MarineTrafficClient] = None,
        nlp_analyzer: Optional[NewsSentimentAnalyzer] = None,
        hscdm_config: Optional[HSCDMConfig] = None,
    ):
        self.mt = mt_client or MarineTrafficClient()
        self.nlp = nlp_analyzer or NewsSentimentAnalyzer()
        self.model = HSCDM(hscdm_config or HSCDMConfig())
        self._calibrated = False
        self._history: List[MTCSnapshot] = []

        # Rate limiting
        self._last_nlp_fetch = datetime.min
        self._nlp_fetch_interval = timedelta(minutes=10)
        self._last_mt_fetch = datetime.min
        self._mt_fetch_interval = timedelta(minutes=5)

    # ── Calibration ───────────────────────────────────────────────

    def calibrate(self, df, baseline_col='phase', baseline_val='P1') -> 'MTCScorer':
        """
        使用歷史數據校準 HSCDM 基線
        只需執行一次
        """
        self.model.run(df, baseline_col, baseline_val)
        self._calibrated = True
        print(f"✅ MTCScorer 校準完成")
        return self

    def calibrate_from_snapshots(self, snapshots: List[MTCSnapshot]) -> 'MTCScorer':
        """
        用一組快照進行校準（適用於歷史回測）
        """
        import pandas as pd

        records = []
        for s in snapshots:
            records.append({
                'throughput': s.throughput,
                'avg_speed': s.avg_speed,
                'deviation_index': s.deviation_index,
                'anchor_count': s.anchor_count,
                'danger_score': s.danger_score,
                'calming_score': s.calming_score,
                'source_reliability': s.source_reliability,
                'phase': 'P1',  # assume all baseline
            })

        df = pd.DataFrame(records)
        self.model.calibrate(df, baseline_col=None, baseline_val=None)
        self._calibrated = True
        return self

    # ── Live Scoring ──────────────────────────────────────────────

    def score_now(
        self,
        nlp_hours: int = 4,
        force_refresh: bool = False,
    ) -> Dict:
        """
        執行一次完整評分

        流程：
        1. 抓取 MarineTraffic (每5分鐘最多一次)
        2. 抓取 News NLP (每10分鐘最多一次)
        3. 計算 HSCDM 分數
        4. 返回結構化結果
        """
        now = datetime.utcnow()

        # 1. AIS snapshot
        if force_refresh or (now - self._last_mt_fetch) > self._mt_fetch_interval:
            mt_snap = self.mt.get_hormuz_snapshot()
            self._last_mt_fetch = now
            self._cached_mt = mt_snap
        else:
            mt_snap = getattr(self, '_cached_mt', self.mt.get_hormuz_snapshot())

        # 2. NLP analysis
        if force_refresh or (now - self._last_nlp_fetch) > self._nlp_fetch_interval:
            nlp_result = self.nlp.analyze(hours=nlp_hours, force_fetch=True)
            self._last_nlp_fetch = now
            self._cached_nlp = nlp_result
        else:
            nlp_result = getattr(self, '_cached_nlp', self.nlp.analyze(hours=nlp_hours))

        # 3. Build snapshot
        snap = MTCSnapshot(
            timestamp=now,
            throughput=mt_snap.throughput_1h,
            avg_speed=mt_snap.avg_speed,
            deviation_index=mt_snap.deviation_index,
            anchor_count=float(mt_snap.anchor_count),
            danger_score=nlp_result.danger_score,
            calming_score=nlp_result.calming_score,
            source_reliability=nlp_result.avg_reliability,
            article_count=nlp_result.article_count,
            nlp_hits=len(nlp_result.danger_hits),
            is_mock_maritime=self.mt.is_mock,
            is_mock_nlp=not bool(self.nlp.news_sources),
        )

        # 4. HSCDM scoring
        snapshot_dict = {
            'throughput': snap.throughput,
            'avg_speed': snap.avg_speed,
            'deviation_index': snap.deviation_index,
            'anchor_count': snap.anchor_count,
            'danger_score': snap.danger_score,
            'calming_score': snap.calming_score,
            'source_reliability': snap.source_reliability,
        }

        score_result = self.model.score_now(snapshot_dict)

        # 5. Build final result
        result = {
            **score_result,
            'timestamp': now.isoformat(),
            'article_count': snap.article_count,
            'nlp_hits': snap.nlp_hits,
            'throughput': snap.throughput,
            'avg_speed': snap.avg_speed,
            'is_mock_data': snap.is_mock_maritime or snap.is_mock_nlp,
        }

        self._history.append(snap)
        if len(self._history) > 500:
            self._history = self._history[-500:]

        return result

    # ── Formatting ────────────────────────────────────────────────

    def format_result(self, result: Dict) -> str:
        """格式化為 Telegram 友好輸出"""
        emoji = STATE_EMOJI.get(result['State'], '⚪')
        score_bar = self._make_bar(result['HCS'])

        lines = [
            f"{emoji} **MTC Score: {result['HCS']}/100**  [{result['State']}]",
            f"   Z-Score: {result['Z']}  |  Composite: {result['I_composite']:.4f}",
            f"   σ₄ₕ: {result['sigma_4h']:.6f}",
            f"   📡 海運: throughput={result['throughput']:.1f}/hr  speed={result['avg_speed']:.1f}kn",
            f"   📰 新聞: {result['article_count']} 篇  danger_hits={result['nlp_hits']}",
            f"   {result['action']}",
        ]

        if result.get('is_mock_data'):
            lines.insert(-2, f"   ⚠️ [MOCK MODE — 需要配置真實 API key]")

        return "\n".join(lines)

    @staticmethod
    def _make_bar(score: float, width: int = 20) -> str:
        """0-100 → 視覺化進度條"""
        filled = int(round(score / 100 * width))
        bar = '█' * filled + '░' * (width - filled)
        return f"[{bar}]"

    def get_history(self) -> List[Dict]:
        """返回評分歷史"""
        return [s.to_dict() for s in self._history]
