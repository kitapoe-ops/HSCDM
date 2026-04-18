"""
MTC Config — 所有超參數集中喺呢度
"""
from dataclasses import dataclass


@dataclass
class HSCDMConfig:
    # ── Composite Index 權重 ──
    w_maritime: float = 0.55   # 海運分量
    w_nlp: float = 0.45        # NLP 情緒分量

    # ── Maritime sub-weights: M(t) = Σ αᵢ · featureᵢ ──
    alpha_throughput: float = 0.30   # 通過量（反轉：越低越危險）
    alpha_speed: float = 0.25         # 航速（反轉）
    alpha_deviation: float = 0.25     # 偏離海峽比例
    alpha_anchor: float = 0.20        # 拋錨等待船數

    # ── NLP sub-weights: S(t) = Σ βⱼ · featureⱼ ──
    beta_danger: float = 0.45         # 危險關鍵詞頻率
    beta_calming: float = 0.30        # 緩和關鍵詞（反向）
    beta_reliability: float = 0.25     # 信源不可靠度

    # ── Volatility Engine ──
    vol_window: int = 16               # 滾動窗口：16 × 15min = 4h
    z_trigger: float = 3.0            # Z-Score 觸發閾值

    # ── Scoring sigmoid ──
    sigmoid_k: float = 0.8            # Sigmoid 陡度
    sigmoid_midpoint: float = 2.0     # Sigmoid 中心點

    # ── State machine thresholds ──
    z_green_yellow: float = 1.0
    z_yellow_orange: float = 2.0
    z_orange_red: float = 3.0


@dataclass
class MTCConfig:
    """MTC 專案層級設定"""
    project_name: str = "mtc"
    version: str = "1.0.0"

    # API keys (可喺環境變數讀取)
    marinetraffic_api_key: str = ""    # MARINETRAFFIC_API_KEY
    news_api_key: str = ""             # NEWS_API_KEY

    # 採樣頻率 (分鐘)
    sample_interval_min: int = 15

    # NLP 關鍵詞
    danger_keywords: tuple = (
        "hormuz", "strait", "closed", "attacked", "attacked",
        "tension", "military", "warship", "navy", "iran",
        "missile", "drone", "seized", "detained", "conflict",
        "blockade", "incidents", "warning",
    )
    calming_keywords: tuple = (
        "reopened", "de-escalation", "talks", "agreement",
        "peace", "normalized", "cleared", "open",
    )

    # News feed sources (RSS / API endpoints)
    news_sources: tuple = (
        "https://feeds.bbci.co.uk/news/world/middle_east/rss.xml",
        "https://feeds.reuters.com/reuters/businessNews",
        "https://feeds.reuters.com/reuters/worldNews",
    )

    # 評分分數級別
    score_normal: tuple = (0, 20)
    score_attention: tuple = (21, 40)
    score_warning: tuple = (41, 60)
    score_crisis: tuple = (61, 80)
    score_emergency: tuple = (81, 100)
