"""
MTC Synthetic Data Generator — 回測用模擬數據
"""
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional


def generate_synthetic_data(
    days: int = 45,
    freq_min: int = 15,
    seed: int = 42,
) -> pd.DataFrame:
    """
    模擬霍爾木茲海峽三階段數據

    Phase 1 (Day 1-30):   常態 — 建立基線
    Phase 2 (Day 31-38):  升級 — 逐步惡化 + 偶發封鎖
    Phase 3 (Day 39-45):  危機 — 高頻開關 + 突發事件
    """
    np.random.seed(seed)
    n = int(days * 24 * 60 / freq_min)
    idx = pd.date_range('2026-03-05', periods=n, freq=f'{freq_min}min')
    df = pd.DataFrame(index=idx)

    p1 = int(30 * 96)   # Day 30 end index
    p2 = int(38 * 96)   # Day 38 end index

    # Initialize columns
    for col in ['throughput', 'avg_speed', 'deviation_index', 'anchor_count',
                'news_count', 'danger_score', 'calming_score', 'source_reliability']:
        df[col] = 0.0

    r = np.random.random
    rn = lambda m=0, s=1: np.random.normal(m, s)
    re = lambda lam: np.random.exponential(lam)
    po = lambda lam: np.random.poisson(lam)

    for i in range(n):
        if i < p1:
            # ── Phase 1: Normal ──
            df.iloc[i] = [
                22 + rn(0, 2.5),       # throughput
                12.5 + rn(0, 1.0),     # avg_speed
                re(0.02),               # deviation
                max(0, po(2)),          # anchor_count
                po(8),                  # news_count
                re(0.5),                # danger_score
                re(1.2),                # calming_score
                0.6 + r() * 0.35,      # source_reliability
            ]

        elif i < p2:
            # ── Phase 2: Escalation ──
            p = (i - p1) / (p2 - p1)  # 0→1

            df.iloc[i] = [
                22 * (1 - p * 0.6) + rn(0, 3 + p * 5),
                12.5 * (1 - p * 0.3) + rn(0, 1.5),
                0.02 + p * 0.15 + re(0.03),
                max(0, 2 + p * 15 + po(3)),
                max(0, 8 + p * 25 + po(5)),
                max(0, 0.5 + p * 4 + re(0.8)),
                max(0.1, 1.2 - p * 0.7 + re(0.5)),
                0.5 + r() * 0.48,
            ]

            # Occasional brief blockade (~10% every 3 days, 1-3h duration)
            if int(i / 96) % 3 == 0 and r() < 0.1:
                burst = np.random.randint(4, 12)
                for j in range(i, min(i + burst, p2)):
                    df.iloc[j] = [
                        r() * 3, r() * 2, 0.3 + r() * 0.4,
                        max(0, 20 + r() * 20),
                        max(0, po(30)), re(3), re(0.3), 0.4 + r() * 0.3,
                    ]
        else:
            # ── Phase 3: Crisis (high-frequency oscillation) ──
            t = i - p2
            cyc = np.sin(t / 8)           # ~2h oscillation
            vs = 1 + 2 * abs(np.sin(t / 6))  # news volatility spike

            df.iloc[i] = [
                max(0, 22 * 0.15 + 22 * 0.12 * max(0, cyc) + rn(0, 4)),
                max(0, 12.5 * 0.2 + 12.5 * 0.3 * max(0, cyc) + rn(0, 2)),
                min(1.0, 0.4 + 0.3 * abs(cyc) + re(0.05)),
                max(0, 25 + 15 * abs(cyc) + po(5)),
                max(0, 33 * vs + po(10)),
                max(0, 4.5 + re(2) * vs),
                max(0.1, 0.5 + re(1.5) * (1 + max(0, np.sin(t / 8)))),
                0.4 + r() * 0.59,
            ]

            # Random firing events (5%, 30min-2hr full closure)
            if r() < 0.05:
                burst = np.random.randint(2, 8)
                for j in range(i, min(i + burst, n)):
                    df.iloc[j] = [
                        0, 0, min(1.0, 0.7 + r() * 0.3),
                        max(0, 35 + r() * 15),
                        max(0, po(50)), max(0, re(5)), max(0, re(0.5)),
                        0.3 + r() * 0.4,
                    ]

    df['deviation_index'] = df['deviation_index'].clip(0, 1)
    df['throughput'] = df['throughput'].clip(0, 50)
    df['anchor_count'] = df['anchor_count'].clip(0, 100)

    df['phase'] = np.where(
        df.index < df.index[p1], 'P1',
        np.where(df.index < df.index[p2], 'P2', 'P3')
    )

    return df
