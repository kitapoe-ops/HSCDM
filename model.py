"""
HSCDM — Hormuz Strait Crisis Detection Model v1.0
Core math model for maritime + NLP composite scoring
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional, Dict, Tuple
import warnings
warnings.filterwarnings('ignore')


# ─── Normalization helpers ────────────────────────────────────────

def _minmax(s: pd.Series) -> pd.Series:
    mn, mx = s.min(), s.max()
    if mx - mn < 1e-10:
        return pd.Series(0.5, index=s.index)
    return (s - mn) / (mx - mn)


def _minmax_scalar(v: float, mn: float, mx: float) -> float:
    if mx - mn < 1e-10:
        return 0.5
    return (v - mn) / (mx - mn)


# ─── State labels ────────────────────────────────────────────────

STATE_EMOJI = {
    'GREEN':  '🟢',
    'YELLOW': '🟡',
    'ORANGE': '🟠',
    'RED':    '🔴',
}

STATE_ORDER = ['GREEN', 'YELLOW', 'ORANGE', 'RED']


# ─── Core Model ───────────────────────────────────────────────────

class HSCDM:
    """
    ═══════════════════════════════════════════════════════════
    Hormuz Strait Crisis Detection Model

    Layer 1 — Feature Engineering:
        M(t) = α₁(1-TP_norm) + α₂(1-SP_norm) + α₃·DEV + α₄·ANC_norm
        S(t) = β₁·DG_norm - β₂·CL_norm + β₃·(1-SR)

    Layer 2 — Composite Index:
        I(t) = wₘ·M(t) + wₛ·S(t)   ∈ [0, 1]

    Layer 3 — Volatility Anomaly:
        Z(t) = (σ₄ₕ(t) - μ₃₀d) / σ₃₀d

    Layer 4 — State Machine:
        GREEN  → Z < 1.0
        YELLOW → 1.0 ≤ Z < 2.0
        ORANGE → 2.0 ≤ Z < 3.0
        RED    → Z ≥ 3.0

    Layer 5 — HCS Score (0-100):
        Base      = sigmoid(Z; k, mid) × 50
        Adj       = I_norm × 30 × (1 + 0.5·max(Z,0))
        Momentum  = ΔI_norm × 20
        HCS       = clip(Base + Adj + Momentum, 0, 100)
    ═══════════════════════════════════════════════════════════
    """

    def __init__(self, config: Optional[object] = None):
        self.cfg = config
        self.baseline_mu: Optional[float] = None
        self.baseline_sigma: Optional[float] = None
        self.p1_I_mean: Optional[float] = None
        self.p1_I_std: Optional[float] = None
        self._calibrated: bool = False
        self._result: Optional[pd.DataFrame] = None
        self._I_buffer: list = []
        self._ref_buffer: list = []
        # P1 feature distributions for global normalization (used in score_now)
        self._p1_stats: Dict[str, Tuple[float, float]] = {}  # col -> (min, max)

    # ── Layer 1 ───────────────────────────────────────────────────

    def compute_maritime(self, df: pd.DataFrame, global_norm: bool = False) -> pd.Series:
        """M(t) — 海運危險分量 (HIGH = DANGER)

        global_norm=True: normalize against P1 feature distributions (for single-point scoring)
        global_norm=False: normalize against input DataFrame (for batch run)
        """
        c = self.cfg
        if global_norm and self._p1_stats:
            # P1 global normalization for single-point scoring
            tp_min, tp_max = self._p1_stats.get('throughput', (0, 1))
            sp_min, sp_max = self._p1_stats.get('avg_speed', (0, 1))
            an_min, an_max = self._p1_stats.get('anchor_count', (0, 1))

            tp_n = _minmax_scalar(float(df['throughput'].iloc[0]), tp_min, tp_max)
            sp_n = _minmax_scalar(float(df['avg_speed'].iloc[0]), sp_min, sp_max)
            dev = float(df['deviation_index'].iloc[0])  # already 0-1, no norm needed
            an_n = _minmax_scalar(float(df['anchor_count'].iloc[0]), an_min, an_max)
            return pd.Series([
                c.alpha_throughput * (1 - tp_n)
                + c.alpha_speed * (1 - sp_n)
                + c.alpha_deviation * dev
                + c.alpha_anchor * an_n
            ], index=df.index)
        else:
            return (
                c.alpha_throughput * (1 - _minmax(df['throughput']))
                + c.alpha_speed * (1 - _minmax(df['avg_speed']))
                + c.alpha_deviation * df['deviation_index']
                + c.alpha_anchor * _minmax(df['anchor_count'])
            )

    def compute_nlp(self, df: pd.DataFrame, global_norm: bool = False) -> pd.Series:
        """S(t) — NLP 情緒危險分量 (HIGH = DANGER)"""
        c = self.cfg
        if global_norm and self._p1_stats:
            dg_min, dg_max = self._p1_stats.get('danger_score', (0, 1))
            cl_min, cl_max = self._p1_stats.get('calming_score', (0, 1))
            sr_min, sr_max = self._p1_stats.get('source_reliability', (0, 1))

            dg_n = _minmax_scalar(float(df['danger_score'].iloc[0]), dg_min, dg_max)
            cl_n = _minmax_scalar(float(df['calming_score'].iloc[0]), cl_min, cl_max)
            sr_n = float(df['source_reliability'].iloc[0])  # already 0-1
            return pd.Series([
                c.beta_danger * dg_n
                - c.beta_calming * cl_n
                + c.beta_reliability * (1 - sr_n)
            ], index=df.index)
        else:
            return (
                c.beta_danger * _minmax(df['danger_score'])
                - c.beta_calming * _minmax(df['calming_score'])
                + c.beta_reliability * (1 - df['source_reliability'])
            )

    # ── Layer 2 ───────────────────────────────────────────────────

    def compute_composite(self, df: pd.DataFrame) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """I(t) = wₘ·M(t) + wₛ·S(t)"""
        M = self.compute_maritime(df)
        S = self.compute_nlp(df)
        I = self.cfg.w_maritime * M + self.cfg.w_nlp * S
        return I, M, S

    # ── Layer 3 ───────────────────────────────────────────────────

    def calibrate(self, df: pd.DataFrame, baseline_col: Optional[str] = 'phase',
                  baseline_val: str = 'P1') -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        用常態期數據建立波動率基線 + I 分佈基線
        """
        I, M, S = self.compute_composite(df)

        if baseline_col and baseline_col in df.columns:
            mask = df[baseline_col] == baseline_val
            baseline_I = I[mask]
        else:
            baseline_I = I
            mask = pd.Series(True, index=I.index)

        # Rolling volatility baseline
        rolling_vol = baseline_I.rolling(self.cfg.vol_window).std().dropna()
        self.baseline_mu = float(rolling_vol.mean())
        self.baseline_sigma = float(rolling_vol.std())

        # I distribution baseline (for level Z)
        self.p1_I_mean = float(baseline_I.mean())
        self.p1_I_std = float(baseline_I.std())

        # P1 feature distributions for global normalization
        # Ensures single-point snapshots are normalized against P1, not themselves
        for col in ['throughput', 'avg_speed', 'anchor_count', 'danger_score', 'calming_score', 'source_reliability']:
            if col in df.columns:
                p1_col = df.loc[mask, col]
                self._p1_stats[col] = (float(p1_col.min()), float(p1_col.max()))

        self._calibrated = True

        # Populate ref_buffer with 64 P1 I values
        p1_I = baseline_I.reset_index(drop=True)
        if len(p1_I) >= 64:
            self._ref_buffer = list(p1_I.tail(64))
        elif len(p1_I) >= self.cfg.vol_window:
            reps = (64 // len(p1_I)) + 2
            self._ref_buffer = (list(p1_I) * reps)[:64]
        else:
            self._ref_buffer = list(p1_I) * (64 // max(len(p1_I), 1) + 1)

        self._I_buffer = list(self._ref_buffer)

        trigger_abs = self.baseline_mu + self.cfg.z_trigger * self.baseline_sigma

        print("+==========================================+")
        print("|  HSCDM v1.0 -- Baseline Calibration      |")
        print("+==========================================+")
        print(f"|  mu_baseline    = {self.baseline_mu:.6f}         |")
        print(f"|  sigma_baseline = {self.baseline_sigma:.6f}         |")
        print(f"|  Z >= {self.cfg.z_trigger:.0f} trigger  = {trigger_abs:.6f}         |")
        print(f"|  samples        = {len(rolling_vol):>6d}               |")
        print(f"|  I_mean(P1)     = {self.p1_I_mean:.6f}         |")
        print(f"|  I_std(P1)      = {self.p1_I_std:.6f}         |")
        print("+==========================================+\n")

        return I, M, S

    def compute_zscore(self, I: pd.Series) -> Tuple[pd.Series, pd.Series]:
        """Z(t) = (σ₄ₕ(t) - μ₃₀d) / σ₃₀d (for time series, not single-point)"""
        if not self._calibrated:
            raise RuntimeError("⚠️ 請先執行 calibrate()")
        sigma_4h = I.rolling(self.cfg.vol_window).std()
        Z = (sigma_4h - self.baseline_mu) / self.baseline_sigma
        return sigma_4h, Z

    # ── Layer 4 ───────────────────────────────────────────────────

    def classify_state(self, Z: pd.Series) -> pd.Series:
        """State machine: GREEN / YELLOW / ORANGE / RED"""
        c = self.cfg
        states = pd.Series('GREEN', index=Z.index)
        states[Z >= c.z_green_yellow] = 'YELLOW'
        states[Z >= c.z_yellow_orange] = 'ORANGE'
        states[Z >= c.z_orange_red] = 'RED'
        return states

    # ── Layer 5 ───────────────────────────────────────────────────

    def compute_hcs(self, I: pd.Series, Z: pd.Series) -> pd.Series:
        """Hormuz Crisis Score (HCS): 0 → 100"""
        k = self.cfg.sigmoid_k
        mid = self.cfg.sigmoid_midpoint
        Base = (1 / (1 + np.exp(-k * (Z - mid)))) * 50
        I_norm = _minmax(I)
        Z_clipped = Z.clip(lower=0)
        Adjustment = I_norm * 30 * (1 + 0.5 * Z_clipped)
        delta_I = I.diff(4)
        delta_I_norm = _minmax(delta_I)
        Momentum = delta_I_norm * 20
        HCS = (Base + Adjustment + Momentum).clip(0, 100)
        return HCS

    # ── Full pipeline ─────────────────────────────────────────────

    def run(self, df: pd.DataFrame, baseline_col: str = 'phase',
            baseline_val: str = 'P1') -> pd.DataFrame:
        """執行完整 5 層 pipeline"""
        I, M, S = self.calibrate(df, baseline_col, baseline_val)
        sigma_4h, Z = self.compute_zscore(I)
        states = self.classify_state(Z)
        HCS = self.compute_hcs(I, Z)

        result = df.copy()
        result['M'] = M.values
        result['S'] = S.values
        result['I'] = I.values
        result['sigma_4h'] = sigma_4h.values
        result['Z'] = Z.values
        result['State'] = states.values
        result['HCS'] = HCS.values
        self._result = result
        return result

    # ── Real-time scoring (single-point snapshot) ─────────────────

    def score_now(self, snapshot: Dict) -> Dict:
        """
        實時單點評分

        對於單點 snapshot，Z 由兩個分量合成：
        - Z_level: 當前 I偏離 P1分佈的程度（σ from P1 mean）
        - Z_vol:   當前 I引入的滾動窓波動率 vs 基線

        Z = max(Z_level, Z_vol) — 捕捉最強的信號
        """
        if not self._calibrated:
            raise RuntimeError("⚠️ 模型未校準，請先執行 run() 或 calibrate()")

        snap_df = pd.DataFrame([snapshot])

        M = self.compute_maritime(snap_df, global_norm=True).iloc[0]
        S = self.compute_nlp(snap_df, global_norm=True).iloc[0]
        I_val = float(self.cfg.w_maritime * M + self.cfg.w_nlp * S)

        # ── Z-score: combined level + volatility ──
        # Z_level: how far I_val is from P1 mean (in units of P1 std)
        Z_level = (I_val - self.p1_I_mean) / self.p1_I_std if self.p1_I_std > 1e-10 else 0.0

        # Z_vol: rolling window std with I_val added vs baseline
        ref_series = pd.Series(self._ref_buffer)
        window_tail = list(self._ref_buffer[-self.cfg.vol_window:]) + [I_val]
        window_series = pd.Series(window_tail[-self.cfg.vol_window:])
        current_std = float(window_series.std())
        Z_vol = ((current_std - self.baseline_mu) / self.baseline_sigma
                  if self.baseline_sigma > 1e-10 else 0.0)

        # Use the larger magnitude — captures whichever signal fires
        Z = Z_level if abs(Z_level) > abs(Z_vol) else Z_vol
        sigma_4h = current_std

        # ── Update live buffer (for momentum only) ──
        self._I_buffer.append(I_val)
        if len(self._I_buffer) > self.cfg.vol_window * 4:
            self._I_buffer = self._I_buffer[-self.cfg.vol_window * 2:]

        # ── State ──
        if Z >= self.cfg.z_orange_red:
            state = 'RED'
        elif Z >= self.cfg.z_yellow_orange:
            state = 'ORANGE'
        elif Z >= self.cfg.z_green_yellow:
            state = 'YELLOW'
        else:
            state = 'GREEN'

        # ── HCS Score ──
        k = self.cfg.sigmoid_k
        mid = self.cfg.sigmoid_midpoint
        Base = (1 / (1 + np.exp(-k * (Z - mid)))) * 50

        # I_norm: normalize against P1 I distribution
        I_norm = _minmax_scalar(I_val, self.p1_I_mean - self.p1_I_std,
                                 self.p1_I_mean + 3 * self.p1_I_std)
        I_norm = max(0.0, min(1.0, I_norm))
        # Z contribution capped at 5 to prevent blow-up (Z>5 already = RED anyway)
        Z_clip = min(max(Z, 0), 5.0)
        Adjustment = I_norm * 30 * (1 + 0.5 * Z_clip)

        # Momentum: uses live buffer
        if len(self._I_buffer) >= 5:
            delta = self._I_buffer[-1] - self._I_buffer[-5]
            dI_std = max(pd.Series(self._I_buffer).diff().std(), 1e-10)
            delta_norm = max(-1.0, min(1.0, delta / (3 * dI_std)))
            Mom = delta_norm * 20
        else:
            Mom = 0.0

        HCS = max(0.0, min(100.0, Base + Adjustment + Mom))

        # ── Actions ──
        actions = {
            'RED':    '🚨 TRIAGE MODE: 做多 OVX 波動率期權，減恒指多頭，加恒指看跌對沖',
            'ORANGE': '⚠️ 高度警戒: 減倉航運/能源多頭，增加現金，預設對沖觸發條件',
            'YELLOW': '🔶 關注: 監控新聞頻率，縮小止損，避免新建高位能源倉位',
            'GREEN':  '✅ 正常: 可維持常規策略，繼續監測',
        }

        return {
            'HCS': round(float(HCS), 1),
            'State': state,
            'State_Emoji': STATE_EMOJI[state],
            'Z': round(float(Z), 2),
            'Z_level': round(float(Z_level), 2),
            'Z_vol': round(float(Z_vol), 2),
            'I_composite': round(I_val, 4),
            'M_maritime': round(float(M), 4),
            'S_nlp': round(float(S), 4),
            'sigma_4h': round(sigma_4h, 6),
            'base': round(float(Base), 2),
            'adjustment': round(float(Adjustment), 2),
            'momentum': round(float(Mom), 2),
            'action': actions[state],
            'reason': f"Z={Z:.2f}(L={Z_level:.2f},V={Z_vol:.2f}) I={I_val:.4f}",
        }

    def summary(self, result: Optional[pd.DataFrame] = None) -> str:
        """返回模型輸出的摘要報告"""
        df = result or self._result
        if df is None:
            return "⚠️ 未有結果，請先執行 run()"

        lines = ["\n📊 HSCDM v1.0 — 階段摘要"]
        for ph in df['phase'].unique() if 'phase' in df.columns else ['ALL']:
            sub = df if ph == 'ALL' else df[df['phase'] == ph]
            states = sub['State'].value_counts(normalize=True)
            label = {'P1': '🟢 常態', 'P2': '🟡 升級', 'P3': '🔴 危機'}.get(ph, f'Phase {ph}')
            lines.append(f"\n  {label}:")
            lines.append(f"    HCS Mean={sub['HCS'].mean():.1f} Max={sub['HCS'].max():.1f}")
            lines.append(f"    Z    Mean={sub['Z'].mean():.2f} Max={sub['Z'].max():.2f}")
            for st in STATE_ORDER:
                pct = states.get(st, 0) * 100
                lines.append(f"    {STATE_EMOJI[st]} {st:8s}: {pct:5.1f}%")
        return "\n".join(lines)
