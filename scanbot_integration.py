"""
MTC × Scanbot 3.4 整合層
HSCDM State → Scanbot Trading Rules 映射
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional
from enum import IntEnum

from .model import HSCDM, STATE_EMOJI
from .scoring import MTCScorer
from .config import HSCDMConfig


# ─── Trading Rule Tables ─────────────────────────────────────────

class StrategyMode(IntEnum):
    FULL = 2    # 全速運行
    REDUCED = 1  # 減速
    DISABLED = 0  # 暫停


@dataclass
class ScanbotRules:
    """
    Scanbot 3.4 策略規則集
    由 HSCDM State 觸發
    """
    state: str
    state_emoji: str

    # Mean Reversion (均值回歸)
    mean_reversion: StrategyMode
    mr_comment: str

    # Momentum (動量)
    momentum: StrategyMode
    mom_comment: str

    # OVX Exposure (原油波動率敞口)
    ovx_exposure_pct: float
    ovx_comment: str

    # HSI Hedge Ratio (恒指對沖比例)
    hsi_hedge_pct: float
    hsi_comment: str

    # Global posture
    posture: str
    priority_action: str
    override: bool  # True = 完全接管，忽略其他信號


# ─── State → Rules 映射 ─────────────────────────────────────────

HSCDM_RULES: Dict[str, ScanbotRules] = {
    'GREEN': ScanbotRules(
        state='GREEN',
        state_emoji='🟢',
        mean_reversion=StrategyMode.FULL,
        mr_comment='常規量化策略全線運作',
        momentum=StrategyMode.FULL,
        mom_comment='動量策略正常運行',
        ovx_exposure_pct=0.0,
        ovx_comment='無需配置 OVX 敞口',
        hsi_hedge_pct=0.0,
        hsi_comment='恒指無需對沖',
        posture='NORMAL',
        priority_action='維持常規倉位，繼續監測 MTC 分數',
        override=False,
    ),

    'YELLOW': ScanbotRules(
        state='YELLOW',
        state_emoji='🟡',
        mean_reversion=StrategyMode.REDUCED,
        mr_comment='均值回歸倉位減半，擴大止損',
        momentum=StrategyMode.FULL,
        mom_comment='動量策略維持（順勢而為）',
        ovx_exposure_pct=5.0,
        ovx_comment='輕倉 OVX 看漲期權（對沖用途）',
        hsi_hedge_pct=10.0,
        hsi_comment='恒指多頭減10%转为 protective put',
        posture='ELEVATED',
        priority_action='縮減能源/航運高位倉位，擴大止損',
        override=False,
    ),

    'ORANGE': ScanbotRules(
        state='ORANGE',
        state_emoji='🟠',
        mean_reversion=StrategyMode.DISABLED,
        mr_comment='暫停所有均值回歸開倉，平倉現有倉位',
        momentum=StrategyMode.REDUCED,
        mom_comment='動量倉位減半，順勢短線操作',
        ovx_exposure_pct=15.0,
        ovx_comment='配置 OVX 買入跨式（long straddle）或買入看漲',
        hsi_hedge_pct=30.0,
        hsi_comment='恒指多頭減30%，加碼恒指 put 對沖',
        posture='HIGH_ALERT',
        priority_action='能源/航運全線減倉，鎖定利潤，配置對沖',
        override=True,  # Orange 開啟，Scanbot 策略需要人工確認
    ),

    'RED': ScanbotRules(
        state='RED',
        state_emoji='🔴',
        mean_reversion=StrategyMode.DISABLED,
        mr_comment='TRIAGE MODE — 均值回歸全停，禁止新建倉位',
        momentum=StrategyMode.DISABLED,
        mom_comment='TRIAGE MODE — 動量策略全停',
        ovx_exposure_pct=25.0,
        ovx_comment='全力做多 OVX — 買入 long straddle + 買入 put',
        hsi_hedge_pct=50.0,
        hsi_comment='恒指多頭砍半，加恒指 put 對沖（50%）',
        posture='TRIAGE',
        priority_action='TRIAGE MODE: 暫停均值回歸，做多 OVX 波動率，減恒指多頭50%，加恒指看跌期權',
        override=True,  # RED 完全接管
    ),
}


# ─── Score thresholds (for HCS range mapping) ────────────────────

HCS_RANGES = {
    'GREEN': (0, 40),
    'YELLOW': (41, 60),
    'ORANGE': (61, 80),
    'RED': (81, 100),
}


def hcs_to_state(hcs: float) -> str:
    """HCS 分數 → State"""
    if hcs < 0 or hcs > 100:
        return 'GREEN'
    if hcs <= 40:
        return 'GREEN'
    elif hcs <= 60:
        return 'YELLOW'
    elif hcs <= 80:
        return 'ORANGE'
    else:
        return 'RED'


# ─── Scanbot Integration Class ──────────────────────────────────

class HSCDMScanbotBridge:
    """
    HSCDM ↔ Scanbot 3.4 橋接器

    用法：
        bridge = HSCDMScanbotBridge()
        bridge.calibrate(history_df)

        # 每次决策前調用
        decision = bridge.get_decision(mtc_result)
        if decision.override:
            print("⚠️ 需要人工確認")

        # Scanbot 直接讀取這些字段
        print(decision.priority_action)
        print(f"MR: {decision.mean_reversion.name}")
        print(f"Ovx: {decision.ovx_exposure_pct}%")
        print(f"HSI Hedge: {decision.hsi_hedge_pct}%")
    """

    def __init__(self, config: Optional[HSCDMConfig] = None):
        self.cfg = config or HSCDMConfig()
        self.model = HSCDM(self.cfg)
        self._calibrated = False

    def calibrate(self, df, baseline_col='phase', baseline_val='P1') -> 'HSCDMScanbotBridge':
        """用歷史數據校準模型"""
        self.model.run(df, baseline_col, baseline_val)
        self._calibrated = True
        return self

    def score_and_bridge(self, snapshot: Dict) -> ScanbotRules:
        """
        一步完成：snapshot → HSCDM → ScanbotRules
        """
        if not self._calibrated:
            raise RuntimeError("⚠️ 請先調用 calibrate()")

        result = self.model.score_now(snapshot)
        state = result['State']
        rules = HSCDM_RULES[state]

        return rules

    def get_decision(self, mtc_result: Dict) -> ScanbotRules:
        """
        將 MTCScorer 的輸出轉換為 Scanbot 決策

        輸入: MTCScorer.score_now() 的返回 dict
        輸出: ScanbotRules
        """
        state = mtc_result.get('State', 'GREEN')
        rules = HSCDM_RULES.get(state, HSCDM_RULES['GREEN'])
        return rules

    # ─── Formatted output ────────────────────────────────────────

    def format_decision(self, mtc_result: Dict) -> str:
        """
        格式化為 Scanbot 友好的决策報告
        """
        rules = self.get_decision(mtc_result)
        hcs = mtc_result['HCS']
        z = mtc_result['Z']
        state_e = rules.state_emoji

        lines = [
            f"{state_e} **Scanbot Decision — {rules.state}**",
            f"   HCS: {hcs}/100  |  Z: {z}  |  Posture: {rules.posture}",
            "",
            f"📐 **策略規則：**",
            f"   均值回歸: {rules.mr_comment}",
            f"   動量:     {rules.mom_comment}",
            f"   OVX 敞口: {rules.ovx_exposure_pct}% — {rules.ovx_comment}",
            f"   恒指對沖: {rules.hsi_hedge_pct}% — {rules.hsi_comment}",
            "",
            f"🎯 **優先操作：**",
            f"   {rules.priority_action}",
        ]

        if rules.override:
            lines.insert(2, "   ⚠️ **OVERRIDE — 需要人工確認**")

        return "\n".join(lines)

    def get_config_overrides(self, rules: ScanbotRules) -> Dict:
        """
        返回 Scanbot 3.4 可直接使用的配置覆寫值
        """
        return {
            'mtc.state': rules.state,
            'mtc.override': rules.override,
            'mtc.posture': rules.posture,
            # Mean Reversion
            'strategy.mean_reversion.enabled': rules.mean_reversion >= StrategyMode.REDUCED,
            'strategy.mean_reversion.scale': rules.mean_reversion / StrategyMode.FULL,
            # Momentum
            'strategy.momentum.enabled': rules.momentum >= StrategyMode.REDUCED,
            'strategy.momentum.scale': rules.momentum / StrategyMode.FULL,
            # Risk
            'risk.ovx_exposure_pct': rules.ovx_exposure_pct,
            'risk.hsi_hedge_pct': rules.hsi_hedge_pct,
            # Priority
            'action.priority': rules.priority_action,
        }
