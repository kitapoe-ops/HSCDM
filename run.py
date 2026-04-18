"""
MTC CLI — 運行界面
用法：
    python -m mtc.run                     # 示範模式（45天回測）
    python -m mtc.run --live              # 實時評分（需要 API key）
    python -m mtc.run --snapshot <json>   # 單次快照評分
"""
from __future__ import annotations

import sys
import json
import argparse
from datetime import datetime

# Add parent to path for local imports
sys.path.insert(0, __file__.rsplit('/', 2)[0] if '/' in __file__ else '.')

from mtc.model import HSCDM, HSCDMConfig, STATE_EMOJI
from mtc.scoring import MTCScorer, MTCSnapshot
from mtc.data_generator import generate_synthetic_data
from mtc.nlp import NewsSentimentAnalyzer
from mtc.api_clients import MarineTrafficClient
from mtc.config import MTCConfig


def run_demo():
    """45天回測演示"""
    print("=" * 65)
    print("  MTC — Marine Traffic Crisis Detection  (Demo Mode)")
    print("  45天回測 | HSCDM v1.0 | 15min 頻率")
    print("=" * 65)

    print("\n🛰️  生成模擬數據...")
    df = generate_synthetic_data(days=45, freq_min=15)
    print(f"   {len(df)} 個數據點  |  {df.index[0].strftime('%Y-%m-%d')} ~ {df.index[-1].strftime('%Y-%m-%d')}")

    print("\n⚙️  初始化模型...")
    cfg = HSCDMConfig()
    model = HSCDM(cfg)

    print("📐 執行 HSCDM 完整 pipeline...")
    result = model.run(df)

    # Summary
    print(model.summary())

    # Phase 3 crisis highlight
    p3 = result[result['phase'] == 'P3']
    print(f"\n🔴 Phase 3 (危機期) 關鍵指標:")
    print(f"   HCS > 80 的時間點: {(p3['HCS'] > 80).sum()} / {len(p3)} 個")
    print(f"   Z > 3 (觸發) 的時間點: {(p3['Z'] > 3).sum()} / {len(p3)} 個")
    print(f"   RED state 比例: {(p3['State'] == 'RED').mean() * 100:.1f}%")

    # Real-time scenario demos
    print("\n" + "=" * 65)
    print("  實時評分情境演示")
    print("=" * 65)

    # Must calibrate first
    model.calibrate(df)

    scenarios = {
        "🟢 常態暢通": {
            'throughput': 22, 'avg_speed': 12.5, 'deviation_index': 0.02,
            'anchor_count': 2, 'danger_score': 0.3, 'calming_score': 1.5,
            'source_reliability': 0.85,
        },
        "🟡 輕度受阻": {
            'throughput': 14, 'avg_speed': 8.0, 'deviation_index': 0.15,
            'anchor_count': 12, 'danger_score': 2.5, 'calming_score': 0.8,
            'source_reliability': 0.7,
        },
        "🟠 嚴重封鎖": {
            'throughput': 1, 'avg_speed': 0.5, 'deviation_index': 0.85,
            'anchor_count': 45, 'danger_score': 8.0, 'calming_score': 0.1,
            'source_reliability': 0.4,
        },
        "🔴 反覆開關中": {
            'throughput': 8, 'avg_speed': 4.0, 'deviation_index': 0.5,
            'anchor_count': 30, 'danger_score': 6.0, 'calming_score': 0.3,
            'source_reliability': 0.5,
        },
    }

    scorer = MTCScorer()
    scorer.calibrate(df)

    for name, snap in scenarios.items():
        score = scorer.score_now.__wrapped__(scorer, snap) if hasattr(scorer.score_now, '__wrapped__') else None

        # Direct model call
        r = model.score_now(snap)
        emoji = STATE_EMOJI[r['State']]
        bar = scorer._make_bar(r['HCS'])

        print(f"\n  {name}")
        print(f"  {emoji} HCS: {r['HCS']}/100  {bar}")
        print(f"     Z={r['Z']}  I={r['I_composite']}  σ₄ₕ={r['sigma_4h']:.6f}")
        print(f"     {r['action']}")

    print("\n✅ 演示完成")
    return result


def run_live(args):
    """實時評分模式"""
    print("=" * 65)
    print("  MTC — Marine Traffic Crisis Detection  (Live Mode)")
    print("=" * 65)

    mtc_cfg = MTCConfig()

    # Initialize components
    mt_client = MarineTrafficClient()
    nlp_analyzer = NewsSentimentAnalyzer(
        news_sources=mtc_cfg.news_sources,
    )
    scorer = MTCScorer(mt_client=mt_client, nlp_analyzer=nlp_analyzer)

    # Try to load historical data for calibration
    try:
        import os
        hist_path = os.path.join(os.path.dirname(__file__), 'history.csv')
        if os.path.exists(hist_path):
            import pandas as pd
            df = pd.read_csv(hist_path, parse_dates=['timestamp'], index_col='timestamp')
            scorer.calibrate(df)
            print(f"✅ 從 history.csv 載入並校準")
        else:
            print("⚠️ 無 history.csv，使用模擬基線")
            df = generate_synthetic_data(days=30)
            scorer.calibrate(df)
    except Exception as e:
        print(f"⚠️ 校準失敗: {e}，使用模擬基線")
        df = generate_synthetic_data(days=30)
        scorer.calibrate(df)

    # Mock warning
    if mt_client.is_mock:
        print("\n⚠️ 警告: 無 MarineTraffic API Key，使用模擬數據")
        print("   設定環境變數 MARINETRAFFIC_API_KEY 以獲取真實數據")

    # Interval scoring
    interval = args.interval or 15
    count = args.count or 1

    print(f"\n⏱️  評分間隔: {interval} 分鐘 | 次數: {count}")
    print("-" * 65)

    for i in range(count):
        if i > 0:
            import time
            time.sleep(interval * 60)

        print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 評分 #{i + 1}")
        result = scorer.score_now(force_refresh=(i == 0))
        print(scorer.format_result(result))

    return scorer


def run_snapshot(args):
    """單次快照評分"""
    if args.snapshot:
        data = json.loads(args.snapshot)
    else:
        # Read from file
        with open(args.file, 'r') as f:
            data = json.load(f)

    # Calibrate with minimal baseline
    df = generate_synthetic_data(days=30)
    scorer = MTCScorer()
    scorer.calibrate(df)

    result = scorer.score_now.__wrapped__(scorer, data) if hasattr(scorer.score_now, '__wrapped__') else None
    r = scorer.model.score_now(data)
    r['timestamp'] = datetime.now().isoformat()

    print(json.dumps(r, indent=2, ensure_ascii=False))
    return r


def main():
    parser = argparse.ArgumentParser(description='MTC — Marine Traffic Crisis Detection')
    sub = parser.add_subparsers(dest='cmd')

    sub.add_parser('demo', help='45天回測演示模式')
    sub.add_parser('live', help='實時評分模式')

    live_p = sub.add_parser('live', help='實時評分')
    live_p.add_argument('--interval', type=int, default=15, help='評分間隔（分鐘）')
    live_p.add_argument('--count', type=int, default=1, help='評分次數')

    snap = sub.add_parser('snapshot', help='單次快照評分')
    snap.add_argument('--snapshot', type=str, help='JSON 字串')
    snap.add_argument('--file', type=str, help='JSON 文件路徑')

    args = parser.parse_args()

    if args.cmd == 'demo' or args.cmd is None:
        run_demo()
    elif args.cmd == 'live':
        run_live(args)
    elif args.cmd == 'snapshot':
        run_snapshot(args)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
