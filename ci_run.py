"""
GitHub Actions에서 실행되는 스크립트.
지표를 계산해서 docs/data.json 파일로 저장합니다. (조회 전용, 매매 기능 없음)
"""
import json
import datetime as dt
from pathlib import Path

import pandas as pd

import coin_monitor as cm

OUT_FILE = Path("docs") / "data.json"
KST = dt.timezone(dt.timedelta(hours=9))


def to_float(value):
    return None if pd.isna(value) else round(float(value), 2)


def main():
    result = {
        "updated_at": dt.datetime.now(KST).strftime("%Y-%m-%d %H:%M KST"),
        "coins": [],
    }
    for market in cm.MARKETS:
        try:
            df = cm.add_indicators(cm.fetch_candles(market))
            last = df.iloc[-1]
            result["coins"].append(
                {
                    "market": market,
                    "price": to_float(last["close"]),
                    "sma_short": to_float(last["sma_short"]),
                    "sma_long": to_float(last["sma_long"]),
                    "rsi": None if pd.isna(last["rsi"]) else round(float(last["rsi"]), 1),
                    "golden_state": bool(last["sma_short"] > last["sma_long"]),
                    "signals": cm.check_signals(df),
                }
            )
        except Exception as e:
            result["coins"].append({"market": market, "error": str(e)})

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUT_FILE.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
