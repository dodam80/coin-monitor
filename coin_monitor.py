"""
업비트 코인 지표 알림 프로그램 (조회 전용 - 매매 기능 없음)

- 업비트 공개 시세 API만 사용합니다. API 키가 필요 없습니다.
- 이동평균선(SMA20, SMA60) 골든크로스 / 데드크로스를 감지합니다.
- RSI(14) 과매수 / 과매도 구간을 감지합니다.
- 알림은 화면 출력 + alerts.log 파일에 기록되고,
  차트는 output 폴더에 PNG로 저장됩니다.

실행 방법
  pip install requests pandas matplotlib
  python coin_monitor.py            # 계속 실행 (5분마다 갱신, Ctrl+C로 종료)
  python coin_monitor.py --once     # 한 번만 실행하고 종료
"""

import sys
import time
import datetime as dt
from pathlib import Path

import requests
import pandas as pd
import matplotlib

matplotlib.use("Agg")  # 화면 없이 PNG 파일로만 저장
import matplotlib.pyplot as plt


# ===================== 설정 =====================
MARKETS = ["KRW-BTC", "KRW-ETH"]   # 감시할 코인 (업비트 마켓 코드)
CANDLE_UNIT = 60                   # 분봉 단위 (60 = 1시간봉)
CANDLE_COUNT = 200                 # 가져올 캔들 개수 (최소 60 이상)
CHECK_INTERVAL_SEC = 300           # 갱신 주기 (초) - 300 = 5분
SMA_SHORT = 20
SMA_LONG = 60
RSI_PERIOD = 14
RSI_OVERSOLD = 30                  # 이 값 이하면 과매도 알림
RSI_OVERBOUGHT = 70                # 이 값 이상이면 과매수 알림
OUT_DIR = Path(__file__).resolve().parent / "output"
API_URL = "https://api.upbit.com/v1/candles/minutes/{unit}"
# =================================================


def fetch_candles(market: str) -> pd.DataFrame:
    """업비트에서 분봉 데이터를 받아 시간 오름차순 DataFrame으로 돌려줍니다."""
    resp = requests.get(
        API_URL.format(unit=CANDLE_UNIT),
        params={"market": market, "count": CANDLE_COUNT},
        timeout=10,
    )
    resp.raise_for_status()
    rows = resp.json()
    df = pd.DataFrame(
        {
            "time": pd.to_datetime([r["candle_date_time_kst"] for r in rows]),
            "close": [float(r["trade_price"]) for r in rows],
        }
    )
    return df.sort_values("time").reset_index(drop=True)


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """이동평균선과 RSI를 계산해서 열로 추가합니다."""
    df = df.copy()
    df["sma_short"] = df["close"].rolling(SMA_SHORT).mean()
    df["sma_long"] = df["close"].rolling(SMA_LONG).mean()

    delta = df["close"].diff()
    avg_gain = delta.clip(lower=0).rolling(RSI_PERIOD).mean()
    avg_loss = (-delta.clip(upper=0)).rolling(RSI_PERIOD).mean()
    rs = avg_gain / avg_loss
    df["rsi"] = 100 - (100 / (1 + rs))
    return df


def check_signals(df: pd.DataFrame) -> list[str]:
    """마지막 캔들 기준으로 알림 문구 목록을 만듭니다."""
    msgs = []
    last = df.iloc[-1]
    prev = df.iloc[-2]

    # 골든크로스: 단기선이 장기선을 아래에서 위로 돌파
    if prev["sma_short"] <= prev["sma_long"] and last["sma_short"] > last["sma_long"]:
        msgs.append("골든크로스 발생 (단기 이평선이 장기 이평선을 상향 돌파)")

    # 데드크로스: 단기선이 장기선을 위에서 아래로 돌파
    if prev["sma_short"] >= prev["sma_long"] and last["sma_short"] < last["sma_long"]:
        msgs.append("데드크로스 발생 (단기 이평선이 장기 이평선을 하향 돌파)")

    if pd.notna(last["rsi"]):
        if last["rsi"] <= RSI_OVERSOLD:
            msgs.append(f"RSI {last['rsi']:.1f}: 과매도 구간 ({RSI_OVERSOLD} 이하)")
        elif last["rsi"] >= RSI_OVERBOUGHT:
            msgs.append(f"RSI {last['rsi']:.1f}: 과매수 구간 ({RSI_OVERBOUGHT} 이상)")
    return msgs


def save_chart(df: pd.DataFrame, market: str) -> Path:
    """가격·이동평균선과 RSI를 한 장의 PNG 차트로 저장합니다."""
    OUT_DIR.mkdir(exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(10, 7), sharex=True, gridspec_kw={"height_ratios": [3, 1]}
    )
    ax1.plot(df["time"], df["close"], label="Close", linewidth=1)
    ax1.plot(df["time"], df["sma_short"], label=f"SMA{SMA_SHORT}")
    ax1.plot(df["time"], df["sma_long"], label=f"SMA{SMA_LONG}")
    ax1.set_title(f"{market} - {CANDLE_UNIT}min candles")
    ax1.legend(loc="upper left")
    ax1.grid(alpha=0.3)

    ax2.plot(df["time"], df["rsi"], color="purple", linewidth=1)
    ax2.axhline(RSI_OVERBOUGHT, ls="--", color="red", linewidth=0.8)
    ax2.axhline(RSI_OVERSOLD, ls="--", color="green", linewidth=0.8)
    ax2.set_ylim(0, 100)
    ax2.set_ylabel(f"RSI({RSI_PERIOD})")
    ax2.grid(alpha=0.3)

    fig.autofmt_xdate()
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M")
    path = OUT_DIR / f"{market}_{stamp}.png"
    fig.savefig(path, dpi=100)
    plt.close(fig)
    return path


def write_log(line: str) -> None:
    """alerts.log 파일에 한 줄 기록합니다."""
    OUT_DIR.mkdir(exist_ok=True)
    with open(OUT_DIR / "alerts.log", "a", encoding="utf-8") as f:
        f.write(line + "\n")


def run_once() -> None:
    now = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"\n=== 갱신 시각: {now} ===")
    for market in MARKETS:
        try:
            df = add_indicators(fetch_candles(market))
            last = df.iloc[-1]
            rsi_text = f"{last['rsi']:.1f}" if pd.notna(last["rsi"]) else "-"
            print(f"{market}: 현재가 {last['close']:,.0f}원 | RSI {rsi_text}")

            msgs = check_signals(df)
            for msg in msgs:
                alert = f"[알림] {now} {market} - {msg}"
                print(alert)
                write_log(alert)
            if not msgs:
                print("  특이 신호 없음")

            chart = save_chart(df, market)
            print(f"  차트 저장: {chart}")
        except requests.RequestException as e:
            err = f"[오류] {now} {market} 시세 조회 실패: {e}"
            print(err)
            write_log(err)
        except Exception as e:  # 예상 못 한 오류도 프로그램이 멈추지 않도록
            err = f"[오류] {now} {market} 처리 중 문제: {e}"
            print(err)
            write_log(err)


def main() -> None:
    once = "--once" in sys.argv
    print("업비트 지표 알림 프로그램 시작 (조회 전용, 매매 기능 없음)")
    print(f"감시 코인: {', '.join(MARKETS)} | 봉 단위: {CANDLE_UNIT}분")
    if once:
        run_once()
        return
    print(f"{CHECK_INTERVAL_SEC // 60}분마다 갱신합니다. 종료하려면 Ctrl+C를 누르세요.")
    try:
        while True:
            run_once()
            time.sleep(CHECK_INTERVAL_SEC)
    except KeyboardInterrupt:
        print("\n프로그램을 종료합니다.")


if __name__ == "__main__":
    main()
