"""
코인 신호 + 급등락 알림 봇 (GitHub Actions용 / 조회 전용, 주문 기능 없음)

실행 모드
  --mode signal : 1시간봉으로 매수·매도 참고 신호 계산 + 차트 저장   (15분마다)
  --mode spike  : 최근 5분 가격 변동 확인, ±2% 이상이면 알림          (5분마다)
  --test-alert  : 텔레그램 연결 테스트 메시지 전송
"""
import os
import sys
import json
import argparse
import datetime as dt
from pathlib import Path

import requests
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ===================== 설정 =====================
MARKETS = ["KRW-BTC", "KRW-ETH"]
UPBIT_URL = "https://api.upbit.com/v1/candles/minutes/{unit}"

SIGNAL_UNIT = 60            # 신호 계산 봉 단위(분). 60 = 1시간봉
SIGNAL_COUNT = 200
SMA_SHORT, SMA_LONG = 20, 60
RSI_PERIOD = 14
RSI_LOW, RSI_HIGH = 30, 70

SPIKE_PCT = 2.0             # 이 % 이상 움직이면 급등/급락 알림
SPIKE_WINDOW = 5            # 몇 분 사이의 변동을 볼지
SPIKE_COOLDOWN_MIN = 15     # 같은 방향 알림은 이 시간(분) 안에 반복하지 않음

HISTORY_KEEP = 50
DOCS = Path("docs")
KST = dt.timezone(dt.timedelta(hours=9))
BUY, SELL, HOLD = "매수 참고", "매도 참고", "관망"
# =================================================


def coin_name(market):
    return market.replace("KRW-", "")


def load_json(path, default):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def save_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


# ---------------- 데이터 ----------------
def fetch_minutes(market, unit, count):
    """업비트 분봉을 오래된 순서의 DataFrame으로 반환합니다."""
    resp = requests.get(
        UPBIT_URL.format(unit=unit),
        params={"market": market, "count": count},
        timeout=10,
    )
    resp.raise_for_status()
    rows = list(reversed(resp.json()))  # 업비트는 최신순으로 줌
    return pd.DataFrame(
        {
            "time": pd.to_datetime([r["candle_date_time_kst"] for r in rows]),
            "close": [float(r["trade_price"]) for r in rows],
        }
    )


# ---------------- 지표 / 신호 ----------------
def indicators(df):
    df = df.copy()
    df["sma_s"] = df["close"].rolling(SMA_SHORT).mean()
    df["sma_l"] = df["close"].rolling(SMA_LONG).mean()
    diff = df["close"].diff()
    gain = diff.clip(lower=0).rolling(RSI_PERIOD).mean()
    loss = (-diff.clip(upper=0)).rolling(RSI_PERIOD).mean()
    df["rsi"] = 100 - 100 / (1 + gain / loss)
    df["cross_up"] = (df["sma_s"].shift(1) <= df["sma_l"].shift(1)) & (df["sma_s"] > df["sma_l"])
    df["cross_dn"] = (df["sma_s"].shift(1) >= df["sma_l"].shift(1)) & (df["sma_s"] < df["sma_l"])
    return df


def judge(df):
    """규칙 기반 참고 신호. 수익을 보장하지 않습니다."""
    last, prev = df.iloc[-1], df.iloc[-2]
    if pd.isna(last["sma_l"]) or pd.isna(last["rsi"]):
        return HOLD, ["데이터 부족"]

    reasons = []
    rsi_now, rsi_prev = float(last["rsi"]), float(prev["rsi"])
    rsi_bounce = rsi_now <= RSI_LOW and rsi_now > rsi_prev
    rsi_fall = rsi_now >= RSI_HIGH and rsi_now < rsi_prev

    buy = bool(last["cross_up"]) or rsi_bounce
    sell = bool(last["cross_dn"]) or rsi_fall

    if last["cross_up"]:
        reasons.append("골든크로스 발생 (단기선이 장기선 위로)")
    if last["cross_dn"]:
        reasons.append("데드크로스 발생 (단기선이 장기선 아래로)")
    if rsi_bounce:
        reasons.append(f"RSI {rsi_now:.1f} 과매도 구간에서 반등")
    if rsi_fall:
        reasons.append(f"RSI {rsi_now:.1f} 과매수 구간에서 하락 전환")

    if buy and sell:
        return HOLD, reasons + ["매수·매도 신호가 동시에 나와 관망"]
    if buy:
        return BUY, reasons
    if sell:
        return SELL, reasons
    return HOLD, ["뚜렷한 신호 없음"]


# ---------------- 차트 ----------------
def make_chart(df, market):
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(10, 6), sharex=True, gridspec_kw={"height_ratios": [3, 1]}
    )
    ax1.plot(df["time"], df["close"], color="#3b82f6", lw=1.2, label="Price")
    ax1.plot(df["time"], df["sma_s"], color="#f59e0b", lw=1.2, label=f"SMA{SMA_SHORT}")
    ax1.plot(df["time"], df["sma_l"], color="#10b981", lw=1.2, label=f"SMA{SMA_LONG}")
    up = df[df["cross_up"]]
    dn = df[df["cross_dn"]]
    ax1.scatter(up["time"], up["close"], marker="^", color="#059669", s=80, zorder=5, label="Buy cross")
    ax1.scatter(dn["time"], dn["close"], marker="v", color="#e11d48", s=80, zorder=5, label="Sell cross")
    ax1.set_title(f"{coin_name(market)} - 1h candles")
    ax1.legend(loc="upper left", fontsize=8)
    ax1.grid(alpha=0.25)

    ax2.plot(df["time"], df["rsi"], color="#8b5cf6", lw=1.2)
    ax2.axhline(RSI_HIGH, ls="--", color="#e11d48", lw=0.8)
    ax2.axhline(RSI_LOW, ls="--", color="#059669", lw=0.8)
    ax2.set_ylim(0, 100)
    ax2.set_ylabel(f"RSI({RSI_PERIOD})")
    ax2.grid(alpha=0.25)

    fig.autofmt_xdate()
    DOCS.mkdir(parents=True, exist_ok=True)
    fig.savefig(DOCS / f"chart_{market}.png", dpi=90)
    plt.close(fig)


# ---------------- 알림 ----------------
def telegram(text):
    token = os.environ.get("TELEGRAM_TOKEN", "").strip()
    chat_id = os.environ.get("TELEGRAM_CHAT_ID", "").strip()
    if not token or not chat_id:
        print("텔레그램 설정(TELEGRAM_TOKEN, TELEGRAM_CHAT_ID)이 없어 알림을 건너뜁니다.")
        return False
    try:
        resp = requests.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            data={"chat_id": chat_id, "text": text},
            timeout=10,
        )
    except Exception as e:  # 예외 메시지에 주소(토큰 포함)가 섞이지 않도록 종류만 출력
        print(f"텔레그램 요청 실패: {type(e).__name__}")
        return False
    if resp.status_code != 200:
        print(f"텔레그램 전송 실패: HTTP {resp.status_code} / {resp.text[:200]}")
        return False
    print("텔레그램 전송 완료")
    return True


def add_history(now, kind, market, message):
    hist = load_json(DOCS / "alerts.json", [])
    hist.insert(0, {
        "time": now.strftime("%m/%d %H:%M"),
        "kind": kind,
        "coin": coin_name(market),
        "message": message,
    })
    save_json(DOCS / "alerts.json", hist[:HISTORY_KEEP])


# ---------------- 모드 1: 신호 (15분) ----------------
def run_signal(now):
    state = load_json(DOCS / "state.json", {})
    coins = []

    for market in MARKETS:
        entry = {"market": market}
        try:
            df = indicators(fetch_minutes(market, SIGNAL_UNIT, SIGNAL_COUNT))
            label, reasons = judge(df)
            last = df.iloc[-1]
            entry.update({
                "price": round(float(last["close"]), 0),
                "sma_short": round(float(last["sma_s"]), 0) if pd.notna(last["sma_s"]) else None,
                "sma_long": round(float(last["sma_l"]), 0) if pd.notna(last["sma_l"]) else None,
                "rsi": round(float(last["rsi"]), 1) if pd.notna(last["rsi"]) else None,
                "golden_state": bool(last["sma_s"] > last["sma_l"]),
                "signal": label,
                "reasons": reasons,
            })
            make_chart(df, market)

            # 신호가 '새로' 바뀌었을 때만 알림 (같은 신호 반복 방지)
            prev_label = state.get(market, {}).get("label")
            if label != prev_label and label in (BUY, SELL):
                msg = (f"[{label}] {coin_name(market)} · {entry['price']:,.0f}원\n"
                       + "\n".join(f"- {r}" for r in reasons)
                       + "\n※ 참고용 신호이며 매수·매도를 권유하지 않습니다.")
                telegram(f"[코인 신호] {now.strftime('%m/%d %H:%M')}\n{msg}")
                add_history(now, "신호", market, f"{label} · {', '.join(reasons)}")
            state.setdefault(market, {})["label"] = label
        except Exception as e:
            entry["error"] = str(e)
        coins.append(entry)

    save_json(DOCS / "data.json", {"updated_at": now.strftime("%Y-%m-%d %H:%M KST"), "coins": coins})
    save_json(DOCS / "state.json", state)
    print(json.dumps(coins, ensure_ascii=False, indent=2))


# ---------------- 모드 2: 급등락 (5분) ----------------
def run_spike(now):
    state = load_json(DOCS / "state.json", {})
    coins = []

    for market in MARKETS:
        entry = {"market": market}
        try:
            df = fetch_minutes(market, 1, SPIKE_WINDOW + 1)
            price = float(df["close"].iloc[-1])
            base = float(df["close"].iloc[0])
            change = round((price - base) / base * 100, 2)
            entry.update({"price": round(price, 0), "change": change})

            if abs(change) >= SPIKE_PCT:
                direction = "급등" if change > 0 else "급락"
                last = state.get(market, {}).get("spike")
                cooled = True
                if last and last.get("dir") == direction:
                    last_t = dt.datetime.fromisoformat(last["time"])
                    if now - last_t < dt.timedelta(minutes=SPIKE_COOLDOWN_MIN):
                        cooled = False
                if cooled:
                    msg = (f"[{direction}] {coin_name(market)} {SPIKE_WINDOW}분 동안 {change:+.2f}%\n"
                           f"현재가 {price:,.0f}원")
                    telegram(f"[급등락 알림] {now.strftime('%m/%d %H:%M')}\n{msg}")
                    add_history(now, direction, market, f"{SPIKE_WINDOW}분 {change:+.2f}%")
                    state.setdefault(market, {})["spike"] = {"dir": direction, "time": now.isoformat()}
        except Exception as e:
            entry["error"] = str(e)
        coins.append(entry)

    save_json(DOCS / "spike.json", {"updated_at": now.strftime("%Y-%m-%d %H:%M KST"), "coins": coins})
    save_json(DOCS / "state.json", state)
    print(json.dumps(coins, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["signal", "spike"])
    parser.add_argument("--test-alert", action="store_true")
    args = parser.parse_args()

    now = dt.datetime.now(KST)

    if args.test_alert:
        ok = telegram("✅ 텔레그램 연결 테스트 성공입니다. 이 메시지가 보이면 알림 설정은 정상입니다.")
        sys.exit(0 if ok else 1)

    if args.mode == "signal":
        run_signal(now)
    elif args.mode == "spike":
        run_spike(now)
    else:
        parser.error("--mode signal 또는 spike 를 지정하세요.")


if __name__ == "__main__":
    main()
