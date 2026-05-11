# backtest.py
# ──────────────────────────────────────────────────────────────────────────────
# Backtest bar-by-bar de la estrategia Bollinger-AS / Tendencia-AS (long-only).
# Fuente: yfinance (SPY u otro ETF/stock).
# Benchmark: buy & hold (floor(CAPITAL / open_0) acciones).
# ──────────────────────────────────────────────────────────────────────────────

import math
import pytz
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from estrategia import compute_model, get_signals, MID_METHOD, MODE

# ── Config ────────────────────────────────────────────────────────────────────
SOURCE             = "yfinance"
TICKER             = "NVDA"
TIMEFRAME          = "1m"
START              = "2026-05-07"
END                = "2026-05-08"

FORM               = "CJ"
CAPITAL            = 10_000
RISK_PCT           = 0.15
ATR_N              = 14
ATR_MULT           = 1.5
MA_N               = 20
TARGET_SPREAD_FRAC = 0.001
INV_LIMIT          = 15
TICK_SIZE          = 0.01
CALIB_BARS         = 20
MID_BARS           = 20
# ─────────────────────────────────────────────────────────────────────────────

_ET = pytz.timezone("America/New_York")


def _session_Tt(ts):
    """Segundos hasta el cierre de sesión (16:00 ET) del mismo día."""
    ts    = ts.astimezone(_ET) if ts.tzinfo else _ET.localize(ts)
    close = ts.replace(hour=16, minute=0, second=0, microsecond=0)
    return max((close - ts).total_seconds(), 1e-9)


def _atr(window_df, n):
    """Average True Range sobre las últimas n barras."""
    df  = window_df.tail(n + 1)
    hl  = df["high"] - df["low"]
    hpc = (df["high"] - df["close"].shift(1)).abs()
    lpc = (df["low"]  - df["close"].shift(1)).abs()
    return float(pd.concat([hl, hpc, lpc], axis=1).max(axis=1).iloc[1:].mean())


# ── Data ──────────────────────────────────────────────────────────────────────
def get_bars():
    import yfinance as yf
    df = yf.download(TICKER, start=START, end=END, interval=TIMEFRAME, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df[["Open","High","Low","Close","Volume"]].copy()
    df.columns = ["open","high","low","close","volume"]
    df.index.name = "timestamp"
    return df.dropna(subset=["open","high","low","close"]).reset_index()


# ── Backtest ──────────────────────────────────────────────────────────────────
def run_strategy(bars):
    bar_sec     = pd.Timedelta(bars["timestamp"].diff().median()).total_seconds()
    Tl          = CALIB_BARS * bar_sec
    min_i       = max(CALIB_BARS, MID_BARS, MA_N)

    q, cash     = 0.0, 0.0
    cash_long   = 0.0
    records     = []
    signals     = []
    announced   = False
    current_day = None

    # Flags Bollinger
    above_ask = False
    below_bid = False

    # Flags Tendency
    above_ask_entry  = False    # precio estuvo bajo ask sin posición → habilita entrada en cruce
    closes_below_ask = 0        # contador de cierres consecutivos bajo ask con posición

    for i, (_, row) in enumerate(bars.iterrows()):
        if i < min_i:
            continue

        calib_window = bars.iloc[i - CALIB_BARS:i]
        mid_window   = bars.iloc[i - MID_BARS:i]
        Tt           = _session_Tt(row["timestamp"])
        price        = row["open"]

        # Reset de flags al inicio de cada sesión
        bar_day = row["timestamp"].date()
        if bar_day != current_day:
            current_day      = bar_day
            above_ask        = False
            below_bid        = False
            above_ask_entry  = False
            closes_below_ask = 0

        # Cierre forzado en las últimas 2 barras de sesión
        if q > 0 and Tt < bar_sec * 2:
            cash      += price * q
            cash_long += price * q
            signals.append({"idx": len(records), "price": price, "action": "sell"})
            q                = 0.0
            above_ask        = False
            below_bid        = True
            above_ask_entry  = False
            closes_below_ask = 0

        try:
            res = compute_model(calib_window, mid_window, q,
                                TARGET_SPREAD_FRAC, Tt, Tl,
                                bar_sec, FORM, announced)
            announced = True
        except (ValueError, ZeroDivisionError):
            continue

        if any(v != v for v in [res["bid"], res["ask"], res["r"]]):
            continue

        mid_price = (row["high"] + row["low"]) / 2.0
        if res["spread"] > mid_price * 0.02:
            continue

        bid = math.floor(res["bid"] / TICK_SIZE) * TICK_SIZE
        ask = math.ceil( res["ask"] / TICK_SIZE) * TICK_SIZE
        if ask - bid < TICK_SIZE:
            bid = math.floor((0.5*(bid+ask) - TICK_SIZE/2) / TICK_SIZE) * TICK_SIZE
            ask = bid + TICK_SIZE
        res["bid"], res["ask"] = bid, ask

        prev_close = float(bars.iloc[i - 1]["close"])
        ma         = float(bars.iloc[i - MA_N:i]["close"].mean())
        atr        = _atr(calib_window, ATR_N)

        # ── Actualización de flags Bollinger ──
        if q > 0 and prev_close > ask:
            above_ask = True
        if q == 0 and prev_close < bid:
            below_bid = True
        if q == 0 and prev_close > ask:
            below_bid = False

        # ── Actualización de flags Tendency ──
        # Capturar ANTES de actualizar — el cruce ocurre en la barra siguiente
        above_ask_entry_signal = above_ask_entry   # estado de la barra anterior

        if q == 0 and prev_close < ask:
            above_ask_entry = True
        if q == 0 and prev_close >= ask:
            above_ask_entry = False
        if q > 0:
            if prev_close < ask:
                closes_below_ask += 1       # cierre bajo ask con posición
            else:
                closes_below_ask = 0        # resetear si vuelve sobre ask

        for action, qty in get_signals(
            prev_close, res, q, CAPITAL, RISK_PCT,
            atr, ATR_MULT, INV_LIMIT, ma,
            above_ask, below_bid,
            above_ask_entry_signal, closes_below_ask):   # ← signal, no el flag actualizado

            if action == "buy":
                cash             -= price * qty
                cash_long        -= price * qty
                q                += qty
                below_bid         = False
                above_ask_entry   = False
                closes_below_ask  = 0
            else:
                cash             += price * qty
                cash_long        += price * qty
                q                -= qty
                above_ask         = False
                below_bid         = True
                above_ask_entry   = False
                closes_below_ask  = 0
            signals.append({"idx": len(records), "price": price, "action": action})

        records.append({"close": row["close"],
                        "mid": res["r"], "bid": bid, "ask": ask,
                        "spread": res["spread"], "q": q,
                        "pnl": cash + q * row["close"],
                        "sigma": res["sigma"], "k": res["k"]})

    last_close = float(bars["close"].iloc[-1])
    cash_long += q * last_close
    cash      += q * last_close
    return pd.DataFrame(records), cash, pd.DataFrame(signals), cash_long


def run_benchmark(bars, df):
    entry  = float(bars["open"].iloc[0])
    shares = max(1, int(CAPITAL / entry))
    return (df["close"] - entry) * shares


# ── Plots ─────────────────────────────────────────────────────────────────────
def plot_results(df, signals, bnh_series, strat_pnl, bnh_pnl):
    x = range(len(df))

    fig = plt.figure(figsize=(14, 9))
    gs  = gridspec.GridSpec(2, 1, figure=fig, hspace=0.35)

    ax1 = fig.add_subplot(gs[0])
    ax1.plot(x, df["close"], color="#222222", linewidth=1.0, label="Precio (close)", zorder=2)
    ax1.plot(x, df["mid"],   color="#1f77b4", linewidth=0.9, linestyle="--", label="AS mid (r)", zorder=2)
    ax1.plot(x, df["bid"],   color="#2ca02c", linewidth=0.7, alpha=0.75, label="AS bid", zorder=2)
    ax1.plot(x, df["ask"],   color="#d62728", linewidth=0.7, alpha=0.75, label="AS ask", zorder=2)

    if len(signals) > 0:
        buys  = signals[signals["action"] == "buy"]
        sells = signals[signals["action"] == "sell"]
        ax1.scatter(buys["idx"],  buys["price"],  marker="^", color="#2ca02c", s=30, zorder=5, label="Entrada long")
        ax1.scatter(sells["idx"], sells["price"], marker="v", color="#d62728", s=30, zorder=5, label="Cierre (TP/SL/EOD)")

    ax1.set_title(f"{TICKER} — Bollinger-AS long-only  |  Mid: {MID_METHOD}  |  Forma: {FORM}  |  Modo: {MODE}", fontsize=10)
    ax1.set_ylabel("Precio (USD)")
    ax1.legend(fontsize=7, loc="upper left", ncol=3)
    ax1.grid(alpha=0.3)

    ax2 = fig.add_subplot(gs[1])
    ax2.plot(x, df["pnl"],  color="#f2a900", linewidth=1.2, label=f"Estrategia  (final: {strat_pnl:+.2f} USD)")
    ax2.plot(x, bnh_series, color="#627eea", linewidth=1.2, alpha=0.85, label=f"Buy & Hold  (final: {bnh_pnl:+.2f} USD)")
    ax2.axhline(0, color="gray", linewidth=0.5, linestyle="--")
    ax2.set_title("Evolución del P&L (USD)", fontsize=10)
    ax2.set_ylabel("P&L (USD)")
    ax2.set_xlabel("Barra")
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3)

    plt.suptitle(
        f"{TICKER}  |  {TIMEFRAME}  |  {START} → {END}"
        f"  |  SpreadObj={TARGET_SPREAD_FRAC:.2%}  |  MA={MA_N}  |  ATR={ATR_N}×{ATR_MULT}",
        fontsize=9)
    plt.show()


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"[*] Cargando datos: {TICKER} {TIMEFRAME} {START} → {END}")
    bars = get_bars()

    if len(bars) == 0:
        print("[!] DataFrame vacío. Verifica TICKER, TIMEFRAME y rango de fechas.")
        print("    yfinance limita: 5m→60 días, 1h→730 días.")
        raise SystemExit

    print(f"[*] {len(bars)} barras  |  {bars['timestamp'].iloc[0]} → {bars['timestamp'].iloc[-1]}")

    df, strat_pnl, signals, pnl_long = run_strategy(bars)

    if len(df) == 0:
        print("[!] Sin registros — ajusta CALIB_BARS, MID_BARS o el rango de fechas.")
        raise SystemExit

    bnh_series = run_benchmark(bars, df)
    bnh_pnl    = float(bnh_series.iloc[-1]) if len(bnh_series) > 0 else 0.0

    dP      = np.diff(df["pnl"].values)
    mdd     = float((np.maximum.accumulate(df["pnl"].values) - df["pnl"].values).max())
    bar_sec = pd.Timedelta(bars["timestamp"].diff().median()).total_seconds()
    bpy     = 252 * 6.5 * 3600 / bar_sec
    sh_ann  = dP.mean() / dP.std() * math.sqrt(bpy) if dP.std() > 0 else 0.0
    vol_ann = dP.std() * math.sqrt(bpy) if len(dP) > 1 else 0.0
    sh_per  = dP.mean() / dP.std() if dP.std() > 0 else 0.0
    vol_per = float(dP.std()) if len(dP) > 1 else 0.0
    bnh_dP  = np.diff(bnh_series.values)
    bnh_sh  = bnh_dP.mean() / bnh_dP.std() * math.sqrt(bpy) if bnh_dP.std() > 0 else 0.0
    bnh_vol = bnh_dP.std() * math.sqrt(bpy) if len(bnh_dP) > 1 else 0.0
    bnh_sh_per  = bnh_dP.mean() / bnh_dP.std() if bnh_dP.std() > 0 else 0.0
    bnh_vol_per = float(bnh_dP.std()) if len(bnh_dP) > 1 else 0.0

    buys_p  = signals[signals["action"] == "buy"]["price"].values  if len(signals) else []
    sells_p = signals[signals["action"] == "sell"]["price"].values if len(signals) else []
    n_tr    = min(len(buys_p), len(sells_p))
    wins    = sum(sells_p[i] > buys_p[i] for i in range(n_tr))

    print(f"\n{'='*52}")
    print(f"  {TICKER}  |  {TIMEFRAME}  |  {START} → {END}")
    print(f"  Mid: {MID_METHOD}  |  Forma: {FORM}  |  Modo: {MODE}")
    print(f"  SpreadObj: {TARGET_SPREAD_FRAC:.2%}  |  MA={MA_N}  |  ATR={ATR_N}×{ATR_MULT}")
    print(f"  CAPITAL={CAPITAL}  RISK_PCT={RISK_PCT:.1%}")
    print(f"{'='*52}")
    print(f"  {'Métrica':<26} {'Estrategia':>10}  {'Buy&Hold':>10}")
    print(f"  {'-'*52}")
    print(f"  {'P&L final (USD)':<26} {strat_pnl:>10.2f}  {bnh_pnl:>10.2f}")
    print(f"  {'  P&L longs (USD)':<26} {pnl_long:>10.2f}")
    print(f"  {'  Max drawdown (USD)':<26} {mdd:>10.2f}")
    print(f"  {'Winrate':<26} {wins/n_tr*100:>9.1f}%  ({wins}W/{n_tr-wins}L)" if n_tr else f"  {'Winrate':<26} {'N/A':>10}")
    print(f"  {'Sharpe (periodo)':<26} {sh_per:>10.4f}  {bnh_sh_per:>10.4f}")
    print(f"  {'Sharpe (anualizado)':<26} {sh_ann:>10.4f}  {bnh_sh:>10.4f}")
    print(f"  {'Volatilidad periodo (USD)':<26} {vol_per:>10.4f}  {bnh_vol_per:>10.4f}")
    print(f"  {'Volatilidad anual (USD)':<26} {vol_ann:>10.2f}  {bnh_vol:>10.2f}")
    print(f"  {'Max inventario (acc)':<26} {df['q'].abs().max():>10.0f}")
    print(f"  {'Spread medio (USD)':<26} {df['spread'].mean():>10.4f}")
    print(f"{'='*52}\n")

    plot_results(df, signals, bnh_series, strat_pnl, bnh_pnl)