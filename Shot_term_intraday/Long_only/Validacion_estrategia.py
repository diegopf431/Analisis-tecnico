# multi_day.py
# ──────────────────────────────────────────────────────────────────────────────
# Análisis multi-día: ejecuta el backtest día por día y compara la estrategia
# contra Buy & Hold en cada sesión intraday.
# Importa run_strategy y run_benchmark desde backtest.py directamente.
# Requiere los mismos parámetros que backtest.py (se heredan automáticamente).
# ──────────────────────────────────────────────────────────────────────────────

import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import yfinance as yf

from backtest import (run_strategy, run_benchmark,
                      TICKER, TIMEFRAME, CAPITAL,
                      CALIB_BARS, MID_BARS, MA_N)
from estrategia import MODE

MIN_BARS_PER_DAY = max(CALIB_BARS, MID_BARS, MA_N) + 10   # mínimo para tener señales


def _sharpe(pnl_series):
    d = np.diff(pnl_series)
    return d.mean() / d.std() if len(d) > 1 and d.std() > 0 else 0.0


def _mdd(pnl_series):
    arr = np.array(pnl_series)
    return float((np.maximum.accumulate(arr) - arr).max())


def _bar_seconds(bars):
    return pd.Timedelta(bars["timestamp"].diff().median()).total_seconds()


def _bpy(bar_sec):
    return 252 * 6.5 * 3600 / bar_sec


# ── Datos ─────────────────────────────────────────────────────────────────────
def load_all():
    """Descarga todo el histórico disponible para 1m (últimos 8 días en yfinance)."""
    df = yf.download(TICKER, period="8d", interval=TIMEFRAME, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df[["Open","High","Low","Close","Volume"]].copy()
    df.columns = ["open","high","low","close","volume"]
    df.index.name = "timestamp"
    df = df.dropna(subset=["open","high","low","close"]).reset_index()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


def split_days(df):
    df["date"] = df["timestamp"].dt.date
    return [g.reset_index(drop=True) for _, g in df.groupby("date") if len(g) >= MIN_BARS_PER_DAY]


# ── Métricas por día ──────────────────────────────────────────────────────────
def day_metrics(bars):
    df, strat_pnl, signals, _ = run_strategy(bars)
    if len(df) == 0:
        return None

    bnh     = run_benchmark(bars, df)
    bsec    = _bar_seconds(bars)
    bpy     = _bpy(bsec)
    ann     = math.sqrt(bpy)

    sp      = _sharpe(df["pnl"].values)
    bnh_sp  = _sharpe(bnh.values)
    vol     = np.diff(df["pnl"].values).std()
    bnh_vol = np.diff(bnh.values).std()
    mdd     = _mdd(df["pnl"].values)

    buys_p  = signals[signals["action"] == "buy"]["price"].values  if len(signals) else []
    sells_p = signals[signals["action"] == "sell"]["price"].values if len(signals) else []
    n_tr    = min(len(buys_p), len(sells_p))
    wr      = sum(sells_p[i] > buys_p[i] for i in range(n_tr)) / n_tr if n_tr else float("nan")

    return {
        "date":         str(bars["timestamp"].iloc[0].date()),
        "strat_pnl":    strat_pnl,
        "bnh_pnl":      float(bnh.iloc[-1]),
        "sharpe":       sp,
        "sharpe_ann":   sp * ann,
        "bnh_sharpe":   bnh_sp,
        "bnh_sharpe_ann": bnh_sp * ann,
        "vol":          vol,
        "vol_ann":      vol * ann,
        "bnh_vol":      bnh_vol,
        "bnh_vol_ann":  bnh_vol * ann,
        "mdd":          mdd,
        "winrate":      wr,
        "n_trades":     n_tr,
        "n_bars":       len(df),
        "pnl_curve":    df["pnl"].values,
        "bnh_curve":    bnh.values,
    }


# ── Plots ─────────────────────────────────────────────────────────────────────
def plot_all(rows):
    dates     = [r["date"] for r in rows]
    s_pnl     = [r["strat_pnl"] for r in rows]
    b_pnl     = [r["bnh_pnl"]   for r in rows]
    s_sh      = [r["sharpe_ann"] for r in rows]
    b_sh      = [r["bnh_sharpe_ann"] for r in rows]
    s_vol     = [r["vol_ann"]    for r in rows]
    b_vol     = [r["bnh_vol_ann"] for r in rows]
    wrs       = [r["winrate"] * 100 for r in rows if not math.isnan(r["winrate"])]
    wr_dates  = [r["date"]      for r in rows if not math.isnan(r["winrate"])]
    mdds      = [r["mdd"]       for r in rows]
    diff_pnl  = [r["strat_pnl"] - r["bnh_pnl"] for r in rows]
    x         = range(len(dates))

    fig = plt.figure(figsize=(16, 20))
    gs  = gridspec.GridSpec(4, 2, figure=fig, hspace=0.55, wspace=0.35)

    # 1. PnL diario por día
    ax1 = fig.add_subplot(gs[0, :])
    ax1.bar([i - 0.2 for i in x], s_pnl, width=0.38, color="#f2a900", label="Estrategia")
    ax1.bar([i + 0.2 for i in x], b_pnl, width=0.38, color="#627eea", alpha=0.8, label="Buy & Hold")
    ax1.axhline(0, color="gray", linewidth=0.6, linestyle="--")
    ax1.set_xticks(list(x)); ax1.set_xticklabels(dates, rotation=45, ha="right", fontsize=7)
    ax1.set_title("P&L diario: Estrategia vs Buy & Hold", fontsize=10)
    ax1.set_ylabel("P&L (USD)"); ax1.legend(fontsize=8); ax1.grid(alpha=0.25, axis="y")

    # 2. Diferencia PnL (estrategia − B&H)
    ax2 = fig.add_subplot(gs[1, :])
    colors = ["#2ca02c" if v >= 0 else "#d62728" for v in diff_pnl]
    ax2.bar(x, diff_pnl, color=colors)
    ax2.axhline(0, color="gray", linewidth=0.6, linestyle="--")
    ax2.set_xticks(list(x)); ax2.set_xticklabels(dates, rotation=45, ha="right", fontsize=7)
    ax2.set_title("Diferencia diaria P&L (Estrategia − B&H)", fontsize=10)
    ax2.set_ylabel("ΔP&L (USD)"); ax2.grid(alpha=0.25, axis="y")

    # 3. Sharpe anualizado
    ax3 = fig.add_subplot(gs[2, 0])
    ax3.plot(dates, s_sh, color="#f2a900", marker="o", markersize=4, label="Estrategia")
    ax3.plot(dates, b_sh, color="#627eea", marker="o", markersize=4, alpha=0.8, label="B&H")
    ax3.axhline(0, color="gray", linewidth=0.6, linestyle="--")
    ax3.set_xticks(dates[::max(1, len(dates)//5)])
    ax3.tick_params(axis="x", rotation=30, labelsize=7)
    ax3.set_title("Sharpe anualizado diario", fontsize=10)
    ax3.legend(fontsize=8); ax3.grid(alpha=0.25)

    # 4. Volatilidad anualizada
    ax4 = fig.add_subplot(gs[2, 1])
    ax4.plot(dates, s_vol, color="#f2a900", marker="o", markersize=4, label="Estrategia")
    ax4.plot(dates, b_vol, color="#627eea", marker="o", markersize=4, alpha=0.8, label="B&H")
    ax4.set_xticks(dates[::max(1, len(dates)//5)])
    ax4.tick_params(axis="x", rotation=30, labelsize=7)
    ax4.set_title("Volatilidad anualizada diaria (USD/barra × √bpy)", fontsize=10)
    ax4.legend(fontsize=8); ax4.grid(alpha=0.25)

    # 5. Winrate diario
    ax5 = fig.add_subplot(gs[3, 0])
    ax5.bar(range(len(wr_dates)), wrs, color="#9467bd")
    ax5.axhline(50, color="gray", linewidth=0.6, linestyle="--")
    ax5.set_xticks(range(len(wr_dates)))
    ax5.set_xticklabels(wr_dates, rotation=45, ha="right", fontsize=7)
    ax5.set_title("Winrate diario (%)", fontsize=10)
    ax5.set_ylabel("%"); ax5.set_ylim(0, 105); ax5.grid(alpha=0.25, axis="y")

    # 6. Max Drawdown diario
    ax6 = fig.add_subplot(gs[3, 1])
    ax6.bar(x, mdds, color="#d62728", alpha=0.7)
    ax6.set_xticks(list(x)); ax6.set_xticklabels(dates, rotation=45, ha="right", fontsize=7)
    ax6.set_title("Max Drawdown diario (USD)", fontsize=10)
    ax6.set_ylabel("MDD (USD)"); ax6.grid(alpha=0.25, axis="y")

    plt.suptitle(
        f"{TICKER} {TIMEFRAME} — Análisis multi-día | Modo: {MODE} | Capital: ${CAPITAL:,}",
        fontsize=11, y=1.01)
    plt.tight_layout()
    plt.savefig("multi_day_analysis.png", dpi=130, bbox_inches="tight")
    plt.show()
    print("[*] Gráfico guardado: multi_day_analysis.png")


# ── Tabla de resumen ──────────────────────────────────────────────────────────
def print_summary(rows):
    n = len(rows)
    s_days   = sum(1 for r in rows if r["strat_pnl"] > 0)
    b_days   = sum(1 for r in rows if r["bnh_pnl"]   > 0)
    wr_vals  = [r["winrate"] for r in rows if not math.isnan(r["winrate"])]

    def avg(key): return sum(r[key] for r in rows) / n
    def moda_sign(key):
        pos = sum(1 for r in rows if r[key] > 0)
        return "positivo" if pos > n / 2 else ("negativo" if pos < n / 2 else "neutro")

    print(f"\n{'='*62}")
    print(f"  RESUMEN MULTI-DÍA  |  {TICKER} {TIMEFRAME}  |  {n} días operados")
    print(f"  Modo: {MODE}  |  Capital: ${CAPITAL:,}")
    print(f"{'='*62}")
    print(f"  {'Métrica':<34} {'Estrategia':>12}  {'B&H':>12}")
    print(f"  {'-'*62}")
    print(f"  {'Días con P&L positivo':<34} {s_days:>11}  {b_days:>12}")
    print(f"  {'Retorno diario promedio (USD)':<34} {avg('strat_pnl'):>+12.2f}  {avg('bnh_pnl'):>+12.2f}")
    print(f"  {'Sharpe diario promedio (ann)':<34} {avg('sharpe_ann'):>12.4f}  {avg('bnh_sharpe_ann'):>12.4f}")
    print(f"  {'Vol diaria promedio (ann, USD)':<34} {avg('vol_ann'):>12.4f}  {avg('bnh_vol_ann'):>12.4f}")
    print(f"  {'Vol diaria promedio (USD)':<34} {avg('vol'):>12.4f}  {avg('bnh_vol'):>12.4f}")
    print(f"  {'MDD promedio (USD)':<34} {avg('mdd'):>12.4f}")
    print(f"  {'Winrate promedio (%)':<34} {sum(wr_vals)/len(wr_vals)*100 if wr_vals else 0:>11.1f}%")
    print(f"  {'Trades promedio por día':<34} {avg('n_trades'):>12.1f}")
    print(f"  {'-'*62}")
    print(f"  {'Diferencia retorno promedio':<34} {avg('strat_pnl')-avg('bnh_pnl'):>+12.2f} USD/día")
    print(f"  {'Diferencia Sharpe promedio':<34} {avg('sharpe_ann')-avg('bnh_sharpe_ann'):>+12.4f}")
    print(f"  {'Diferencia vol promedio':<34} {avg('vol_ann')-avg('bnh_vol_ann'):>+12.4f}")
    print(f"  {'Moda dirección P&L estrategia':<34} {moda_sign('strat_pnl'):>12}")
    print(f"  {'Moda dirección diferencia PnL':<34} {moda_sign('strat_pnl'):>12}")
    print(f"{'='*62}\n")

    # Tabla por día
    print(f"  {'Fecha':<12} {'PnL Est':>9} {'PnL B&H':>9} {'ΔPnL':>8} "
          f"{'Sh.Est':>7} {'Sh.B&H':>7} {'WR%':>6} {'Trades':>7}")
    print(f"  {'-'*72}")
    for r in rows:
        wr_str = f"{r['winrate']*100:5.1f}%" if not math.isnan(r["winrate"]) else "  N/A "
        print(f"  {r['date']:<12} {r['strat_pnl']:>+9.2f} {r['bnh_pnl']:>+9.2f} "
              f"{r['strat_pnl']-r['bnh_pnl']:>+8.2f} "
              f"{r['sharpe_ann']:>7.3f} {r['bnh_sharpe_ann']:>7.3f} "
              f"{wr_str:>6} {r['n_trades']:>7}")
    total_strat = sum(r["strat_pnl"] for r in rows)
    total_bnh   = sum(r["bnh_pnl"]   for r in rows)
    print(f"  {'-'*72}")
    print(f"  {'TOTAL':<12} {total_strat:>+9.2f} {total_bnh:>+9.2f} "
          f"{total_strat-total_bnh:>+8.2f}")

    def fmt_pct(v): return f"{v:+.1f}%" if not math.isnan(v) else "N/A"
    bnh_avg_pnl    = avg('bnh_pnl')
    bnh_avg_sh_ann = avg('bnh_sharpe_ann')
    bnh_avg_sh     = avg('bnh_sharpe')
    pnl_pct_avg   = avg('strat_pnl')  / bnh_avg_pnl    * 100 if bnh_avg_pnl    != 0 else float('nan')
    pnl_pct_total = total_strat       / total_bnh       * 100 if total_bnh      != 0 else float('nan')
    sh_pct_ann    = avg('sharpe_ann') / bnh_avg_sh_ann  * 100 if bnh_avg_sh_ann != 0 else float('nan')
    sh_pct_noann  = avg('sharpe')     / bnh_avg_sh      * 100 if bnh_avg_sh     != 0 else float('nan')
    print(f"  {'-'*72}")
    bnh_avg_vol_ann = avg('bnh_vol_ann')
    bnh_avg_vol     = avg('bnh_vol')
    vol_pct_ann   = avg('vol_ann') / bnh_avg_vol_ann * 100 if bnh_avg_vol_ann != 0 else float('nan')
    vol_pct_noann = avg('vol')     / bnh_avg_vol     * 100 if bnh_avg_vol     != 0 else float('nan')
    print(f"  {'% PnL Estrat./B&H (prom/día | total)':<36}  {fmt_pct(pnl_pct_avg):>10}  {fmt_pct(pnl_pct_total):>10}")
    print(f"  {'% Vol Estrat./B&H (anualiz | cruda)':<36}  {fmt_pct(vol_pct_ann):>10}  {fmt_pct(vol_pct_noann):>10}")
    print(f"  {'% Sharpe Estrat./B&H (anualiz | crudo)':<36}  {fmt_pct(sh_pct_ann):>10}  {fmt_pct(sh_pct_noann):>10}")
    print(f"{'='*62}\n")


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"[*] Descargando {TICKER} {TIMEFRAME} (últimos 30 días)...")
    all_bars = load_all()
    days     = split_days(all_bars)
    print(f"[*] {len(days)} días con ≥{MIN_BARS_PER_DAY} barras disponibles")

    rows = []
    for bars in days:
        date = str(bars["timestamp"].iloc[0].date())
        print(f"    → {date} ({len(bars)} barras)...", end=" ")
        try:
            m = day_metrics(bars)
            if m:
                rows.append(m)
                print(f"OK  PnL={m['strat_pnl']:+.2f} vs B&H={m['bnh_pnl']:+.2f}")
            else:
                print("sin registros")
        except Exception as e:
            print(f"error: {e}")

    if not rows:
        print("[!] Sin días válidos — verifica configuración en backtest.py")
        raise SystemExit

    print_summary(rows)
    plot_all(rows)