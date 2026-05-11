#!/usr/bin/env python3
"""
Pricing Machine Chile — Execution Optimization · Bolsa de Santiago
===================================================================
Modifica la sección CONFIG y presiona Play.

Tickers comunes:
  Acciones:  FALABELLA.SN  SQM-B.SN  COPEC.SN  BSANTANDER.SN
             CMPC.SN  ENELCHILE.SN  CENCOSUD.SN  LTM.SN
  ETFs IPSA: No hay ETF directo del IPSA en .SN — la mejor aproximación
             es ECH (iShares MSCI Chile) en NYSE, que funciona con
             el pricing_machine.py del mercado US.

Limitaciones vs versión US:
  - Sin OFI ni microprecio real (Yahoo no entrega bid/ask size)
  - VWAP intradía estimado (barras de minuto poco confiables en .SN)
  - Quotes pueden tener delay de 15-20 min según el instrumento
"""

# ══════════════════════════════════════════════════════════════════════════════
# CONFIG — edita aquí y presiona Play
# ══════════════════════════════════════════════════════════════════════════════

TICKER = "CFIETFIPSA.SN"   # Ticker Yahoo Finance (siempre con sufijo .SN)
SIDE   = "buy"            # "buy" o "sell"
SIZES  = [50_000, 100_000, 500_000, 1_000_000, 5_000_000]  # Tamaños en CLP

# ══════════════════════════════════════════════════════════════════════════════

import sys
from datetime import datetime, date
from zoneinfo import ZoneInfo

import numpy as np
import yfinance as yf
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()

CLT            = ZoneInfo("America/Santiago")
OPEN_H, OPEN_M  = 9, 30
SESSION_MINUTES = 480   # 9:30–17:30 CLT


# ── Data ──────────────────────────────────────────────────────────────────────

def fetch_data(ticker: str):
    t    = yf.Ticker(ticker)
    info = t.fast_info

    # Barras intradía (1 min desde apertura)
    bars = t.history(period="1d", interval="1m")

    return t, info, bars


def compute_ms(t, info) -> dict:
    raw  = t.info  # full info para bid/ask
    bid  = raw.get("bid") or 0
    ask  = raw.get("ask") or 0
    last = info.last_price or raw.get("regularMarketPrice", 0)

    # Si bid/ask no disponibles, aproximar desde last con spread típico mercado CL
    if not bid or not ask or bid == ask:
        spread_est = last * 0.003   # ~30 bps, spread típico Bolsa de Santiago
        bid = last - spread_est / 2
        ask = last + spread_est / 2

    mid    = (bid + ask) / 2
    spread = ask - bid

    return dict(
        bid=bid, ask=ask, mid=mid, last=last,
        spread=spread,
        spread_bps=spread / mid * 10_000 if mid else 0,
        microprice=mid,         # sin bid/ask size, mid es la mejor aproximación
        ofi=None,               # no disponible en Yahoo
        bid_size_available=False,
    )


def compute_intraday(t, info, bars) -> dict:
    raw     = t.info
    now     = datetime.now(CLT)
    open_dt = now.replace(hour=OPEN_H, minute=OPEN_M, second=0, microsecond=0)
    session_pct = min(1.0, max(0.0,
        (now - open_dt).total_seconds() / 60 / SESSION_MINUTES
    ))

    hi   = info.day_high  or raw.get("dayHigh", 0)
    lo   = info.day_low   or raw.get("dayLow", 0)
    last = info.last_price or raw.get("regularMarketPrice", 0)
    adv  = raw.get("averageVolume") or raw.get("averageDailyVolume10Day") or 1

    price_pct = (last - lo) / (hi - lo) if hi and lo and hi != lo else None

    # VWAP: desde barras intradía si hay datos, si no desde precio del día
    if not bars.empty and "Volume" in bars.columns and bars["Volume"].sum() > 0:
        vwap    = (bars["Close"] * bars["Volume"]).sum() / bars["Volume"].sum()
        returns = bars["Close"].pct_change().dropna()
        vol_ann = returns.std() * np.sqrt(252 * SESSION_MINUTES) if len(returns) > 1 else None
        n_bars  = len(bars)
    else:
        vwap    = raw.get("regularMarketDayHigh", hi + lo) / 2 if hi and lo else last
        vol_ann = None
        n_bars  = 0

    prev_close = raw.get("previousClose") or raw.get("regularMarketPreviousClose")
    currency   = raw.get("currency", "CLP")

    return dict(
        vwap=vwap, vol_ann=vol_ann, price_pct=price_pct,
        session_pct=session_pct, hi=hi, lo=lo, adv=adv,
        n_bars=n_bars, prev_close=prev_close, currency=currency,
    )


def compute_costs(ms: dict, intra: dict, sizes_clp: list) -> list:
    last    = ms["last"]
    adv_clp = intra["adv"] * last if intra["adv"] else 1
    vol_daily = (intra["vol_ann"] or 0.25) / np.sqrt(252)

    rows = []
    for clp in sizes_clp:
        participation = clp / adv_clp if adv_clp else 0
        half_sp = ms["spread_bps"] / 2
        impact  = vol_daily * np.sqrt(participation) * 10_000 * 0.5
        # Comisión Zesty: 0.15% + IVA ≈ 0.1785%  → 178.5 bps
        comision = 178.5
        total    = half_sp + impact + comision
        rows.append(dict(
            clp=clp, shares=clp / last if last else 0,
            part_pct=participation * 100,
            half_sp=half_sp, impact=impact,
            comision=comision, total=total,
            cost_clp=clp * total / 10_000,
        ))
    return rows


def compute_timing(ms: dict, intra: dict, side: str) -> tuple:
    last = ms["last"]
    signals = []
    score   = 0

    def sig(label, detail, pts):
        nonlocal score
        signals.append((label, detail, pts))
        score += pts

    # 1. Precio vs VWAP
    if intra["vwap"] and intra["vwap"] != last:
        diff_bps = (last - intra["vwap"]) / intra["vwap"] * 10_000
        if side == "buy":
            if diff_bps < 0:
                sig("✅ Bajo VWAP",   f"{diff_bps:.1f} bps bajo VWAP → favorable para compra", 2)
            else:
                sig("⚠️  Sobre VWAP", f"+{diff_bps:.1f} bps sobre VWAP → precio sobre promedio del día", -1)
        else:
            if diff_bps > 0:
                sig("✅ Sobre VWAP",  f"+{diff_bps:.1f} bps sobre VWAP → favorable para venta", 2)
            else:
                sig("⚠️  Bajo VWAP", f"{diff_bps:.1f} bps bajo VWAP → precio bajo promedio del día", -1)

    # 2. Precio vs cierre anterior
    if intra["prev_close"]:
        gap_bps = (last - intra["prev_close"]) / intra["prev_close"] * 10_000
        if side == "buy" and gap_bps < -100:
            sig("📉 Gap bajista",  f"Precio {gap_bps:.0f} bps bajo cierre anterior → oportunidad de entrada", 1)
        elif side == "sell" and gap_bps > 100:
            sig("📈 Gap alcista",  f"Precio +{gap_bps:.0f} bps sobre cierre anterior → buen momento de venta", 1)

    # 3. Progreso de sesión
    sp = intra["session_pct"]
    if sp > 0.88:
        sig("⏰ Cierre próximo",    f"Sesión {sp:.0%} → liquidez cae al cierre; ejecutar reduce riesgo overnight", 2)
    elif sp < 0.08:
        sig("🌅 Apertura reciente", f"Sesión {sp:.0%} → spread ancho en apertura; considera esperar ~15 min", -1)

    # 4. Percentil intradía
    pp = intra["price_pct"]
    if pp is not None:
        if side == "buy":
            if pp > 0.75:
                sig("📈 Zona alta del día", f"Percentil {pp:.0%} → comprando cerca del máximo intradía", -2)
            elif pp < 0.30:
                sig("📉 Zona baja del día", f"Percentil {pp:.0%} → precio barato dentro del rango del día", 2)
        else:
            if pp < 0.25:
                sig("📉 Zona baja del día", f"Percentil {pp:.0%} → vendiendo cerca del mínimo intradía", -2)
            elif pp > 0.70:
                sig("📈 Zona alta del día", f"Percentil {pp:.0%} → precio caro, buen momento de venta", 2)

    if   score >= 4:  verdict = ("EJECUTAR INMEDIATAMENTE",              "bright_green")
    elif score >= 2:  verdict = ("EJECUTAR — usar limit en bid/ask medio","green")
    elif score >= 0:  verdict = ("NEUTRAL — limit order en mid",          "yellow")
    elif score >= -2: verdict = ("ESPERAR o usar limit agresivo",         "orange1")
    else:             verdict = ("ESPERAR — condiciones desfavorables",   "red")

    return signals, verdict, score


# ── Display ───────────────────────────────────────────────────────────────────

def fmt_clp(n: float) -> str:
    return f"${n:,.0f} CLP"

def fmt_price(p: float, currency: str) -> str:
    return f"${p:,.2f} {currency}"


def print_report(ticker, side, ms, intra, costs, timing):
    signals, (verdict_label, verdict_color), score = timing
    cur = intra["currency"]

    console.print()
    console.rule(f"[bold cyan]⚙  PRICING MACHINE CL  ·  {ticker}  ·  {side.upper()}[/bold cyan]")

    # ── Microestructura ───────────────────────────────────────────────────────
    t1 = Table(box=box.SIMPLE, show_header=False, padding=(0, 2))
    t1.add_column("", style="dim", width=22)
    t1.add_column("", style="bold white", width=26)
    t1.add_column("", style="dim", width=22)
    t1.add_column("", style="bold white")
    t1.add_row(
        "Último precio",  fmt_price(ms["last"], cur),
        "Mid (aprox.)",   fmt_price(ms["mid"], cur),
    )
    t1.add_row(
        "Bid (estimado)",  fmt_price(ms["bid"], cur),
        "Ask (estimado)",  fmt_price(ms["ask"], cur),
    )
    t1.add_row(
        "Spread",  f"{fmt_price(ms['spread'], cur)}  ({ms['spread_bps']:.1f} bps)",
        "OFI",     "[dim]No disponible (Yahoo no entrega book size)[/dim]",
    )
    t1.add_row(
        "VWAP intradía",   fmt_price(intra["vwap"], cur),
        "Vol. anualizada", f"{intra['vol_ann']:.1%}" if intra["vol_ann"] else "[dim]Estimada (sin barras de minuto)[/dim]",
    )
    t1.add_row(
        "Cierre anterior", fmt_price(intra["prev_close"], cur) if intra["prev_close"] else "N/A",
        "", "",
    )
    console.print(Panel(t1,
        title="[bold]📊 Microestructura[/bold]  [dim](bid/ask estimados — Yahoo no entrega book real)[/dim]",
        border_style="cyan"))

    # ── Contexto intradía ─────────────────────────────────────────────────────
    t2 = Table(box=box.SIMPLE, show_header=False, padding=(0, 2))
    t2.add_column("", style="dim", width=22)
    t2.add_column("", style="bold white", width=26)
    t2.add_column("", style="dim", width=22)
    t2.add_column("", style="bold white")
    pp_str = f"{intra['price_pct']:.0%} del rango" if intra["price_pct"] is not None else "N/A"
    t2.add_row(
        "Rango del día",    f"{fmt_price(intra['lo'], cur)} — {fmt_price(intra['hi'], cur)}",
        "Percentil precio", pp_str,
    )
    t2.add_row(
        "Sesión transcurrida", f"{intra['session_pct']:.0%}  (9:30–17:30 CLT)",
        "ADV (prom. 10d)",      f"{intra['adv']:,.0f} acc.",
    )
    t2.add_row("Barras intradía", str(intra["n_bars"]), "", "")
    console.print(Panel(t2, title="[bold]📈 Contexto Intradía[/bold]", border_style="blue"))

    # ── Tabla de costos ───────────────────────────────────────────────────────
    t3 = Table(box=box.SIMPLE, padding=(0, 2))
    t3.add_column("Orden (CLP)",    justify="right", style="cyan")
    t3.add_column("Acciones",       justify="right")
    t3.add_column("% ADV",          justify="right")
    t3.add_column("½ Spread",       justify="right")
    t3.add_column("Impacto mkt",    justify="right")
    t3.add_column("Comisión Zesty", justify="right")
    t3.add_column("Total (bps)",    justify="right")
    t3.add_column("Costo ($CLP)",   justify="right", style="bold")
    for r in costs:
        col = "green" if r["total"] < 200 else ("yellow" if r["total"] < 250 else "red")
        t3.add_row(
            fmt_clp(r["clp"]),
            f"{r['shares']:.1f}",
            f"{r['part_pct']:.3f}%",
            f"{r['half_sp']:.1f}",
            f"{r['impact']:.1f}",
            f"178.5",
            f"[{col}]{r['total']:.1f}[/{col}]",
            f"[{col}]{fmt_clp(r['cost_clp'])}[/{col}]",
        )
    console.print(Panel(t3,
        title="[bold]💰 Costo de Ejecución  [dim](½ spread + impacto sqrt + comisión Zesty 0.15%+IVA)[/dim][/bold]",
        border_style="green"))

    # ── Señales ───────────────────────────────────────────────────────────────
    t4 = Table(box=box.SIMPLE, show_header=False, padding=(0, 2))
    t4.add_column("", style="bold", width=26)
    t4.add_column("", style="dim white")
    for label, detail, _ in signals:
        t4.add_row(label, detail)
    t4.add_row("[dim]Score total[/dim]", f"{score:+d} punto{'s' if abs(score) != 1 else ''}")
    console.print(Panel(t4, title="[bold]⏳ Señales de Timing[/bold]", border_style="magenta"))

    # ── Recomendación ─────────────────────────────────────────────────────────
    limit   = ms["bid"] + ms["spread"] * 0.25 if side == "buy" else ms["ask"] - ms["spread"] * 0.25
    saving  = ms["spread_bps"] / 2
    ref_clp = costs[0]["clp"]
    body = "\n".join([
        f"[bold {verdict_color}]{verdict_label}[/bold {verdict_color}]",
        "",
        f"  Precio mid (referencia)     →  [bold white]{fmt_price(ms['mid'], cur)}[/bold white]",
        f"  Limit order sugerido        →  [bold white]{fmt_price(limit, cur)}[/bold white]"
        f"  [dim]({'25% spread sobre bid' if side == 'buy' else '25% spread bajo ask'})[/dim]",
        f"  Ahorro est. vs market order →  [bold green]{saving:.1f} bps[/bold green]"
        f"  [dim](≈ {fmt_clp(ref_clp * saving / 10_000)} por cada {fmt_clp(ref_clp)})[/dim]",
        "",
        f"  [dim]⚠  Spread y microprecio son estimados. No hay orderbook real disponible para .SN[/dim]",
    ])
    console.print(Panel(body, title="[bold]🎯 Recomendación[/bold]", border_style=verdict_color))
    console.print()


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    ticker = TICKER.upper()
    side   = SIDE.lower()

    if side not in ("buy", "sell"):
        console.print("[red]❌  SIDE debe ser 'buy' o 'sell'.[/red]")
        sys.exit(1)

    if not ticker.endswith(".SN"):
        console.print("[yellow]⚠  El ticker no termina en .SN — ¿es realmente un activo de la Bolsa de Santiago?[/yellow]")

    with console.status(f"[cyan]Fetching {ticker} data (Yahoo Finance)...[/cyan]"):
        t, info, bars = fetch_data(ticker)

    ms     = compute_ms(t, info)
    intra  = compute_intraday(t, info, bars)
    costs  = compute_costs(ms, intra, SIZES)
    timing = compute_timing(ms, intra, side)

    print_report(ticker, side, ms, intra, costs, timing)