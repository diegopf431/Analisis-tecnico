#!/usr/bin/env python3
"""
Pricing Machine — Execution Optimization for Long-Term Investors
================================================================
Modifica la sección CONFIG y presiona Play.
"""

# ══════════════════════════════════════════════════════════════════════════════
# CONFIG — edita aquí y corre el script
# ══════════════════════════════════════════════════════════════════════════════

API_KEY    = "APIKEY_AQUI"  # Completa tu API key de Alpaca (https://alpaca.markets)
API_SECRET = "APISECRET_AQUI"  # Completa tu API secret de Alpaca

TICKER = "SPY"           # Ticker del activo (ej: SPY, AAPL, QQQ, MSFT, VTI)
SIDE   = "buy"           # Dirección: "buy" o "sell"
SIZES  = [1_00, 5_00, 1_000, 5_000, 10_000]  # Tamaños de orden en USD

# ══════════════════════════════════════════════════════════════════════════════

import os
import sys
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from alpaca.data import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest, StockSnapshotRequest
from alpaca.data.timeframe import TimeFrame
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()

ET              = ZoneInfo("America/New_York")
OPEN_H, OPEN_M  = 9, 30
CLOSE_H, CLOSE_M = 16, 0
SESSION_MINUTES = 390   # 9:30–16:00 ET


# ── Data ──────────────────────────────────────────────────────────────────────

def get_client() -> StockHistoricalDataClient:
    if not API_KEY or API_KEY == "TU_API_KEY_AQUI":
        console.print("[red]❌  Completa API_KEY y API_SECRET en la sección CONFIG del script.[/red]")
        sys.exit(1)
    return StockHistoricalDataClient(API_KEY, API_SECRET)


def fetch_data(client: StockHistoricalDataClient, ticker: str):
    """Retorna (snapshot, bars_df_intradía). bars_df puede estar vacío."""
    snap = client.get_stock_snapshot(
        StockSnapshotRequest(symbol_or_symbols=ticker)
    )[ticker]

    now     = datetime.now(ET)
    open_dt = now.replace(hour=OPEN_H, minute=OPEN_M, second=0, microsecond=0)
    bars_df = pd.DataFrame()

    if now > open_dt:   # mercado ya abrió
        raw = client.get_stock_bars(StockBarsRequest(
            symbol_or_symbols=ticker,
            timeframe=TimeFrame.Minute,
            start=open_dt,
        )).df
        if not raw.empty:
            bars_df = (
                raw.xs(ticker, level="symbol")
                if isinstance(raw.index, pd.MultiIndex)
                else raw
            )

    return snap, bars_df


# ── Métricas ──────────────────────────────────────────────────────────────────

def compute_ms(snap) -> dict:
    q    = snap.latest_quote
    last = snap.latest_trade.price
    bid, ask = q.bid_price, q.ask_price
    bsz = max(q.bid_size or 1, 1)
    asz = max(q.ask_size or 1, 1)

    # Feed IEX puede tener quotes stale para ETFs que transan en NYSE Arca.
    # Si bid/ask difieren >5% del last trade, se descartan y se estiman.
    quote_ok = (
        bid and ask and bid < ask
        and abs(bid - last) / last < 0.05
        and abs(ask - last) / last < 0.05
    )
    if not quote_ok:
        spread_est = last * 0.0001   # ~1 bp, típico para ETFs líquidos US
        bid, ask   = last - spread_est / 2, last + spread_est / 2
        bsz = asz  = 1
        console.print(
            "[yellow]⚠  Quote IEX stale o fuera de rango — bid/ask estimados "
            "desde last trade. Considera actualizar a feed SIP (Alpaca Algo Trader+).[/yellow]"
        )

    mid    = (bid + ask) / 2
    spread = ask - bid
    return dict(
        bid=bid, ask=ask, mid=mid,
        spread=spread,
        spread_bps=spread / mid * 10_000,
        microprice=(asz * bid + bsz * ask) / (bsz + asz),
        ofi=(bsz - asz) / (bsz + asz),
        bsz=bsz, asz=asz, quote_ok=quote_ok,
    )


def compute_intraday(snap, bars_df: pd.DataFrame) -> dict:
    """Contexto intradía: VWAP, volatilidad, percentil de precio, sesión."""
    db  = snap.daily_bar
    now = datetime.now(ET)
    open_dt = now.replace(hour=OPEN_H, minute=OPEN_M, second=0, microsecond=0)
    session_pct = min(1.0, max(0.0,
        (now - open_dt).total_seconds() / 60 / SESSION_MINUTES
    ))

    # ADV desde sesión anterior (volumen completo del día anterior)
    prev = snap.previous_daily_bar
    adv_shares = prev.volume if prev and prev.volume else db.volume or 1

    if bars_df.empty:
        return dict(vwap=db.vwap, vol_ann=None, price_pct=None,
                    session_pct=session_pct, hi=db.high, lo=db.low,
                    adv=adv_shares, n_bars=0)

    vwap    = (bars_df["close"] * bars_df["volume"]).sum() / bars_df["volume"].sum()
    vol_ann = bars_df["close"].pct_change().std() * np.sqrt(252 * SESSION_MINUTES)
    hi, lo  = bars_df["high"].max(), bars_df["low"].min()
    last    = snap.latest_trade.price
    price_pct = (last - lo) / (hi - lo) if hi != lo else 0.5

    return dict(vwap=vwap, vol_ann=vol_ann, price_pct=price_pct,
                session_pct=session_pct, hi=hi, lo=lo,
                adv=adv_shares, n_bars=len(bars_df))


def compute_costs(ms: dict, intra: dict, sizes_usd: list, last_price: float) -> list:
    """
    Costo all-in de ejecución para cada tamaño de orden.

    Modelo: costo = ½ spread  +  impacto de mercado
    Impacto (sqrt-impact): σ_daily · √(Q/ADV) · 0.5  (bps)
    Basado en el modelo de Almgren-Chriss simplificado para órdenes retail.
    """
    adv_usd   = intra["adv"] * last_price
    vol_daily = (intra["vol_ann"] or 0.20) / np.sqrt(252)
    rows = []
    for usd in sizes_usd:
        participation = usd / adv_usd if adv_usd else 0
        half_sp = ms["spread_bps"] / 2
        impact  = vol_daily * np.sqrt(participation) * 10_000 * 0.5
        total   = half_sp + impact
        rows.append(dict(
            usd=usd, shares=usd / last_price,
            part_pct=participation * 100,
            half_sp=half_sp, impact=impact,
            total=total, cost_usd=usd * total / 10_000,
        ))
    return rows


def compute_timing(ms: dict, intra: dict, side: str, last_price: float):
    """
    Sistema de señales de timing basado en microestructura.
    Devuelve (señales, veredicto, score).
    Score > 0 → ejecutar ahora; < 0 → esperar.
    """
    signals = []
    score   = 0

    def sig(label: str, detail: str, pts: int):
        nonlocal score
        signals.append((label, detail, pts))
        score += pts

    # 1. Precio vs VWAP intradía
    diff_bps = (last_price - intra["vwap"]) / intra["vwap"] * 10_000
    if side == "buy":
        if diff_bps < 0:
            sig("✅ Bajo VWAP",  f"{diff_bps:.1f} bps bajo VWAP → favorable para compra", 2)
        else:
            sig("⚠️  Sobre VWAP", f"+{diff_bps:.1f} bps sobre VWAP → por encima del precio promedio del día", -1)
    else:
        if diff_bps > 0:
            sig("✅ Sobre VWAP",  f"+{diff_bps:.1f} bps sobre VWAP → favorable para venta", 2)
        else:
            sig("⚠️  Bajo VWAP", f"{diff_bps:.1f} bps bajo VWAP → por debajo del precio promedio del día", -1)

    # 2. Order Flow Imbalance
    ofi = ms["ofi"]
    if abs(ofi) < 0.1:
        sig("→ OFI neutro",    f"Desequilibrio={ofi:+.3f} → mercado balanceado en bid/ask", 0)
    elif (side == "buy" and ofi < 0) or (side == "sell" and ofi > 0):
        sig("✅ Flujo favorable", f"OFI={ofi:+.3f} → presión del lado contrario beneficia tu ejecución", 1)
    else:
        sig("⚠️  Flujo adverso",  f"OFI={ofi:+.3f} → presión del mismo lado encarece el spread", -1)

    # 3. Progreso de sesión
    sp = intra["session_pct"]
    if sp > 0.85:
        sig("⏰ Cierre próximo",   f"Sesión {sp:.0%} → ejecutar reduce riesgo overnight; spread se amplía al cierre", 2)
    elif sp < 0.10:
        sig("🌅 Apertura reciente", f"Sesión {sp:.0%} → spread típicamente más ancho; considera esperar ~10 min", -1)

    # 4. Percentil de precio en el rango intradía
    pp = intra["price_pct"]
    if pp is not None:
        if side == "buy":
            if pp > 0.72:
                sig("📈 Zona alta del día", f"Percentil {pp:.0%} → comprar cerca de máximo intradía", -2)
            elif pp < 0.35:
                sig("📉 Zona baja del día", f"Percentil {pp:.0%} → precio barato vs rango del día", 2)
        else:
            if pp < 0.28:
                sig("📉 Zona baja del día", f"Percentil {pp:.0%} → vender cerca de mínimo intradía", -2)
            elif pp > 0.65:
                sig("📈 Zona alta del día", f"Percentil {pp:.0%} → precio caro, buen momento para vender", 2)

    if   score >= 4:  verdict = ("EJECUTAR INMEDIATAMENTE",           "bright_green")
    elif score >= 2:  verdict = ("EJECUTAR — usar limit en microprecio", "green")
    elif score >= 0:  verdict = ("NEUTRAL — limit order en microprecio", "yellow")
    elif score >= -2: verdict = ("ESPERAR o usar limit agresivo",       "orange1")
    else:             verdict = ("ESPERAR — condiciones desfavorables",  "red")

    return signals, verdict, score


# ── Display ───────────────────────────────────────────────────────────────────

def print_report(ticker, side, snap, ms, intra, costs, timing):
    signals, (verdict_label, verdict_color), score = timing
    last = snap.latest_trade.price

    console.print()
    console.rule(f"[bold cyan]⚙  PRICING MACHINE  ·  {ticker}  ·  {side.upper()}[/bold cyan]")

    # ── Microestructura ───────────────────────────────────────────────────────
    t1 = Table(box=box.SIMPLE, show_header=False, padding=(0, 2))
    t1.add_column("", style="dim", width=20)
    t1.add_column("", style="bold white", width=28)
    t1.add_column("", style="dim", width=20)
    t1.add_column("", style="bold white")
    t1.add_row(
        "Último trade",   f"${last:.4f}",
        "Microprecio",    f"${ms['microprice']:.4f}",
    )
    t1.add_row(
        "Bid",  f"${ms['bid']:.4f}  (×{ms['bsz']:.0f} acc.)",
        "Ask",  f"${ms['ask']:.4f}  (×{ms['asz']:.0f} acc.)",
    )
    t1.add_row(
        "Spread",  f"${ms['spread']:.4f}  ({ms['spread_bps']:.2f} bps)",
        "OFI",     f"{ms['ofi']:+.3f}  ({'buy pressure' if ms['ofi'] > 0 else 'sell pressure'})",
    )
    t1.add_row(
        "VWAP intradía",   f"${intra['vwap']:.4f}",
        "Vol. anualizada", f"{intra['vol_ann']:.1%}" if intra["vol_ann"] else "N/A (mercado cerrado)",
    )
    console.print(Panel(t1, title="[bold]📊 Microestructura[/bold]", border_style="cyan"))

    # ── Contexto intradía ─────────────────────────────────────────────────────
    t2 = Table(box=box.SIMPLE, show_header=False, padding=(0, 2))
    t2.add_column("", style="dim", width=20)
    t2.add_column("", style="bold white", width=28)
    t2.add_column("", style="dim", width=20)
    t2.add_column("", style="bold white")
    pp_str = f"{intra['price_pct']:.0%} del rango" if intra["price_pct"] is not None else "N/A"
    t2.add_row(
        "Rango día",   f"${intra['lo']:.2f} — ${intra['hi']:.2f}",
        "Percentil precio", pp_str,
    )
    t2.add_row(
        "Sesión transcurrida", f"{intra['session_pct']:.0%}",
        "ADV (día anterior)",  f"{intra['adv']:,.0f} acc.",
    )
    t2.add_row("Barras intradía", str(intra["n_bars"]), "", "")
    console.print(Panel(t2, title="[bold]📈 Contexto Intradía[/bold]", border_style="blue"))

    # ── Tabla de costos ───────────────────────────────────────────────────────
    t3 = Table(box=box.SIMPLE, padding=(0, 2))
    t3.add_column("Orden (USD)",  justify="right", style="cyan")
    t3.add_column("Acciones",     justify="right")
    t3.add_column("% ADV",        justify="right")
    t3.add_column("½ Spread",     justify="right")
    t3.add_column("Impacto mkt",  justify="right")
    t3.add_column("Total (bps)",  justify="right")
    t3.add_column("Costo ($)",    justify="right", style="bold")
    for r in costs:
        col = "green" if r["total"] < 5 else ("yellow" if r["total"] < 15 else "red")
        t3.add_row(
            f"${r['usd']:,.0f}",
            f"{r['shares']:.1f}",
            f"{r['part_pct']:.4f}%",
            f"{r['half_sp']:.2f}",
            f"{r['impact']:.2f}",
            f"[{col}]{r['total']:.2f}[/{col}]",
            f"[{col}]${r['cost_usd']:.2f}[/{col}]",
        )
    console.print(Panel(t3,
        title="[bold]💰 Costo de Ejecución por Tamaño  [dim](½ spread + impacto sqrt)[/dim][/bold]",
        border_style="green"))

    # ── Señales de timing ─────────────────────────────────────────────────────
    t4 = Table(box=box.SIMPLE, show_header=False, padding=(0, 2))
    t4.add_column("", style="bold", width=26)
    t4.add_column("", style="dim white")
    for label, detail, _ in signals:
        t4.add_row(label, detail)
    t4.add_row("[dim]Score total[/dim]",
               f"{score:+d} punto{'s' if abs(score) != 1 else ''}")
    console.print(Panel(t4, title="[bold]⏳ Señales de Timing[/bold]", border_style="magenta"))

    # ── Recomendación final ───────────────────────────────────────────────────
    mp      = ms["microprice"]
    limit   = ms["bid"] + ms["spread"] * 0.25 if side == "buy" else ms["ask"] - ms["spread"] * 0.25
    saving  = ms["spread_bps"] / 2
    ref_usd = costs[0]["usd"]
    body = "\n".join([
        f"[bold {verdict_color}]{verdict_label}[/bold {verdict_color}]",
        "",
        f"  Precio justo (microprecio)   →  [bold white]${mp:.4f}[/bold white]",
        f"  Limit order sugerido         →  [bold white]${limit:.4f}[/bold white]"
        f"  [dim]({'25% spread sobre bid' if side == 'buy' else '25% spread bajo ask'})[/dim]",
        f"  Ahorro est. vs market order  →  [bold green]{saving:.1f} bps[/bold green]"
        f"  [dim](≈ ${ref_usd * saving / 10_000:.2f} por cada ${ref_usd:,.0f} invertidos)[/dim]",
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

    client = get_client()

    with console.status(f"[cyan]Fetching {ticker} data...[/cyan]"):
        snap, bars_df = fetch_data(client, ticker)

    ms     = compute_ms(snap)
    intra  = compute_intraday(snap, bars_df)
    last   = snap.latest_trade.price
    costs  = compute_costs(ms, intra, SIZES, last)
    timing = compute_timing(ms, intra, side, last)

    print_report(ticker, side, snap, ms, intra, costs, timing)