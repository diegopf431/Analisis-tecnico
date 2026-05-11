# estrategia.py
# ──────────────────────────────────────────────────────────────────────────────
# Estrategia Bollinger-AS / Tendencia-AS (long-only, seleccionable por MODE).
#
# MODE = "bollinger"  — mean-reversion:
#   Entrada : prev_close cruza bid hacia arriba (below_bid + prev_close >= bid)
#   TP      : prev_close cruza ask hacia abajo  (above_ask + prev_close <= ask)
#   SL      : prev_close <= bid mientras hay posición
#
# MODE = "tendency"   — momentum / breakout:
#   Entrada : prev_close cruza ask hacia arriba (above_ask_entry + prev_close >= ask)
#   TP      : 2 cierres consecutivos bajo el ask tras la entrada
#   SL      : prev_close <= bid mientras hay posición
#
# Filtro MA (ambos modos): solo entradas si prev_close >= MA_N.
# Sizing ATR: qty = floor(capital × risk_pct / (ATR × atr_mult)), cap inv_limit.
#
# Nota ejecución: señal detectada en prev_close (barra i-1),
#   ejecución al open de barra i (row["open"] en el loop de backtest).
#   Para probar ejecución en barra i+1: en backtest.py, dentro del loop
#   acumular las señales en una lista pendiente y ejecutarlas en la iteración
#   siguiente, usando el open de esa barra como precio.
# ──────────────────────────────────────────────────────────────────────────────

import random
from avellaneda_stoikov import avellaneda_stoikov
from mid import compute_mid

MID_METHOD = "vwap"
MODE       = "tendency"    # "bollinger" | "tendency"


def _bars_to_trades(window_df, bar_sec):
    """4 trades sintéticos por barra. Orden high/low aleatorio para no sesgar calibración."""
    trades = []
    for _, b in window_df.iterrows():
        t0 = b["timestamp"].timestamp()
        hl = [b["high"], b["low"]] if random.random() < 0.5 else [b["low"], b["high"]]
        for offset, px in zip([0, bar_sec/3, 2*bar_sec/3, bar_sec - 1],
                               [b["open"]] + hl + [b["close"]]):
            trades.append({"price": float(px), "timestamp": t0 + offset})
    return trades


def compute_model(calib_window, mid_window, q,
                  target_spread_frac, Tt, Tl, bar_sec, form, announced):
    mid          = compute_mid(mid_window, MID_METHOD)
    model_bars   = calib_window[["open","high","low","close"]].to_dict("records")
    model_trades = _bars_to_trades(calib_window, bar_sec)
    return avellaneda_stoikov(
        mid=mid, trades=model_trades, bars=model_bars,
        q=q, Tt=Tt, Tl=Tl,
        bar_seconds=bar_sec, form=form,
        target_spread_frac=target_spread_frac,
        verbose=not announced)


def get_signals(prev_close, res, q, capital, risk_pct, atr, atr_mult,
                inv_limit, ma, above_ask, below_bid,
                above_ask_entry, closes_below_ask):
    """
    Retorna lista de acciones: [("buy"|"sell", qty)].

    Parámetros de estado (gestionados en backtest.py):
      above_ask       — True si precio superó el ask mientras hay posición (Bollinger TP)
      below_bid       — True si precio estuvo bajo el bid sin posición    (Bollinger entrada)
      above_ask_entry — True si precio estuvo bajo el ask sin posición    (Tendency entrada)
      closes_below_ask — contador de cierres consecutivos bajo el ask     (Tendency TP)

    Sizing: qty = floor(capital × risk_pct / (atr × atr_mult)), mínimo 1, cap inv_limit.
    Filtro MA: solo entradas si prev_close >= ma.
    """
    bid, ask = res["bid"], res["ask"]
    acts     = []

    sl_dist  = max(atr * atr_mult, 0.01)
    qty_raw  = int((capital * risk_pct) / sl_dist)
    capacity = int(inv_limit - q)
    qty      = max(1, min(qty_raw, capacity))

    if MODE == "bollinger":
        if q > 0:
            if above_ask and prev_close <= ask:        # TP: cruce ask hacia abajo
                acts.append(("sell", q))
            elif prev_close <= bid:                    # SL: precio vuelve al bid
                acts.append(("sell", q))
        else:
            if below_bid and prev_close >= bid and prev_close >= ma and capacity > 0:
                acts.append(("buy", qty))

    elif MODE == "tendency":
        if q > 0:
            if closes_below_ask >= 2:                  # TP: 2 cierres consecutivos bajo ask
                acts.append(("sell", q))
            elif prev_close <= bid:                    # SL: precio cae al bid
                acts.append(("sell", q))
        else:
            if above_ask_entry and prev_close >= ask and prev_close >= ma and capacity > 0:
                acts.append(("buy", qty))

    return acts