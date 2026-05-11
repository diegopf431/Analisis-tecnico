# avellaneda_stoikov.py
# ──────────────────────────────────────────────────────────────────────────────
# Modelo completo Avellaneda-Stoikov (2008), unidades ABSOLUTAS (USD).
#
# Consistencia dimensional:
#   sigma  [USD/√s]   — Garman-Klass × mid / √bar_seconds
#   k      [1/USD]    — decay de intensidad de órdenes, calibrado en USD
#   A      [trades/s] — intensidad base de órdenes
#   gamma  [1/USD]    — CARA risk aversion: γ = frac·mid / (σ²·Tt)
#   s1,Sb,Sa [USD]    — bid = mid − Sb,  ask = mid + Sa
#
# Nota: en CARA, gamma·W debe ser adimensional → gamma ~ 1/USD ✓
# ──────────────────────────────────────────────────────────────────────────────

import math
from collections import Counter


def _sigma(bars, mid, bar_seconds):
    """Volatilidad absoluta USD/√s via Garman-Klass (o fallback)."""
    has_open = "open" in bars[0]
    has_hl   = "high" in bars[0] and "low" in bars[0]
    scale    = math.sqrt(bar_seconds)

    if has_open and has_hl:
        gk = sum(
            0.5 * math.log(b["high"] / b["low"])**2
            - (2*math.log(2)-1) * math.log(b["close"] / b["open"])**2
            for b in bars
        ) / len(bars)
        return math.sqrt(gk) * mid / scale, "Garman-Klass"

    if has_hl:
        pk = sum(math.log(b["high"] / b["low"])**2 for b in bars) / (4 * math.log(2) * len(bars))
        return math.sqrt(pk) * mid / scale, "Parkinson"

    closes = [b["close"] for b in bars]
    rs = [math.log(closes[i] / closes[i-1]) for i in range(1, len(closes))]
    rm = sum(rs) / len(rs)
    return math.sqrt(sum((r - rm)**2 for r in rs) / (len(rs) - 1)) * mid / scale, "log-returns"


def _calibrate(trades, Tl):
    """
    Calibra κ [1/USD] y A [trades/s] desde distancias absolutas en USD.
    Usa un grid proporcional al precio para ser robusto en cualquier activo.
    """
    prices    = [t["price"] for t in trades]
    mid_ref   = sum(prices) / len(prices)
    tick      = mid_ref * 0.0001           # grid = 0.01% del precio
    w         = max(1, len(prices) // 10)
    lmid      = lambda i: sum(prices[max(0, i-w):i+w+1]) / len(prices[max(0, i-w):i+w+1])
    distances = Counter(
        round(abs(prices[i] - lmid(i)) / tick) * tick
        for i in range(len(prices))
    )
    distances.pop(0.0, None)               # distancia cero no aporta info
    if not distances:
        raise ValueError("Todas las distancias son cero.")

    lambdas = {xi: n / Tl for xi, n in distances.items()}
    xs = list(lambdas.keys())
    ys = [math.log(l) for l in lambdas.values()]
    xm, ym = sum(xs) / len(xs), sum(ys) / len(ys)
    denom  = sum((x - xm)**2 for x in xs)
    if denom < 1e-30:
        raise ValueError("Varianza de distancias demasiado baja para calibrar k.")
    k  = -(sum((x-xm)*(y-ym) for x, y in zip(xs, ys)) / denom)
    k  = max(k, 1e-6)                      # guard mínimo, sin floor agresivo
    A  = math.exp(ym + k * xm)
    return k, A


def avellaneda_stoikov(mid, trades, bars, q, Tt, Tl,
                       bar_seconds=300, form="AS",
                       target_spread_frac=0.002, verbose=True):
    """
    Calcula quotes óptimos del modelo Avellaneda-Stoikov.

    γ = target_spread_frac · mid / (σ² · Tt)  [1/USD]
    Todas las variables intermedias están en USD.
    bid = mid − Sb,  ask = mid + Sa  (en USD).

    form="AS" → canónica AS 2008
    form="CJ" → Cartea-Jaimungal
    """
    if len(trades) < 5:
        raise ValueError(f"Se necesitan al menos 5 trades, recibidos: {len(trades)}")
    if len(bars) < 2:
        raise ValueError(f"Se necesitan al menos 2 barras, recibidas: {len(bars)}")

    Tt    = max(Tt, 1e-9)
    k, A  = _calibrate(trades, Tl)
    sigma, s_method = _sigma(bars, mid, bar_seconds)
    sigma = max(sigma, 1e-10)

    # γ [1/USD] derivado de política de spread objetivo
    gamma = (target_spread_frac * mid) / max(sigma**2 * Tt, 1e-10)
    gamma = min(gamma, k * 0.9)            # mantiene γ < k para estabilidad de s1

    s1 = (1.0 / gamma) * math.log(1.0 + gamma / k)   # USD

    if verbose:
        print(f"[AS] σ: {s_method} ({sigma:.4f} USD/√s) | k: {k:.4f} | γ: {gamma:.4f} | forma: {form}")

    if form == "AS":
        inv = gamma * sigma**2 * Tt                    # USD
        Sb  = s1 + (q + 0.5) * inv
        Sa  = s1 - (q - 0.5) * inv

    elif form == "CJ":
        s2  = math.sqrt((gamma / (2.0 * A * k)) * (1.0 + gamma / k)**(1.0 + k / gamma))
        Sb  = s1 + (q + 0.5) * sigma * s2 * math.sqrt(Tt)
        Sa  = s1 - (q - 0.5) * sigma * s2 * math.sqrt(Tt)

    else:
        raise ValueError(f"form debe ser 'AS' o 'CJ', recibido: '{form}'")

    # Seguridad: clamp al spread objetivo (política no negociable)
    max_half = target_spread_frac * mid
    Sb = max(min(Sb, max_half), max_half * 0.05)
    Sa = max(min(Sa, max_half), max_half * 0.05)

    r = mid - q * gamma * sigma**2 * Tt               # precio reserva [USD]

    return {
        "r":            r,
        "Sb":           Sb,
        "Sa":           Sa,
        "bid":          mid - Sb,
        "ask":          mid + Sa,
        "spread":       Sb + Sa,
        "k":            k,
        "A":            A,
        "sigma":        sigma,
        "sigma_method": s_method,
        "gamma":        gamma,
        "form":         form,
    }