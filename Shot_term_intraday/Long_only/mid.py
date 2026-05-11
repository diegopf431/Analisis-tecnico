# mid.py
# ──────────────────────────────────────────────────────────────────────────────
# Funciones puras de cálculo de mid price.
# Entrada: window_df (DataFrame con columnas open/high/low/close/volume).
# Salida : float (mid price actual).
#
# Métodos disponibles:
#   "classic" — (high + low) / 2 de la barra actual
#   "hlc3"    — (high + low + close) / 3 de la barra actual
#   "twap"    — media simple de closes en la ventana
#   "vwap"    — media ponderada por volumen (typical price × volume)
#   "ema"     — EMA de closes sobre la ventana (span = mitad de la ventana)
# ──────────────────────────────────────────────────────────────────────────────

def compute_mid(window_df, method="classic"):
    row = window_df.iloc[-1]

    if method == "classic":
        return (row["high"] + row["low"]) / 2.0

    if method == "hlc3":
        return (row["high"] + row["low"] + row["close"]) / 3.0

    if method == "twap":
        return float(window_df["close"].mean())

    if method == "vwap":
        tp  = (window_df["high"] + window_df["low"] + window_df["close"]) / 3.0
        vol = window_df["volume"].replace(0, float("nan"))
        val = float((tp * vol).sum() / vol.sum())
        # fallback a hlc3 si el volumen era todo cero
        if val != val:  # NaN check sin importar numpy
            return (row["high"] + row["low"] + row["close"]) / 3.0
        return val

    if method == "ema":
        span = max(len(window_df) // 2, 2)
        return float(window_df["close"].ewm(span=span).mean().iloc[-1])

    raise ValueError(f"MID_METHOD desconocido: '{method}'. "
                     f"Opciones: classic, hlc3, twap, vwap, ema")