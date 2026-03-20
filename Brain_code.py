import Analisis_core as core  # Importamos tu motor
import numpy as np

# ========================================================================
# 1. DEFINICIÓN DEL PORTAFOLIO / TICKER
# ========================================================================
# OPCIÓN A: Un solo activo
# mis_activos = ['NVDA']
# mis_pesos = [1.0]

# OPCIÓN B: Portafolio Tecnológico
mis_activos = ['NVDA', 'MSFT', 'GOOGL', 'AMD']
mis_pesos = [0.4, 0.2, 0.2, 0.2]  # Nvidia pesa más

# ========================================================================
# 2. EJECUCIÓN DEL ANÁLISIS
# ========================================================================
print("╔════════════════════════════════════════════════════╗")
print("║    INICIANDO SISTEMA DE DECISIÓN ALGORÍTMICA       ║")
print("╚════════════════════════════════════════════════════╝")

resultados = core.ejecutar_analisis_completo(mis_activos, mis_pesos)

if "error" in resultados:
    print(f"Error crítico: {resultados['error']}")
    exit()

# Extraer señales (-1: Venta, 0: Neutro/Hold, 1: Compra)
s_simple = resultados['ma_simple']['senal']
s_doble = resultados['ma_doble']['senal']
s_rsi = resultados['rsi']['senal']

# ========================================================================
# 3. META-ESTRATEGIA (LÓGICA DE CONSENSO)
# ========================================================================
# Sumamos las señales para ver la fuerza de la tendencia
# Rango posible: de -3 (Venta Fuerte) a +3 (Compra Fuerte)
score_consenso = s_simple + s_doble + s_rsi

decision_final = ""
accion_sugerida = ""

if score_consenso == 3:
    decision_final = "COMPRA FUERTE (Strong Buy)"
    accion_sugerida = "Entrar con posición completa. Todas las estrategias coinciden."
elif score_consenso >= 1:
    decision_final = "COMPRA (Buy)"
    accion_sugerida = "Entrar con cautela o aumentar posición. Tendencia positiva mayoritaria."
elif score_consenso == 0:
    decision_final = "NEUTRO / MANTENER (Hold)"
    accion_sugerida = "El mercado está indeciso o en transición. No abrir nuevas operaciones."
elif score_consenso > -3:
    decision_final = "VENTA (Sell)"
    accion_sugerida = "Reducir exposición. Tendencia negativa mayoritaria."
else:
    decision_final = "VENTA FUERTE (Strong Sell)"
    accion_sugerida = "Cerrar posiciones inmediatamente o abrir cortos. Colapso técnico."

# ========================================================================
# 4. REPORTE FINAL
# ========================================================================
print("\n" + "="*60)
print(f"REPORTE PARA: {mis_activos}")
print(f"PRECIO SINTÉTICO ACTUAL: {resultados['precio_actual']:.2f}")
print("="*60)

print(f"\n[1] MA Simple (Lookback: {resultados['ma_simple']['param']}d):")
print(f"    --> Señal: {'COMPRA' if s_simple==1 else 'VENTA' if s_simple==-1 else 'NEUTRO'}")

print(f"\n[2] MA Doble (Corto: {resultados['ma_doble']['param'][0]}d, Largo: {resultados['ma_doble']['param'][1]}d):")
print(f"    --> Señal: {'COMPRA' if s_doble==1 else 'VENTA' if s_doble==-1 else 'NEUTRO'}")

print(f"\n[3] RSI (Periodo: {resultados['rsi']['param'][0]}):")
print(f"    --> Señal: {'COMPRA' if s_rsi==1 else 'VENTA' if s_rsi==-1 else 'NEUTRO'}")

print("\n" + "-"*60)
print(f"PUNTAJE DE CONSENSO: {score_consenso}/3")
print("-"*60)
print(f"DECISIÓN FINAL:  {decision_final}")
print(f"ACCIÓN:          {accion_sugerida}")
print("="*60 + "\n")