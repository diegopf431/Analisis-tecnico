import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import os
import sys
import yfinance as yf
from itertools import product
import warnings
warnings.filterwarnings('ignore')

# ╔════════════════════════════════════════════════════════════════════════════╗
# ║                                                                            ║
# ║    SECCIÓN DE CONFIGURACIÓN DEL ACTIVO / PORTAFOLIO                        ║
# ║                                                                            ║
# ╚════════════════════════════════════════════════════════════════════════════╝

MODO_ANALISIS = 'activo_unico'  # Usar 'activo_unico' o 'portafolio'

# CONFIGURACIÓN WALK-FORWARD
NUM_FOLDS = 5          # Número de ventanas de testeo
MESES_TEST = 6         # Duración de cada ventana Out-Of-Sample (en meses)

# CONFIGURACIÓN PARA ACTIVO ÚNICO
TICKER_UNICO = 'SPY'  

# CONFIGURACIÓN PARA PORTAFOLIO
PORTAFOLIO = {
    'NVDA': 0.30,   
    'AAPL': 0.25,   
    'MSFT': 0.25,   
    'GOOGL': 0.20,  
}

FECHA_INICIO = '2020-01-01'

# ========================================================================
# FUNCIÓN PARA DESCARGAR Y PROCESAR DATOS
# ========================================================================
def descargar_portafolio(portafolio, fecha_inicio):
    suma_pesos = sum(portafolio.values())
    if abs(suma_pesos - 1.0) > 0.01:
        print(f"⚠️ Advertencia: Los pesos del portafolio suman {suma_pesos:.2%}, no 100%")
        print("   Normalizando pesos...")
        portafolio = {k: v/suma_pesos for k, v in portafolio.items()}
    
    print("\n" + "="*70)
    print("DESCARGANDO DATOS DEL PORTAFOLIO")
    print("="*70)
    
    tickers_list = list(portafolio.keys())
    datos = yf.download(tickers_list, start=fecha_inicio, 
                        end=datetime.now().strftime('%Y-%m-%d'), progress=False)
    
    if len(tickers_list) == 1:
        precios_cierre = datos['Close'].to_frame(tickers_list[0])
    else:
        precios_cierre = datos['Close']
    
    precios_cierre = precios_cierre.dropna()
    retornos = precios_cierre.pct_change().dropna()
    
    retornos_portafolio = pd.Series(0.0, index=retornos.index)
    for ticker, peso in portafolio.items():
        retornos_portafolio += retornos[ticker] * peso
    
    precio_inicial = 100
    precios_portafolio = pd.Series(index=retornos.index)
    precios_portafolio.iloc[0] = precio_inicial * (1 + retornos_portafolio.iloc[0])
    
    for i in range(1, len(retornos_portafolio)):
        precios_portafolio.iloc[i] = precios_portafolio.iloc[i-1] * (1 + retornos_portafolio.iloc[i])
    
    return precios_portafolio, portafolio

try:
    if MODO_ANALISIS == 'activo_unico':
        ipsa = yf.download(TICKER_UNICO, start=FECHA_INICIO, 
                          end=datetime.now().strftime('%Y-%m-%d'), progress=False)
        hoy = pd.Timestamp.now().normalize()
        ipsa = ipsa[ipsa.index <= hoy]
        precios = ipsa['Close']
        precios = pd.Series(precios.values.flatten(), index=precios.index)
        NOMBRE_ACTIVO = TICKER_UNICO
        
    elif MODO_ANALISIS == 'portafolio':
        precios, PORTAFOLIO = descargar_portafolio(PORTAFOLIO, FECHA_INICIO)
        NOMBRE_ACTIVO = "Portafolio (" + ", ".join([f"{t}:{p:.0%}" for t, p in PORTAFOLIO.items()]) + ")"
    else:
        sys.exit(1)
        
except Exception as e:
    print(f"Error al descargar datos: {str(e)}")
    sys.exit(1)

# ========================================================================
# FUNCIONES DE ESTRATEGIAS
# ========================================================================
def MA_simple(precios, t_inicial, lookback):
    H = len(precios)
    senales = np.zeros(H)
    ultima_senal = 0
    MA = np.zeros(H)
    precios_array = precios.values.flatten()
    
    for t in range(t_inicial-1, H):
        MA[t] = np.mean(precios_array[t-lookback+1:t+1])
        if precios_array[t] > MA[t] and ultima_senal <= 0:
            senales[t] = 1
            ultima_senal = 1
        elif precios_array[t] < MA[t] and ultima_senal >= 0:
            senales[t] = -1
            ultima_senal = -1
            
    return pd.DataFrame({'precios': precios_array, 'MA': MA, 'senales': senales}, index=precios.index)

def MA_doble(precios, t_inicial, lookback_corto, lookback_largo):
    H = len(precios)
    senales = np.zeros(H)
    ultima_senal = 0
    MA_corta = np.zeros(H)
    MA_larga = np.zeros(H)
    precios_array = precios.values.flatten()
    
    for t in range(t_inicial-1, H):
        MA_corta[t] = np.mean(precios_array[t-lookback_corto+1:t+1])
        MA_larga[t] = np.mean(precios_array[t-lookback_largo+1:t+1])
        
        if precios_array[t] > MA_corta[t] and precios_array[t] > MA_larga[t] and ultima_senal <= 0:
            senales[t] = 1
            ultima_senal = 1
        elif precios_array[t] < MA_corta[t] and precios_array[t] < MA_larga[t] and ultima_senal >= 0:
            senales[t] = -1
            ultima_senal = -1
            
    return pd.DataFrame({'precios': precios_array, 'MA_corta': MA_corta, 'MA_larga': MA_larga, 'senales': senales}, index=precios.index)

def RSI_strategy(precios, t_inicial, periodo_rsi=14, nivel_sobrecompra=70, nivel_sobreventa=30, modo_tendencia=False):
    H = len(precios)
    senales = np.zeros(H)
    ultima_senal = 0
    RSI = np.zeros(H)
    RS = np.zeros(H)
    RSI_anterior = 50
    precios_array = precios.values.flatten()
    
    for t in range(t_inicial-1, H):
        ventana = precios_array[t-periodo_rsi+1:t+1]
        cambios = np.diff(ventana)
        alzas = cambios[cambios > 0]
        bajas = -cambios[cambios < 0]
        
        promedio_alzas = np.mean(alzas) if len(alzas) > 0 else 0
        promedio_bajas = np.mean(bajas) if len(bajas) > 0 else 0
        
        if promedio_bajas == 0:
            RS[t] = 100
            RSI[t] = 100
        else:
            RS[t] = promedio_alzas / promedio_bajas
            RSI[t] = 100 - (100 / (1 + RS[t]))
        
        if modo_tendencia:
            if RSI_anterior <= nivel_sobreventa and RSI[t] > nivel_sobreventa:
                senales[t] = 1
        else:
            if RSI_anterior <= nivel_sobreventa and RSI[t] > nivel_sobreventa and ultima_senal <= 0:
                senales[t] = 1
                ultima_senal = 1
            elif RSI_anterior >= nivel_sobrecompra and RSI[t] < nivel_sobrecompra and ultima_senal >= 0:
                senales[t] = -1
                ultima_senal = -1
        
        RSI_anterior = RSI[t]
        
    return pd.DataFrame({'precios': precios_array, 'RSI': RSI, 'RS': RS, 'senales': senales}, index=precios.index)

# CONSTANTE DE COSTO DE TRANSACCIÓN
COSTO_TRANSACCION = 0.0025 * (1 + 0.19)  # 0.2975%

def calcular_ATR(precios, periodo_atr):
    precios_array = precios.values.flatten() if hasattr(precios, 'values') else np.array(precios).flatten()
    H = len(precios_array)
    ATR = np.zeros(H)
    TR = np.zeros(H)
    TR[1:] = np.abs(precios_array[1:] - precios_array[:-1])
    for t in range(periodo_atr, H):
        ATR[t] = np.mean(TR[t-periodo_atr+1:t+1])
    return ATR

def aplicar_chandelier_exit(precios, senales_df, periodo_chandelier, multiplicador_atr, periodo_atr=14):
    precios_array = precios.values.flatten() if hasattr(precios, 'values') else np.array(precios).flatten()
    senales_originales = senales_df['senales'].values.copy()
    senales_nuevas = senales_originales.copy()
    H = len(precios_array)
    ATR = calcular_ATR(precios, periodo_atr)
    chandelier_exit = np.zeros(H)
    
    for t in range(periodo_chandelier, H):
        maximo_periodo = np.max(precios_array[t-periodo_chandelier+1:t+1])
        chandelier_exit[t] = maximo_periodo - multiplicador_atr * ATR[t]
        
    en_posicion = False
    for t in range(max(periodo_chandelier, periodo_atr), H):
        if senales_originales[t] == 1:
            if not en_posicion:
                en_posicion = True
            else:
                senales_nuevas[t] = 0
        elif senales_originales[t] == -1:
            en_posicion = False
        elif en_posicion and precios_array[t] < chandelier_exit[t]:
            senales_nuevas[t] = -1
            en_posicion = False
            
    resultado = senales_df.copy()
    resultado['senales'] = senales_nuevas
    resultado['chandelier_exit'] = chandelier_exit
    resultado['ATR'] = ATR
    return resultado

def calcular_retornos(precios, senales_df):
    r = np.log(precios / precios.shift(1))
    ret_largos = []
    perdidas_evitadas = []
    num_operaciones = 0
    H = len(precios)
    senales = senales_df['senales'].values
    
    for t in range(H):
        if senales[t] > 0:
            rango_busqueda = senales[t+1:H]
            if len(rango_busqueda) > 0:
                siguiente_corto = np.where(rango_busqueda == -1)[0]
                aux = siguiente_corto[0] if len(siguiente_corto) > 0 else len(rango_busqueda)
                if aux > 0:
                    ret_largos.extend(r.iloc[t+1:t+1+aux].dropna().tolist())
                    num_operaciones += 1
                    
        if senales[t] < 0:
            rango_busqueda = senales[t+1:H]
            if len(rango_busqueda) > 0:
                siguiente_largo = np.where(rango_busqueda == 1)[0]
                aux = siguiente_largo[0] if len(siguiente_largo) > 0 else len(rango_busqueda)
                if aux > 0:
                    retorno_periodo = r.iloc[t+1:t+1+aux].dropna().tolist()
                    for ret in retorno_periodo:
                        perdidas_evitadas.append(0)
                    num_operaciones += 1
                    
    ret_largos = np.array(ret_largos)
    perdidas_evitadas = np.array(perdidas_evitadas)
    
    costo_log = np.log(1 - COSTO_TRANSACCION)
    costo_total = num_operaciones * costo_log
    
    ret_combinado = ret_largos.copy()
    if len(ret_combinado) > 0:
        ret_combinado = ret_combinado + (costo_total / len(ret_combinado))
        
    return ret_largos, perdidas_evitadas, ret_combinado, num_operaciones

def mostrar_estadisticas(ret_largos, perdidas_evitadas, ret_combinado, senales, titulo, num_operaciones=0):
    print("\n" + "="*70)
    print(titulo)
    print("="*70)
    
    print(f"\nNúmero de señales generadas: {np.sum(senales != 0)}")
    print(f"Señales de compra (entrar al mercado): {np.sum(senales == 1)}")
    print(f"Señales de venta (salir del mercado): {np.sum(senales == -1)}")
    print(f"Número total de operaciones: {num_operaciones}")
    print(f"Costo de transacción por operación: {COSTO_TRANSACCION*100:.4f}%")
    print(f"Costo total de transacciones: {num_operaciones * COSTO_TRANSACCION * 100:.4f}%")
    
    print("\n" + "-"*70)
    print("RETORNOS DE POSICIONES LARGAS (EN EL MERCADO)")
    print("-"*70)
    print(f"Total de días en posición: {len(ret_largos)}")
    if len(ret_largos) > 0:
        num_trades_largos = np.sum(senales == 1)
        media_largos = np.mean(ret_largos)
        std_largos = np.std(ret_largos)
        sharpe_largos = media_largos / std_largos if std_largos > 0 else 0
        sharpe_anualizado_largos = sharpe_largos * np.sqrt(252) if std_largos > 0 else 0
        retorno_por_trade_largos = np.sum(ret_largos) / num_trades_largos if num_trades_largos > 0 else 0
        
        print(f"Número de trades largos: {num_trades_largos}")
        print(f"Retorno promedio diario: {media_largos:.6f}")
        print(f"Retorno acumulado (sin costos): {np.sum(ret_largos):.6f}")
        print(f"Retorno promedio por trade: {retorno_por_trade_largos:.6f}")
        print(f"Desviación estándar: {std_largos:.6f}")
        print(f"Sharpe Ratio: {sharpe_largos:.6f}")
        print(f"Sharpe Ratio Anualizado: {sharpe_anualizado_largos:.6f}")
    else:
        print("No hay retornos de posiciones largas")
        media_largos = std_largos = sharpe_largos = sharpe_anualizado_largos = retorno_por_trade_largos = num_trades_largos = 0
    
    print("\n" + "-"*70)
    print("PERÍODOS FUERA DEL MERCADO (SALIDAS DE POSICIÓN)")
    print("-"*70)
    num_salidas = np.sum(senales == -1)
    dias_fuera = len(perdidas_evitadas)
    print(f"Número de salidas de posición: {num_salidas}")
    print(f"Total de días fuera del mercado: {dias_fuera}")
    
    print("\n" + "-"*70)
    print("RETORNO TOTAL DE LA ESTRATEGIA (CON COSTOS DE TRANSACCIÓN)")
    print("-"*70)
    print(f"Total de días evaluados: {len(ret_combinado)}")
    if len(ret_combinado) > 0:
        num_trades_total = np.sum(senales != 0)
        media_combinado = np.mean(ret_combinado)
        std_combinado = np.std(ret_combinado)
        sharpe_combinado = media_combinado / std_combinado if std_combinado > 0 else 0
        sharpe_anualizado_combinado = sharpe_combinado * np.sqrt(252) if std_combinado > 0 else 0
        num_compras = np.sum(senales == 1)
        retorno_por_trade_combinado = np.sum(ret_combinado) / num_compras if num_compras > 0 else 0
        
        print(f"Número total de señales: {num_trades_total}")
        print(f"Número de operaciones (compras + ventas): {num_operaciones}")
        print(f"Retorno promedio diario: {media_combinado:.6f}")
        print(f"Retorno acumulado (con costos): {np.sum(ret_combinado):.6f}")
        print(f"Retorno promedio por trade: {retorno_por_trade_combinado:.6f}")
        print(f"Desviación estándar: {std_combinado:.6f}")
        print(f"Sharpe Ratio: {sharpe_combinado:.6f}")
        print(f"Sharpe Ratio Anualizado: {sharpe_anualizado_combinado:.6f}")
    else:
        print("No hay retornos")
        media_combinado = std_combinado = sharpe_combinado = sharpe_anualizado_combinado = retorno_por_trade_combinado = num_trades_total = 0
    
    return {
        'largos': {
            'media': media_largos, 'acum': np.sum(ret_largos) if len(ret_largos) > 0 else 0, 
            'sharpe': sharpe_largos, 'sharpe_anualizado': sharpe_anualizado_largos,
            'retorno_por_trade': retorno_por_trade_largos, 'num_trades': num_trades_largos
        },
        'fuera_mercado': {'num_salidas': num_salidas, 'dias_fuera': dias_fuera},
        'combinado': {
            'media': media_combinado, 'acum': np.sum(ret_combinado) if len(ret_combinado) > 0 else 0, 
            'sharpe': sharpe_combinado, 'sharpe_anualizado': sharpe_anualizado_combinado,
            'retorno_por_trade': retorno_por_trade_combinado, 'num_trades': num_trades_total, 'num_operaciones': num_operaciones
        }
    }

def calcular_metrica_optimizacion(ret_combinado):
    if len(ret_combinado) > 0:
        return np.sum(ret_combinado), np.std(ret_combinado)
    else:
        return -999, 999

def calcular_win_rate(precios, senales_df):
    precios_array = precios.values.flatten() if hasattr(precios, 'values') else np.array(precios).flatten()
    senales = senales_df['senales'].values
    H = len(precios_array)
    trades_exitosos = trades_totales = 0
    
    for t in range(H):
        if senales[t] > 0:
            precio_compra = precios_array[t]
            rango_busqueda = senales[t+1:H]
            if len(rango_busqueda) > 0:
                siguiente_venta = np.where(rango_busqueda == -1)[0]
                if len(siguiente_venta) > 0:
                    precio_venta = precios_array[t + 1 + siguiente_venta[0]]
                    trades_totales += 1
                    if precio_venta > precio_compra: trades_exitosos += 1
                    
        if senales[t] < 0:
            precio_venta = precios_array[t]
            rango_busqueda = senales[t+1:H]
            if len(rango_busqueda) > 0:
                siguiente_compra = np.where(rango_busqueda == 1)[0]
                if len(siguiente_compra) > 0:
                    precio_compra = precios_array[t + 1 + siguiente_compra[0]]
                    trades_totales += 1
                    if precio_compra < precio_venta: trades_exitosos += 1
                    
    return trades_exitosos / trades_totales if trades_totales > 0 else 0, trades_totales, trades_exitosos

def calc_sharpe_ratio(retornos_array):
    if len(retornos_array) == 0: return 0
    std = np.std(retornos_array)
    if std == 0: return 0
    return (np.mean(retornos_array) / std) * np.sqrt(252)

# ========================================================================
# OPTIMIZADORES (Con lógica Top 5%) - Retornan Diccionario Completo
# ========================================================================
def optimizar_MA_simple(precios, rangos_parametros, rangos_chandelier):
    resultados = []
    combinaciones = list(product(rangos_parametros['lookback'], rangos_chandelier['periodo_chandelier'], rangos_chandelier['multiplicador_atr']))
    total_combinaciones = len(combinaciones)
    contador = 0
    
    for lookback, periodo_chand, mult_atr in combinaciones:
        contador += 1
        t_inicial = lookback + 1
        if t_inicial >= len(precios): continue
        senales = MA_simple(precios, t_inicial, lookback)
        senales_con_stop = aplicar_chandelier_exit(precios, senales, periodo_chand, mult_atr)
        ret_largos, perdidas_evitadas, ret_combinado, num_ops = calcular_retornos(precios, senales_con_stop)
        retorno_acumulado, volatilidad = calcular_metrica_optimizacion(ret_combinado)
        
        resultados.append({'lookback': lookback, 'periodo_chandelier': periodo_chand, 'multiplicador_atr': mult_atr,
                           'retorno_acumulado': retorno_acumulado, 'volatilidad': volatilidad})
                           
    if len(resultados) == 0:
        return {'lookback': rangos_parametros['lookback'][0], 'periodo_chandelier': rangos_chandelier['periodo_chandelier'][0],
                'multiplicador_atr': rangos_chandelier['multiplicador_atr'][0], 'retorno_acumulado': 0, 'volatilidad': 999}
                
    # Lógica de Top 5%: Seleccionar el 5% con mayor retorno, luego elegir la de menor volatilidad
    resultados_ordenados = sorted(resultados, key=lambda x: x['retorno_acumulado'], reverse=True)
    top_k = max(1, int(len(resultados_ordenados) * 0.05))
    top_5_percent = resultados_ordenados[:top_k]
    mejor_resultado = sorted(top_5_percent, key=lambda x: (x['volatilidad'], -x['retorno_acumulado']))[0]
    
    return mejor_resultado

def optimizar_MA_doble(precios, rangos_parametros, rangos_chandelier):
    resultados = []
    combinaciones = list(product(rangos_parametros['lookback_corto'], rangos_parametros['lookback_largo'],
                                 rangos_chandelier['periodo_chandelier'], rangos_chandelier['multiplicador_atr']))
    total_combinaciones = len(combinaciones)
    contador = 0
    
    for lookback_corto, lookback_largo, periodo_chand, mult_atr in combinaciones:
        contador += 1
        if lookback_corto >= lookback_largo: continue
        t_inicial = lookback_largo + 1
        if t_inicial >= len(precios): continue
        senales = MA_doble(precios, t_inicial, lookback_corto, lookback_largo)
        senales_con_stop = aplicar_chandelier_exit(precios, senales, periodo_chand, mult_atr)
        ret_largos, perdidas_evitadas, ret_combinado, num_ops = calcular_retornos(precios, senales_con_stop)
        retorno_acumulado, volatilidad = calcular_metrica_optimizacion(ret_combinado)
        
        resultados.append({'lookback_corto': lookback_corto, 'lookback_largo': lookback_largo,
                           'periodo_chandelier': periodo_chand, 'multiplicador_atr': mult_atr,
                           'retorno_acumulado': retorno_acumulado, 'volatilidad': volatilidad})
                           
    if len(resultados) == 0:
        return {'lookback_corto': rangos_parametros['lookback_corto'][0], 'lookback_largo': rangos_parametros['lookback_largo'][0],
                'periodo_chandelier': rangos_chandelier['periodo_chandelier'][0], 'multiplicador_atr': rangos_chandelier['multiplicador_atr'][0], 
                'retorno_acumulado': 0, 'volatilidad': 999}
                
    # Lógica de Top 5%: Seleccionar el 5% con mayor retorno, luego elegir la de menor volatilidad
    resultados_ordenados = sorted(resultados, key=lambda x: x['retorno_acumulado'], reverse=True)
    top_k = max(1, int(len(resultados_ordenados) * 0.05))
    top_5_percent = resultados_ordenados[:top_k]
    mejor_resultado = sorted(top_5_percent, key=lambda x: (x['volatilidad'], -x['retorno_acumulado']))[0]
    
    return mejor_resultado

def optimizar_RSI(precios, rangos_parametros, rangos_chandelier):
    resultados = []
    combinaciones = list(product(rangos_parametros['periodo_rsi'], rangos_parametros['nivel_sobrecompra'],
                                 rangos_parametros['nivel_sobreventa'], rangos_chandelier['periodo_chandelier'],
                                 rangos_chandelier['multiplicador_atr']))
    total_combinaciones = len(combinaciones)
    contador = 0
    
    for periodo_rsi, nivel_sobrecompra, nivel_sobreventa, periodo_chand, mult_atr in combinaciones:
        contador += 1
        if nivel_sobreventa >= nivel_sobrecompra: continue
        t_inicial = periodo_rsi + 1
        if t_inicial >= len(precios): continue
        
        senales = RSI_strategy(precios, t_inicial, periodo_rsi, nivel_sobrecompra, nivel_sobreventa, MODO_TENDENCIA_RSI)
        senales_con_stop = aplicar_chandelier_exit(precios, senales, periodo_chand, mult_atr)
        ret_largos, perdidas_evitadas, ret_combinado, num_ops = calcular_retornos(precios, senales_con_stop)
        retorno_acumulado, volatilidad = calcular_metrica_optimizacion(ret_combinado)
        win_rate, num_trades, trades_exitosos = calcular_win_rate(precios, senales_con_stop)
        
        if num_trades < 1: continue
        
        resultados.append({'periodo_rsi': periodo_rsi, 'nivel_sobrecompra': nivel_sobrecompra, 'nivel_sobreventa': nivel_sobreventa,
                           'periodo_chandelier': periodo_chand, 'multiplicador_atr': mult_atr, 'retorno_acumulado': retorno_acumulado,
                           'volatilidad': volatilidad, 'win_rate': win_rate, 'num_trades': num_trades, 'trades_exitosos': trades_exitosos})
                           
    if len(resultados) == 0:
        return {'periodo_rsi': rangos_parametros['periodo_rsi'][0], 'nivel_sobrecompra': rangos_parametros['nivel_sobrecompra'][0],
                'nivel_sobreventa': rangos_parametros['nivel_sobreventa'][0], 'periodo_chandelier': rangos_chandelier['periodo_chandelier'][0],
                'multiplicador_atr': rangos_chandelier['multiplicador_atr'][0], 'retorno_acumulado': 0, 'volatilidad': 999}
                
    # Lógica de Top 5%: Seleccionar el 5% con mayor win rate, luego elegir el de mayor retorno
    resultados_ordenados = sorted(resultados, key=lambda x: x['win_rate'], reverse=True)
    top_k = max(1, int(len(resultados_ordenados) * 0.05))
    top_5_percent = resultados_ordenados[:top_k]
    mejor_resultado = sorted(top_5_percent, key=lambda x: (-x['retorno_acumulado'], x['volatilidad']))[0]
    
    return mejor_resultado

# RANGOS ORIGINALES
rangos_MA_simple = {'lookback': list(range(10, 51, 5))}
rangos_MA_doble = {'lookback_corto': list(range(5, 31, 5)), 'lookback_largo': list(range(30, 101, 10))}
rangos_RSI = {'periodo_rsi': list(range(10, 25, 2)), 'nivel_sobrecompra': list(range(60, 81, 5)), 'nivel_sobreventa': list(range(35, 36, 5))}
MODO_TENDENCIA_RSI = True
rangos_chandelier = {'periodo_chandelier': list(range(10, 31, 5)), 'multiplicador_atr': np.arange(2.0, 6.5, 0.5)}

# ========================================================================
# CONSTRUCCIÓN DE VENTANAS WALK-FORWARD Y LOG DE AUDITORÍA
# ========================================================================
fecha_actual = precios.index[-1]
ventanas = []

for i in range(NUM_FOLDS):
    fecha_inicio_test = fecha_actual - pd.DateOffset(months=MESES_TEST)
    ventanas.append({
        'fold': NUM_FOLDS - i,
        'train_end': fecha_inicio_test,
        'test_start': fecha_inicio_test,
        'test_end': fecha_actual
    })
    fecha_actual = fecha_inicio_test

# Ordenar cronológicamente (Fold 1 primero)
ventanas = ventanas[::-1]

print("\n" + "╔" + "="*80 + "╗")
print("║" + " "*22 + "INICIANDO WALK-FORWARD BACKTESTING" + " "*24 + "║")
print(f"║" + f"   Configuración: {NUM_FOLDS} Folds de {MESES_TEST} meses OOS. Top 5% Selection.".center(78) + "║")
print("╚" + "="*80 + "╝")

oos_dfs_simple = []
oos_dfs_doble = []
oos_dfs_rsi = []

# Lista para almacenar el LOG DE AUDITORÍA
log_auditoria = []

for v in ventanas:
    print(f"\n" + "-"*80)
    print(f"[FOLD {v['fold']}] Entrenando datos hasta: {v['train_end'].date()} | Operando (Test): {v['test_start'].date()} a {v['test_end'].date()}")
    print("-"*80)
    
    # Datos in-sample (expandibles en cada fold)
    precios_train = precios[precios.index <= v['train_end']]
    
    print("   Optimizando MA Simple...")
    param_S = optimizar_MA_simple(precios_train, rangos_MA_simple, rangos_chandelier)
    print("   Optimizando MA Doble...")
    param_D = optimizar_MA_doble(precios_train, rangos_MA_doble, rangos_chandelier)
    print("   Optimizando RSI...")
    param_R = optimizar_RSI(precios_train, rangos_RSI, rangos_chandelier)
    
    precios_hasta_test = precios[precios.index <= v['test_end']]
    
    # Generar señales base
    s_base_S = MA_simple(precios_hasta_test, param_S['lookback'] + 1, param_S['lookback'])
    s_base_D = MA_doble(precios_hasta_test, param_D['lookback_largo'] + 1, param_D['lookback_corto'], param_D['lookback_largo'])
    s_base_R = RSI_strategy(precios_hasta_test, param_R['periodo_rsi'] + 1, param_R['periodo_rsi'], param_R['nivel_sobrecompra'], param_R['nivel_sobreventa'], MODO_TENDENCIA_RSI)
    
    # Aplicar Chandelier
    s_opt_S = aplicar_chandelier_exit(precios_hasta_test, s_base_S, param_S['periodo_chandelier'], param_S['multiplicador_atr'])
    s_opt_D = aplicar_chandelier_exit(precios_hasta_test, s_base_D, param_D['periodo_chandelier'], param_D['multiplicador_atr'])
    s_opt_R = aplicar_chandelier_exit(precios_hasta_test, s_base_R, param_R['periodo_chandelier'], param_R['multiplicador_atr'])
    
    # Recortar SOLAMENTE la ventana de Test (Out-of-Sample puro)
    mascara_test = (s_opt_S.index > v['test_start']) & (s_opt_S.index <= v['test_end'])
    
    df_test_S = s_opt_S[mascara_test]
    df_test_D = s_opt_D[mascara_test]
    df_test_R = s_opt_R[mascara_test]
    
    oos_dfs_simple.append(df_test_S)
    oos_dfs_doble.append(df_test_D)
    oos_dfs_rsi.append(df_test_R)
    
    # Calcular retornos OOS solo para este fold
    _, _, rc_S_fold, _ = calcular_retornos(precios_hasta_test[mascara_test], df_test_S)
    _, _, rc_D_fold, _ = calcular_retornos(precios_hasta_test[mascara_test], df_test_D)
    _, _, rc_R_fold, _ = calcular_retornos(precios_hasta_test[mascara_test], df_test_R)
    
    ret_oos_S = np.sum(rc_S_fold) if len(rc_S_fold) > 0 else 0
    ret_oos_D = np.sum(rc_D_fold) if len(rc_D_fold) > 0 else 0
    ret_oos_R = np.sum(rc_R_fold) if len(rc_R_fold) > 0 else 0

    # Calcular Retornos Acumulados y Sharpe hasta este Fold (Cosido continuo OOS hasta ahora)
    df_acum_S = pd.concat(oos_dfs_simple)
    df_acum_D = pd.concat(oos_dfs_doble)
    df_acum_R = pd.concat(oos_dfs_rsi)
    
    precios_acum = precios.loc[df_acum_S.index]
    
    _, _, rc_S_acum, _ = calcular_retornos(precios_acum, df_acum_S)
    _, _, rc_D_acum, _ = calcular_retornos(precios_acum, df_acum_D)
    _, _, rc_R_acum, _ = calcular_retornos(precios_acum, df_acum_R)
    
    ret_acum_S = np.sum(rc_S_acum) if len(rc_S_acum) > 0 else 0
    ret_acum_D = np.sum(rc_D_acum) if len(rc_D_acum) > 0 else 0
    ret_acum_R = np.sum(rc_R_acum) if len(rc_R_acum) > 0 else 0
    
    sharpe_S = calc_sharpe_ratio(rc_S_acum)
    sharpe_D = calc_sharpe_ratio(rc_D_acum)
    sharpe_R = calc_sharpe_ratio(rc_R_acum)

    # Guardar en el Log
    log_auditoria.append({
        'Fold': v['fold'],
        'Periodo OOS': f"{v['test_start'].date()} / {v['test_end'].date()}",
        
        'Params MA Simple': f"L:{param_S['lookback']} Ch:{param_S['periodo_chandelier']} ATR:{param_S['multiplicador_atr']}",
        'Ret IS Simple': param_S['retorno_acumulado'],
        'Ret OOS Simple': ret_oos_S,
        'Ret Acum S': ret_acum_S,
        'Sharpe Acum S': sharpe_S,
        
        'Params MA Doble': f"C:{param_D['lookback_corto']} L:{param_D['lookback_largo']} Ch:{param_D['periodo_chandelier']} ATR:{param_D['multiplicador_atr']}",
        'Ret IS Doble': param_D['retorno_acumulado'],
        'Ret OOS Doble': ret_oos_D,
        'Ret Acum D': ret_acum_D,
        'Sharpe Acum D': sharpe_D,
        
        'Params RSI': f"P:{param_R['periodo_rsi']} SC:{param_R['nivel_sobrecompra']} SV:{param_R['nivel_sobreventa']} Ch:{param_R['periodo_chandelier']} ATR:{param_R['multiplicador_atr']}",
        'Ret IS RSI': param_R['retorno_acumulado'],
        'Ret OOS RSI': ret_oos_R,
        'Ret Acum R': ret_acum_R,
        'Sharpe Acum R': sharpe_R
    })

# Concatenar todos los fragmentos OOS para crear la serie final continua
df_oos_simple_final = pd.concat(oos_dfs_simple)
df_oos_doble_final = pd.concat(oos_dfs_doble)
df_oos_rsi_final = pd.concat(oos_dfs_rsi)

# Precios correspondientes exactos a la serie concatenada
precios_oos_final = precios.loc[df_oos_simple_final.index]

# ========================================================================
# EVALUAR RESULTADOS FINALES OUT-OF-SAMPLE
# ========================================================================
print("\n" + "╔" + "="*80 + "╗")
print("║" + " "*20 + "RESULTADOS FINALES 100% OUT-OF-SAMPLE" + " "*21 + "║")
print("║" + " "*12 + "Estadísticas calculadas uniendo los Folds Walk-Forward" + " "*12 + "║")
print("╚" + "="*80 + "╝")

# MA SIMPLE
rl_S, pe_S, rc_S, ops_S = calcular_retornos(precios_oos_final, df_oos_simple_final)
stats_S = mostrar_estadisticas(rl_S, pe_S, rc_S, df_oos_simple_final['senales'].values, "MA SIMPLE (WALK-FORWARD OOS)", ops_S)

# MA DOBLE
rl_D, pe_D, rc_D, ops_D = calcular_retornos(precios_oos_final, df_oos_doble_final)
stats_D = mostrar_estadisticas(rl_D, pe_D, rc_D, df_oos_doble_final['senales'].values, "MA DOBLE (WALK-FORWARD OOS)", ops_D)

# RSI
rl_R, pe_R, rc_R, ops_R = calcular_retornos(precios_oos_final, df_oos_rsi_final)
stats_R = mostrar_estadisticas(rl_R, pe_R, rc_R, df_oos_rsi_final['senales'].values, "RSI (WALK-FORWARD OOS)", ops_R)


# ========================================================================
# IMPRIMIR TABLA DE AUDITORÍA (DETECCIÓN DE OVERFITTING)
# ========================================================================
print("\n" + "╔" + "="*160 + "╗")
print("║" + " "*62 + "TABLA DE AUDITORÍA WALK-FORWARD" + " "*67 + "║")
print("║" + " "*7 + "Evalúa la estabilidad de parámetros y la evolución continua de Retorno OOS y Sharpe Ratio a través del tiempo" + " "*22 + "║")
print("╚" + "="*160 + "╝")

df_log = pd.DataFrame(log_auditoria)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 200)

for index, row in df_log.iterrows():
    print(f"\n[{row['Periodo OOS']}] FOLD {row['Fold']}")
    print("-" * 162)
    print(f"{'Estrategia':<10} | {'Parámetros Seleccionados':<42} | {'Ret IS (Train)':<14} | {'Ret OOS (Fold)':<14} | {'Ret Acum OOS':<14} | {'Sharpe Acum OOS':<14}")
    print("-" * 162)
    print(f"{'MA Simple':<10} | {row['Params MA Simple']:<42} | {row['Ret IS Simple']:>13.4%} | {row['Ret OOS Simple']:>13.4%} | {row['Ret Acum S']:>13.4%} | {row['Sharpe Acum S']:>13.4f}")
    print(f"{'MA Doble':<10} | {row['Params MA Doble']:<42} | {row['Ret IS Doble']:>13.4%} | {row['Ret OOS Doble']:>13.4%} | {row['Ret Acum D']:>13.4%} | {row['Sharpe Acum D']:>13.4f}")
    print(f"{'RSI':<10} | {row['Params RSI']:<42} | {row['Ret IS RSI']:>13.4%} | {row['Ret OOS RSI']:>13.4%} | {row['Ret Acum R']:>13.4%} | {row['Sharpe Acum R']:>13.4f}")

# ========================================================================
# GRÁFICOS WALK-FORWARD
# ========================================================================
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(18, 16))

def graficar_wf(ax, df_wf, titulo, indicador1=None, indicador2=None):
    ax.plot(df_wf.index, df_wf['precios'], label='Precio Real', color='black', alpha=0.5, linewidth=2)
    ax.plot(df_wf.index, df_wf['chandelier_exit'], label='Trailing Stop (Chandelier)', color='red', linestyle='--', alpha=0.8)
    
    if indicador1: ax.plot(df_wf.index, df_wf[indicador1], label=indicador1, color='orange', linewidth=1.5)
    if indicador2: ax.plot(df_wf.index, df_wf[indicador2], label=indicador2, color='purple', linewidth=1.5)

    compras, ventas = df_wf[df_wf['senales'] == 1], df_wf[df_wf['senales'] == -1]
    ax.scatter(compras.index, compras['precios'], color='green', marker='^', s=80, zorder=5, label='Compra')
    ax.scatter(ventas.index, ventas['precios'], color='red', marker='v', s=80, zorder=5, label='Venta')
    
    # Dibujar las divisiones de los folds
    colors = ['#e6f2ff', '#ffffff']
    for i, v in enumerate(ventanas):
        # Asegurarse de que el span no se salga de los límites de los datos graficados
        start_x = max(v['test_start'], df_wf.index[0])
        end_x = min(v['test_end'], df_wf.index[-1])
        if start_x < end_x:
            ax.axvspan(start_x, end_x, alpha=0.4, color=colors[i % 2])
            ax.axvline(x=start_x, color='grey', linestyle=':', linewidth=1)
            ax.text(start_x + (end_x - start_x)/2, ax.get_ylim()[1] * 0.98, f"Fold {v['fold']}", 
                    ha='center', va='top', fontsize=10, alpha=0.7, fontweight='bold')

    ax.set_title(titulo, fontweight='bold', fontsize=12)
    ax.legend(fontsize=9, loc='upper left')
    ax.grid(True, alpha=0.3)

graficar_wf(ax1, df_oos_simple_final, 'MA Simple Walk-Forward (Curva Continua OOS)', 'MA')
graficar_wf(ax2, df_oos_doble_final, 'MA Doble Walk-Forward (Curva Continua OOS)', 'MA_corta', 'MA_larga')
graficar_wf(ax3, df_oos_rsi_final, 'RSI Walk-Forward (Curva Continua OOS)')

plt.suptitle(f'Rendimiento 100% Out-of-Sample (Walk-Forward Validation) - {NOMBRE_ACTIVO}', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.show()