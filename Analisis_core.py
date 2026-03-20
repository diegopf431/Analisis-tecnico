# Archivo: analisis_core.py
import pandas as pd
import numpy as np
import yfinance as yf
from itertools import product
from datetime import datetime
import warnings

# Suprimir advertencias para limpiar la salida
warnings.filterwarnings("ignore")

# ========================================================================
# FUNCIONES AUXILIARES (Tus funciones originales de cálculo)
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
        else:
            senales[t] = ultima_senal # Mantener posición
    return pd.DataFrame({'senales': senales}, index=precios.index)

def MA_doble(precios, t_inicial, lookback_corto, lookback_largo):
    H = len(precios)
    senales = np.zeros(H)
    ultima_senal = 0
    MA_corta = np.zeros(H)
    MA_larga = np.zeros(H)
    precios_array = precios.values
    for t in range(t_inicial-1, H):
        MA_corta[t] = np.mean(precios_array[t-lookback_corto+1:t+1])
        MA_larga[t] = np.mean(precios_array[t-lookback_largo+1:t+1])
        if precios_array[t] > MA_corta[t] and precios_array[t] > MA_larga[t] and ultima_senal <= 0:
            senales[t] = 1
            ultima_senal = 1
        elif precios_array[t] < MA_corta[t] and precios_array[t] < MA_larga[t] and ultima_senal >= 0:
            senales[t] = -1
            ultima_senal = -1
        else:
            senales[t] = ultima_senal
    return pd.DataFrame({'senales': senales}, index=precios.index)

def RSI_strategy(precios, t_inicial, periodo_rsi, nivel_sobrecompra, nivel_sobreventa):
    H = len(precios)
    senales = np.zeros(H)
    ultima_senal = 0
    RSI = np.zeros(H)
    precios_array = precios.values.flatten()
    for t in range(t_inicial-1, H):
        ventana = precios_array[t-periodo_rsi+1:t+1]
        cambios = np.diff(ventana)
        alzas = cambios[cambios > 0]
        bajas = -cambios[cambios < 0]
        prom_alzas = np.mean(alzas) if len(alzas) > 0 else 0
        prom_bajas = np.mean(bajas) if len(bajas) > 0 else 0
        
        if prom_bajas == 0: RSI[t] = 100
        else:
            rs = prom_alzas / prom_bajas
            RSI[t] = 100 - (100 / (1 + rs))
            
        if RSI[t] <= nivel_sobreventa and ultima_senal <= 0:
            senales[t] = 1
            ultima_senal = 1
        elif RSI[t] >= nivel_sobrecompra and ultima_senal >= 0:
            senales[t] = -1
            ultima_senal = -1
        else:
            senales[t] = ultima_senal
    return pd.DataFrame({'senales': senales}, index=precios.index)

def calcular_retornos(precios, senales_df, D):
    r = np.log(precios / precios.shift(1))
    senales = senales_df['senales'].values
    ret_combinado = []
    H = len(precios)
    # Simulación simplificada para optimización rápida
    posicion_actual = 0
    for t in range(H-1):
        posicion_actual = senales[t]
        if posicion_actual != 0:
            ret = r.iloc[t+1] * posicion_actual
            if not np.isnan(ret): ret_combinado.append(ret)
    return np.array(ret_combinado)

# ========================================================================
# FUNCIÓN PRINCIPAL QUE LLAMARÁ EL CÓDIGO B
# ========================================================================
def ejecutar_analisis_completo(tickers, pesos=None):
    """
    Descarga datos, crea portafolio, optimiza estrategias y retorna la señal actual.
    """
    
    # 1. GESTIÓN DE PESOS
    if pesos is None:
        pesos = [1.0 / len(tickers)] * len(tickers) # Equiponderado por defecto
    
    # 2. DESCARGA Y CREACIÓN DE PORTAFOLIO
    try:
        print(f"--> Procesando datos para: {tickers}")
        datos = yf.download(tickers, start='2023-01-01', progress=False)['Close']
        datos = datos.dropna()
        
        # Lógica de portafolio
        if isinstance(datos, pd.Series): # Solo un ticker
            precios = datos
        else: # Múltiples tickers
            retornos_norm = datos / datos.iloc[0]
            precios = retornos_norm.dot(pesos) * 100
            
        precios = pd.Series(precios.values, index=datos.index)
        
    except Exception as e:
        return {"error": str(e)}

    # Configuración rápida de rangos (versión reducida para velocidad)
    D = 10
    # Rangos más pequeños para que la decisión sea rápida
    rango_simple = range(10, 51, 5) 
    rango_doble_c = range(5, 31, 5)
    rango_doble_l = range(30, 101, 10)
    rango_rsi_p = range(10, 25, 2)
    rango_rsi_oc = range(65, 81, 5)
    rango_rsi_ov = range(35, 41, 1)

    # 3. OPTIMIZACIÓN "MA SIMPLE"
    best_sharpe = -999
    best_param = 0
    for lb in rango_simple:
        s = MA_simple(precios, lb+1, lb)
        rets = calcular_retornos(precios, s, D)
        if len(rets) > 0 and np.std(rets) > 0:
            sharpe = np.mean(rets)/np.std(rets)
            if sharpe > best_sharpe:
                best_sharpe = sharpe
                best_param = lb
    
    # Obtener señal actual MA Simple con el mejor parámetro
    res_simple = MA_simple(precios, best_param+1, best_param)
    senal_simple_hoy = res_simple['senales'].iloc[-1]

    # 4. OPTIMIZACIÓN "MA DOBLE"
    best_sharpe = -999
    best_params = (0,0)
    for c, l in product(rango_doble_c, rango_doble_l):
        if c >= l: continue
        s = MA_doble(precios, l+1, c, l)
        rets = calcular_retornos(precios, s, D)
        if len(rets) > 0 and np.std(rets) > 0:
            sharpe = np.mean(rets)/np.std(rets)
            if sharpe > best_sharpe:
                best_sharpe = sharpe
                best_params = (c, l)

    res_doble = MA_doble(precios, best_params[1]+1, best_params[0], best_params[1])
    senal_doble_hoy = res_doble['senales'].iloc[-1]

    # 5. OPTIMIZACIÓN "RSI"
    best_sharpe = -999
    best_params_rsi = (0,0,0)
    for p, oc, ov in product(rango_rsi_p, rango_rsi_oc, rango_rsi_ov):
        s = RSI_strategy(precios, p+1, p, oc, ov)
        rets = calcular_retornos(precios, s, D)
        if len(rets) > 0 and np.std(rets) > 0:
            sharpe = np.mean(rets)/np.std(rets)
            if sharpe > best_sharpe:
                best_sharpe = sharpe
                best_params_rsi = (p, oc, ov)
    
    res_rsi = RSI_strategy(precios, best_params_rsi[0]+1, *best_params_rsi)
    senal_rsi_hoy = res_rsi['senales'].iloc[-1]

    # Retornar diccionario con resultados
    return {
        "precio_actual": precios.iloc[-1],
        "ma_simple": {"senal": senal_simple_hoy, "param": best_param},
        "ma_doble": {"senal": senal_doble_hoy, "param": best_params},
        "rsi": {"senal": senal_rsi_hoy, "param": best_params_rsi}
    }