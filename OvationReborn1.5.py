import cdflib
import numpy as np
import matplotlib.pyplot as plt
import json
#%%
def cargar_datos(ruta_archivo):
    """Carga las variables necesarias desde un archivo CDF y aplica filtros de rango basados en VALIDMIN y VALIDMAX."""
    try:
        cdf = cdflib.CDF(ruta_archivo)
        datos = {}
        for variable in cdf.cdf_info().zVariables:
            attrs = cdf.varattsget(variable)
            if 'VALIDMIN' in attrs and 'VALIDMAX' in attrs:
                valid_min = attrs['VALIDMIN']
                valid_max = attrs['VALIDMAX']
                raw_data = cdf.varget(variable)
                datos[variable] = np.where((raw_data >= valid_min) & (raw_data <= valid_max), raw_data, np.nan)
            else:
                datos[variable] = cdf.varget(variable)
        return datos
    except Exception as e:
        print(f"Error al cargar datos: {e}")
        return None

def calcular_b2e(latitudes, energias):
    """Calcula el borde 2be basado en el gradiente de la energía promedio."""
    gradiente = np.gradient(energias, latitudes)
    for i in range(len(gradiente)):
        if gradiente[i] <= 0:
            return latitudes[i], i
    return None, None

def calcular_b2i(latitudes, flujo_total):
    """Calcula el borde 2bi como la latitud donde el flujo total es máximo."""
    max_indice = np.argmax(flujo_total)
    return latitudes[max_indice], max_indice

def calcular_b3a_b3b(latitudes, flujos):
    """Calcula los bordes 3a y 3b en función de los picos monoenergéticos."""
    b3a_lat, b3a_ind, b3b_lat, b3b_ind = None, None, None, None
    for i in range(len(flujos)):
        if np.max(flujos[i]) > 5 * np.mean(flujos[i]):
            if b3a_lat is None:
                b3a_lat, b3a_ind = latitudes[i], i
            b3b_lat, b3b_ind = latitudes[i], i
    return b3a_lat, b3a_ind, b3b_lat, b3b_ind

def calcular_b4s(latitudes, flujos):
    """Calcula el borde b4s basado en la correlación promedio."""
    r_promedios = [np.mean([np.corrcoef(flujos[i], flujos[j])[0, 1] for j in range(i-5, i)]) for i in range(5, len(flujos))]
    for i in range(len(r_promedios) - 6):
        if np.mean(r_promedios[i:i+7]) < 4.0:
            for j in range(i+6, i-1, -1):
                if r_promedios[j] > 0.60:
                    return latitudes[j + 5], j + 5
    return None, None

def calcular_b5e_b5i(tiempo, flujo, tipo):
    """Calcula los bordes b5e o b5i basado en el descenso del flujo precipitado."""
    b5_tiempo, b5_indice = None, None
    ventana = 12  # Número de datos para promediar antes y después
    factor_descenso = 4 # Factor de descenso para detectar el borde
    
    for i in range(ventana, len(flujo) - ventana):
        promedio_anterior = np.mean(flujo[i - ventana:i])
        promedio_siguiente = np.mean(flujo[i:i + ventana])
        
        if promedio_anterior > factor_descenso * promedio_siguiente:
            if tipo == 'electron':
                verificacion = np.mean(flujo[i:i + 35])
                umbral = 10 ** 10.5
            else:
                verificacion = np.mean(flujo[i:i + 30])
                umbral = 10 ** 9.7
            
            if verificacion < umbral:
                b5_tiempo, b5_indice = tiempo[i], i
                break
    
    return b5_tiempo, b5_indice

def graficar_borde(latitudes, valores, borde, titulo, ylabel):
    """Genera un gráfico con el borde detectado."""
    plt.figure(figsize=(12, 6))
    plt.plot(latitudes, valores, label=ylabel)
    if borde is not None:
        plt.axvline(borde, color='r', linestyle='--', label='Borde detectado')
    plt.xlabel('Latitud Magnética (°)')
    plt.ylabel(ylabel)
    plt.title(titulo)
    plt.legend()
    plt.grid()
    plt.show()

def analizar_archivo(filename, rango, espectro, bandas, particula, borde_tipo):
    """Proceso principal para analizar los datos y calcular bordes."""
    datos = cargar_datos(filename)
    if datos is None:
        return
    
    tiempo = datos['Epoch'][rango[0]:rango[1]]
    latitudes = datos['SC_GEOCENTRIC_LAT'][rango[0]:rango[1]]
    energias = datos['CHANNEL_ENERGIES']
    flujo_diferencial = datos['ELE_DIFF_ENERGY_FLUX' if particula == "electrons" else 'ION_DIFF_ENERGY_FLUX'][rango[0]:rango[1]]
    
    flujo_total = np.sum(flujo_diferencial[:, :bandas], axis=1)
    borde = None
    
    if borde_tipo == "2be":
        borde, _ = calcular_b2e(latitudes, flujo_total)
        graficar_borde(latitudes, flujo_total, borde, "Borde 2be", "Flujo Total de Energía")
    elif borde_tipo == "2bi":
        borde, _ = calcular_b2i(latitudes, flujo_total)
        graficar_borde(latitudes, flujo_total, borde, "Borde 2bi", "Flujo Total de Energía")
    elif borde_tipo == "3a" or borde_tipo == "3b":
        b3a, _, b3b, _ = calcular_b3a_b3b(latitudes, flujo_diferencial)
        borde = b3a if borde_tipo == "3a" else b3b
        graficar_borde(latitudes, np.max(flujo_diferencial, axis=1), borde, f"Borde {borde_tipo}", "Flujo Máximo")
    elif borde_tipo == "b4s":
        borde, _ = calcular_b4s(latitudes, flujo_diferencial)
        graficar_borde(latitudes, np.mean(flujo_diferencial, axis=1), borde, "Borde b4s", "Flujo Promedio")
    elif borde_tipo == "b5e":
        borde, _ = calcular_b5e_b5i(tiempo, flujo_total, 'electron')
        graficar_borde(latitudes, np.mean(flujo_diferencial, axis=1), borde, "Borde b5e", "Flujo Promedio")
    elif borde_tipo == "b5i":
        borde, _ = calcular_b5e_b5i(tiempo, flujo_total, 'ion')
        graficar_borde(latitudes, np.mean(flujo_diferencial, axis=1), borde, "Borde b5i", "Flujo Promedio")

    
    print(f"Borde {borde_tipo} encontrado en latitud: {borde}")

# Leer configuración desde parametros.json
with open('parametros.json', 'r') as file:
    data = json.load(file)

for archivo in data['archivos']:
    print(f"Procesando archivo: {archivo['archivo']}")
    analizar_archivo(
        archivo["archivo"],
        archivo["rango_datos"],
        archivo["espectro_datos"],
        archivo["bandas_energeticas"],
        archivo["seleccion_particula"],
        archivo["borde"]
    )
