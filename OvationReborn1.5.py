#%%
import cdflib
import numpy as np
import matplotlib.pyplot as plt
import json

#%%
def cargar_datos(ruta_archivo):
    """Loads the necessary variables from a CDF file and applies VALIDMIN/VALIDMAX filters."""
    try:
        cdf = cdflib.CDF(ruta_archivo)
        datos = {}
        for variable in cdf.cdf_info().zVariables:
            attrs = cdf.varattsget(variable)
            raw_data = cdf.varget(variable)
            if 'VALIDMIN' in attrs and 'VALIDMAX' in attrs:
                valid_min = attrs['VALIDMIN']
                valid_max = attrs['VALIDMAX']
                datos[variable] = np.where((raw_data >= valid_min) & (raw_data <= valid_max), raw_data, np.nan)
            else:
                datos[variable] = raw_data
        return datos
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def calcular_b2e(latitudes, energias):
    """Calculates the b2e boundary based on the gradient of the average energy."""
    print("Max energy:", max(energias), "Number of latitudes:", len(latitudes))
    gradiente = np.gradient(energias, latitudes)
    for i in range(len(gradiente)):
        if gradiente[i] <= 0:
            return latitudes[i], i
    return None, None

def calcular_b2i(latitudes, flujo_total):
    """Calculates the b2i boundary as the latitude where the total flux is maximum."""
    max_indice = np.argmax(flujo_total)
    return latitudes[max_indice], max_indice

def calcular_b3a_b3b(latitudes, flujos):
    """
    Calculates the b3a and b3b boundaries based on monoenergetic peaks.
    b3a is the most equatorward and b3b the most poleward occurrence.
    """
    b3a_lat, b3a_ind, b3b_lat, b3b_ind = None, None, None, None
    for i in range(len(flujos)):
        # Check if the maximum in this spectrum is significantly larger than the mean (factor 5)
        if np.max(flujos[i]) > 5 * np.mean(flujos[i]):
            if b3a_lat is None:
                b3a_lat, b3a_ind = latitudes[i], i
            b3b_lat, b3b_ind = latitudes[i], i
    return b3a_lat, b3a_ind, b3b_lat, b3b_ind

def calcular_b4s(latitudes, flujos):
    """Calculates the b4s boundary based on the average correlation of each spectrum with its 5 predecessors."""
    r_promedios = []
    for i in range(5, len(flujos)):
        corr_vals = []
        for j in range(i-5, i):
            try:
                corr = np.corrcoef(flujos[i], flujos[j])[0, 1]
            except Exception:
                corr = np.nan
            corr_vals.append(corr)
        r_promedios.append(np.nanmean(corr_vals))
    r_promedios = np.array(r_promedios)
    # Look for a 7-point window where the average correlation drops below 0.60.
    for i in range(len(r_promedios) - 6):
        if np.mean(r_promedios[i:i+7]) < 0.60:
            for j in range(i+6, i-1, -1):
                if r_promedios[j] > 0.60:
                    return latitudes[j + 5], j + 5
    return None, None

def detectar_b5_electrons(latitudes, flux, window=12):
    """
    Detects the b5 boundary for electrons as the point where the moving average of the flux
    (over 'window' points) drops by a factor of 4.
    """
    if flux.ndim == 1:
        avg_flux = flux
    else:
        avg_flux = np.nanmean(flux, axis=1)
    ma_flux = np.convolve(avg_flux, np.ones(window)/window, mode='valid')
    for i in range(len(ma_flux)-1):
        if ma_flux[i] > 0 and (ma_flux[i] / ma_flux[i+1] >= 4):
            idx = i + window//2  # approximate index
            return latitudes[idx], idx
    return None, None

def graficar_borde(latitudes, valores, borde, titulo, ylabel):
    """
    Plots a graph with the detected boundary.
    If the boundary value is None, it will not attempt to plot a vertical line.
    """
    plt.figure(figsize=(12, 6))
    plt.plot(latitudes, valores, label=ylabel)
    
    if isinstance(borde, (list, tuple)):
        # For cases with multiple boundaries (e.g., b3a and b3b)
        for lim in borde:
            if lim is not None:
                plt.axvline(lim, color='r', linestyle='--', label=f'{titulo} = {lim:.2f}°')
    else:
        if borde is not None:
            plt.axvline(borde, color='r', linestyle='--', label=f'{titulo} = {borde:.2f}°')
    
    plt.xlabel('Magnetic Latitude (°)')
    plt.ylabel(ylabel)
    plt.title(titulo)
    plt.legend()
    plt.grid()
    plt.show()

def analizar_archivo(filename, rango, espectro, bandas, particula, borde_tipo):
    """
    Main process for analyzing data and calculating boundaries.
    
    Parameters:
    -----------
    filename: Name of the CDF file to analyze.
    rango: Data range to use.
    espectro: Energy spectrum index to use.
    bandas: Number of energy bands to consider.
    particula: Particle to analyze ('electrons' or 'ions').
    borde_tipo: Type of boundary to calculate.
    """
    datos = cargar_datos(filename)
    if datos is None:
        return
    
    tiempo = datos['Epoch'][rango[0]:rango[1]]
    latitudes = datos['SC_GEOCENTRIC_LAT'][rango[0]:rango[1]]
    if particula == "electrons":
        energias = datos['ELE_AVG_ENERGY'][rango[0]:rango[1]]
        flujos = datos['ELE_DIFF_ENERGY_FLUX'][rango[0]:rango[1]]
    else:
        energias = datos['ION_AVG_ENERGY'][rango[0]:rango[1]]
        flujos = datos['ION_DIFF_ENERGY_FLUX'][rango[0]:rango[1]]
        
    flujo_diferencial = flujos 
    flujo_total = np.sum(flujo_diferencial[:, :bandas], axis=1)
    borde = None
    
    if borde_tipo == "2be":
        borde, _ = calcular_b2e(latitudes, energias)
        graficar_borde(latitudes, energias, borde, borde_tipo, "Average Energy")
    elif borde_tipo == "2bi":
        borde, _ = calcular_b2i(latitudes, flujo_total)
        graficar_borde(latitudes, flujo_total, borde, borde_tipo, "Total Energy Flux")
    elif borde_tipo in ["b3a", "b3b"]:
        b3a, _, b3b, _ = calcular_b3a_b3b(latitudes, flujos)
        if borde_tipo == "b3a":
            borde = b3a
        else:
            borde = b3b
        graficar_borde(latitudes, np.max(flujos, axis=1), borde, borde_tipo, "Maximum Flux")
    elif borde_tipo == "b4s":
        borde, _ = calcular_b4s(latitudes, flujo_diferencial)
        graficar_borde(latitudes, np.mean(flujo_diferencial, axis=1), borde, borde_tipo, "Average Flux")
    elif borde_tipo == "b5":
        borde, _ = detectar_b5_electrons(latitudes, flujo_total)
        graficar_borde(latitudes, np.mean(flujo_diferencial, axis=1), borde, borde_tipo, "Average Flux")
    else:
        print(f"Boundary type '{borde_tipo}' not implemented.")
    
    print(f"Boundary {borde_tipo} found at latitude: {borde}")

# Read configuration from parametros.json
with open('parametros.json', 'r') as file:
    data = json.load(file)

for archivo in data['archivos']:
    print(f"Processing file: {archivo['archivo']}")
    analizar_archivo(
        archivo["archivo"],
        archivo["rango_datos"],
        archivo["espectro_datos"],
        archivo["bandas_energeticas"],
        archivo["seleccion_particula"],
        archivo["borde"]
    )
