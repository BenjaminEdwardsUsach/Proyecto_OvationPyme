#%%
import numpy as np
import cdflib
import matplotlib.pyplot as plt
import json
#%%
# -----------------------------------------------------
# Data loading and filtering function
# -----------------------------------------------------
def cargar_datos(ruta_archivo):
    """
    Loads the necessary variables from a CDF file and applies VALIDMIN/VALIDMAX filters.
    """
    try:
        cdf = cdflib.CDF(ruta_archivo)
        print(f"File loaded: {ruta_archivo}")
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
        return datos, cdf
    except Exception as e:
        print(f"Error loading file {ruta_archivo}: {e}")
        return None, None

# -----------------------------------------------------
# Utility: moving average
# -----------------------------------------------------
def moving_average(data, window=3):
    """Compute the moving average of an array."""
    return np.convolve(data, np.ones(window)/window, mode='valid')

# -----------------------------------------------------
# Electron boundary detection functions
# -----------------------------------------------------
def detectar_b2e_electrons(latitudes, energy_avg, window=3):
    """
    Detects the b2e boundary for electrons as the first point where the gradient
    of the smoothed average energy (ELE_AVERAGE_ENERGY) changes from positive to <= 0.
    """
    smooth_energy = moving_average(energy_avg, window=window)
    # For the latitudes corresponding to the moving average, compute midpoints:
    lat_mid = moving_average(latitudes, window=window)
    grad = np.gradient(smooth_energy, lat_mid)
    for i in range(len(grad) - 1):
        if grad[i] > 0 and grad[i+1] <= 0:
            # Approximate original index offset (i+window//2)
            return lat_mid[i], i + window//2
    return None, None

def detectar_b3a_electrons(latitudes, flux, umbral_factor=5):
    """
    Detects the b3a boundary for electrons (the most equatorward acceleration event)
    by scanning for the first occurrence of a monoenergetic peak in each spectrum.
    """
    indices_detectados = []
    for i, spectrum in enumerate(flux):
        if np.all(np.isnan(spectrum)):
            continue
        max_val = np.nanmax(spectrum)
        max_channel = np.nanargmax(spectrum)
        left = spectrum[max_channel - 1] if max_channel - 1 >= 0 else max_val
        right = spectrum[max_channel + 1] if max_channel + 1 < len(spectrum) else max_val
        if max_val > umbral_factor * left and max_val > umbral_factor * right:
            indices_detectados.append(i)
    if indices_detectados:
        idx = indices_detectados[0]
        return latitudes[idx], idx
    else:
        return None, None

def detectar_b3b_electrons(latitudes, flux, umbral_factor=5):
    """
    Detects the b3b boundary for electrons (the most poleward acceleration event)
    by scanning for the last occurrence of a monoenergetic peak.
    """
    indices_detectados = []
    for i, spectrum in enumerate(flux):
        if np.all(np.isnan(spectrum)):
            continue
        max_val = np.nanmax(spectrum)
        max_channel = np.nanargmax(spectrum)
        left = spectrum[max_channel - 1] if max_channel - 1 >= 0 else max_val
        right = spectrum[max_channel + 1] if max_channel + 1 < len(spectrum) else max_val
        if max_val > umbral_factor * left and max_val > umbral_factor * right:
            indices_detectados.append(i)
    if indices_detectados:
        idx = indices_detectados[-1]
        return latitudes[idx], idx
    else:
        return None, None

def detectar_b4s_electrons(flux, latitudes):
    """
    Detects the b4s boundary for electrons based on the average correlation coefficient
    of each spectrum with its five predecessors dropping below 0.60.
    """
    avg_corr = []
    for i in range(5, len(flux)):
        corr_vals = []
        for j in range(i-5, i):
            # Compute Pearson correlation coefficient
            try:
                corr = np.corrcoef(flux[i], flux[j])[0, 1]
            except Exception:
                corr = np.nan
            corr_vals.append(corr)
        avg_corr.append(np.nanmean(corr_vals))
    avg_corr = np.array(avg_corr)
    for i, corr in enumerate(avg_corr):
        if corr < 0.60:
            # Mapping back to original index
            return latitudes[i+5], i+5
    return None, None

def detectar_b5_electrons(latitudes, flux, window=12):
    """
    Detects the b5 boundary for electrons as the point where the moving average of the flux
    (over 'window' points) drops by a factor of 4.
    """
    # Compute moving average of the flux (averaged over energy channels first)
    avg_flux = np.nanmean(flux, axis=1)
    ma_flux = np.convolve(avg_flux, np.ones(window)/window, mode='valid')
    for i in range(len(ma_flux)-1):
        if ma_flux[i] > 0 and (ma_flux[i] / ma_flux[i+1] >= 4):
            idx = i + window//2  # approximate index
            return latitudes[idx], idx
    return None, None

def detectar_b6_electrons(latitudes, flux, flux_threshold):
    """
    Detects the b6 boundary for electrons as the first point where the average flux
    (over energy channels) falls below a given threshold (flux_threshold).
    """
    avg_flux = np.nanmean(flux, axis=1)
    for i, f in enumerate(avg_flux):
        if f < flux_threshold:
            return latitudes[i], i
    return None, None

# -----------------------------------------------------
# Ion boundary detection function (b2i)
# -----------------------------------------------------
def detectar_b2i_iones(latitudes, flux, energias, bandas_energeticas):
    """
    For ions, b2i is defined as the peak in the integrated energy flux (over the first 'bandas_energeticas'
    channels). The integration uses the energy band widths computed from CHANNEL_ENERGIES.
    """
    # Calculate energy channel widths
    delta = [(energias[i+1] - energias[i]) / 2 for i in range(len(energias)-1)]
    left = energias[1] - energias[0]
    right = energias[-1] - energias[-2]
    delta.insert(0, left)
    delta.append(right)
    delta = np.array(delta)
    
    total_flux = []
    for row in flux:
        if len(row) < bandas_energeticas:
            total_flux.append(np.nan)
        else:
            total_flux.append(np.nansum(row[:bandas_energeticas] * delta[:bandas_energeticas]))
    total_flux = np.array(total_flux)
    idx = np.nanargmax(total_flux)
    return latitudes[idx], idx

# -----------------------------------------------------
# Main processing function
# -----------------------------------------------------
def main():
    # Load configuration from JSON file
    with open('parametros.json', 'r') as file:
        config = json.load(file)
        
    for archivo_config in config['archivos']:
        ruta_archivo = archivo_config["archivo"]
        borde_config = archivo_config["borde"].lower()
        rango = archivo_config["rango_datos"]
        espectro_datos = archivo_config["espectro_datos"]
        bandas_energeticas = archivo_config["bandas_energeticas"]
        seleccion = archivo_config["seleccion_particula"].lower()
        
        print(f"\nProcessing file: {ruta_archivo}")
        datos, cdf = cargar_datos(ruta_archivo)
        if datos is None:
            continue
        
        # Convert time and select data range
        tiempo = cdflib.cdfepoch.to_datetime(datos['Epoch'])
        N1, N2 = rango
        tiempo_rango = tiempo[N1:N2]
        latitudes = datos['SC_GEOCENTRIC_LAT'][N1:N2]
        energias = datos['CHANNEL_ENERGIES']  # assumed common for both particles
        
        # For electrons, we assume the existence of ELE_DIFF_ENERGY_FLUX and ELE_AVERAGE_ENERGY
        if seleccion == "electrons":
            flux = datos['ELE_DIFF_ENERGY_FLUX'][N1:N2]
            # For b2e, we need an energy average; assuming the variable exists:
            if 'ELE_AVG_ENERGY' in datos:
                energy_avg = datos['ELE_AVG_ENERGY'][N1:N2]
            else:
                print("ELE_AVERAGE_ENERGY not found; b2e cannot be computed.")
                energy_avg = None
            
            if borde_config == "b2e" and energy_avg is not None:
                lat_borde, idx_borde = detectar_b2e_electrons(latitudes, energy_avg)
                label_borde = "b2e"
            elif borde_config == "b3a":
                lat_borde, idx_borde = detectar_b3a_electrons(latitudes, flux)
                label_borde = "b3a"
            elif borde_config == "b3b":
                lat_borde, idx_borde = detectar_b3b_electrons(latitudes, flux)
                label_borde = "b3b"
            elif borde_config == "b4s":
                lat_borde, idx_borde = detectar_b4s_electrons(flux, latitudes)
                label_borde = "b4s"
            elif borde_config == "b5":
                lat_borde, idx_borde = detectar_b5_electrons(latitudes, flux)
                label_borde = "b5"
            elif borde_config == "b6":
                # Set a flux threshold (this value may need adjustment)
                flux_threshold = 3.0
                lat_borde, idx_borde = detectar_b6_electrons(latitudes, flux, flux_threshold)
                label_borde = "b6"
            else:
                print(f"[Electrons] Boundary '{borde_config}' not implemented.")
                lat_borde, idx_borde = None, None

            if lat_borde is not None:
                print(f"[Electrons] {label_borde} detected at magnetic latitude {lat_borde:.3f}° at time {tiempo_rango[idx_borde]}")
            else:
                print(f"[Electrons] {label_borde} not detected.")
        
        elif seleccion == "ions":
            flux = datos['ION_DIFF_ENERGY_FLUX'][N1:N2]
            if borde_config == "b2i" or borde_config == "b3a":
                # Even if the JSON says b3a, for ions we apply the b2i algorithm.
                print("[Ions] For ions, b2i detection is recommended. Using b2i algorithm.")
                lat_borde, idx_borde = detectar_b2i_iones(latitudes, flux, energias, bandas_energeticas)
                label_borde = "b2i"
            elif borde_config == "b5":
                # For ions, one might use a similar method to b5 (not fully implemented here).
                lat_borde, idx_borde = detectar_b5_electrons(latitudes, flux)  # reuse electrons method as example
                label_borde = "b5"
            elif borde_config == "b6":
                flux_threshold = 10.0  # example threshold for ions (adjust as needed)
                avg_flux = np.nanmean(flux, axis=1)
                for i, f in enumerate(avg_flux):
                    if f < flux_threshold:
                        lat_borde, idx_borde = latitudes[i], i
                        break
                else:
                    lat_borde, idx_borde = None, None
                label_borde = "b6"
            else:
                print(f"[Ions] Boundary '{borde_config}' not implemented.")
                lat_borde, idx_borde = None, None

            if lat_borde is not None:
                print(f"[Ions] {label_borde} detected at magnetic latitude {lat_borde:.3f}° at time {tiempo_rango[idx_borde]}")
            else:
                print(f"[Ions] {label_borde} not detected.")
        
        else:
            print("Unknown particle selection.")
            continue
        
        # Plotting: plot the average flux vs. magnetic latitude and mark the detected boundary.
        plt.figure(figsize=(12, 7))
        avg_flux = np.nanmean(flux, axis=1)
        plt.plot(latitudes, avg_flux, label=f"Average Differential Flux ({seleccion.capitalize()})")
        if lat_borde is not None:
            plt.axvline(lat_borde, color='r', linestyle='--', label=f"Boundary {label_borde}")
        plt.xlabel("Magnetic Latitude (°)")
        plt.ylabel("Differential Energy Flux")
        plt.title(f"Precipitating {seleccion.capitalize()} - {ruta_archivo}")
        plt.legend()
        plt.grid(True)
        plt.gca().invert_xaxis()
        plt.show()

if __name__ == '__main__':
    main()
