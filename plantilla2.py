# %%
import cdflib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
# =============================================================================
# Funciones de Utilidad
# =============================================================================

def moving_average(a, n=2):
    """
    Calcula el promedio móvil de un arreglo 'a' con una ventana de tamaño 'n'.
    
    Parámetros:
        a (array-like): Datos de entrada.
        n (int): Tamaño de la ventana.
    
    Retorna:
        np.array: Promedio móvil.
    """
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


def detectar_b2i_sliding(flux, window_avg=2, lookahead=10, sliding_window=3, min_flux=10.5):
    """
    Detecta el índice candidato para el límite b2i en la serie 'flux' utilizando
    un promedio en ventana y una búsqueda hacia adelante.
    
    Parámetros:
        flux (array-like): Serie de flujos integrados.
        window_avg (int): Tamaño de la ventana para el promedio candidato.
        lookahead (int): Número de espectros posteriores a considerar.
        sliding_window (int): Tamaño de la ventana para el promedio deslizante.
        min_flux (float): Valor mínimo de flujo para considerar un candidato.
        
    Retorna:
        int o None: Índice del candidato b2i o None si no se encuentra.
    """
    n = len(flux)
    for i in range(n - (window_avg + lookahead) + 1):
        candidate_avg = np.mean(flux[i: i + window_avg])
        # Mensaje de depuración (se puede comentar o eliminar)
        print(f"i={i}, candidate_avg={candidate_avg:.3f}")
        if candidate_avg < min_flux:
            continue
        subsequent = flux[i + window_avg: i + window_avg + lookahead]
        if len(subsequent) < lookahead:
            break
        sliding_avgs = moving_average(subsequent, n=sliding_window)
        max_sliding = np.max(sliding_avgs)
        if candidate_avg > max_sliding:
            print(f"Promedio de {window_avg} espectros ({candidate_avg:.3f}) > Máximo de promedios deslizantes ({max_sliding:.3f})")
            return i
    return None


def load_variable(cdf, varname):
    """
    Carga una variable del archivo CDF aplicando filtros de VALIDMIN y VALIDMAX
    (si se encuentran en los atributos).
    
    Parámetros:
        cdf (cdflib.CDF): Objeto CDF abierto.
        varname (str): Nombre de la variable a cargar.
        
    Retorna:
        np.array: Variable cargada (con NaN en valores fuera del rango válido, si corresponde).
    """
    attrs = cdf.varattsget(varname)
    raw = cdf.varget(varname)
    if 'VALIDMIN' in attrs and 'VALIDMAX' in attrs:
        valid_min = attrs['VALIDMIN']
        valid_max = attrs['VALIDMAX']
        return np.where((raw >= valid_min) & (raw <= valid_max), raw, np.nan)
    return raw


# =============================================================================
# Función Principal
# =============================================================================

def main():
    # -----------------------------
    # Carga de datos del archivo CDF
    # -----------------------------
    cdf_file = 'dmsp-f16_ssjs_precipitating-electrons-ions_20130531000000_20130531230000_cdaweb.cdf'
    # Nota: Ajusta el nombre del archivo según corresponda.
    archivo = cdflib.CDF(cdf_file)
    info = archivo.cdf_info()  # Información general del CDF (opcional)
    
    # Cargar la variable 'Epoch' y convertir a datetime.
    tiempo = load_variable(archivo, 'Epoch')
    # Convertir a datetime usando cdflib; se filtran tiempos “inválidos” (posteriores a 2030)
    tiempo_final = [t for t in cdflib.cdfepoch.to_datetime(tiempo)
                    if t.astype('datetime64[Y]').astype(int) + 1970 < 2030]
    
    # Cargar las demás variables necesarias
    CHANNEL_ENERGIES      = load_variable(archivo, 'CHANNEL_ENERGIES')
    ELE_DIFF_ENERGY_FLUX  = load_variable(archivo, 'ELE_DIFF_ENERGY_FLUX')
    ELE_TOTAL_ENERGY_FLUX = load_variable(archivo, 'ELE_TOTAL_ENERGY_FLUX')  # Serie integrada
    ION_DIFF_ENERGY_FLUX  = load_variable(archivo, 'ION_DIFF_ENERGY_FLUX')
    ION_TOTAL_ENERGY_FLUX = load_variable(archivo, 'ION_TOTAL_ENERGY_FLUX')  # Serie integrada
    SC_AACGM_LAT          = load_variable(archivo, 'SC_AACGM_LAT')
    SC_GEOCENTRIC_LAT     = load_variable(archivo, 'SC_GEOCENTRIC_LAT')
    
    # -----------------------------
    # Filtrado de canales de energía (30 eV a 30000 eV)
    # -----------------------------
    mask_chan = (CHANNEL_ENERGIES >= 30) & (CHANNEL_ENERGIES <= 30000)
    CHANNEL_ENERGIES      = CHANNEL_ENERGIES[mask_chan]
    ELE_DIFF_ENERGY_FLUX  = ELE_DIFF_ENERGY_FLUX[:, mask_chan]
    ION_DIFF_ENERGY_FLUX  = ION_DIFF_ENERGY_FLUX[:, mask_chan]
    # ION_TOTAL_ENERGY_FLUX es 1-D y no depende de canales.
    
    # -----------------------------
    # Cálculo del flujo integrado a partir de ION_DIFF_ENERGY_FLUX
    # Se utiliza la aproximación de integración en los primeros 'canal' canales.
    # -----------------------------
    canal = 6  # Número de canales a usar en la integración
    
    # Se estima la diferencia tomando los puntos intermedios y ajustando extremos.
    Ec2 = CHANNEL_ENERGIES[:-3]
    Ec1 = CHANNEL_ENERGIES[2:-1]

    delta=(Ec2-Ec1)/2
    Left  = (CHANNEL_ENERGIES[0] - CHANNEL_ENERGIES[1])
    Right = (CHANNEL_ENERGIES[-2] - CHANNEL_ENERGIES[-1])
    print(f"antes {delta}")
    delta = np.insert(delta, 0, Left)
    delta = np.append(delta, Right)
    delta = np.array(delta)
    print(f"delta: {delta}")
    
    # Integración del flujo para cada registro temporal usando los primeros 'canal' canales.
    flujos_iones = []
    for elem in ION_DIFF_ENERGY_FLUX:
        flux_value = np.sum(elem[0:canal] * delta[0:canal])
        flujos_iones.append(flux_value)
    flujos_iones = np.array(flujos_iones)

    flujos_electrones = []
    for elem in ELE_DIFF_ENERGY_FLUX:
        flux_value = np.sum(elem[0:canal] * delta[0:canal])
        flujos_electrones.append(flux_value)
    flujos_electrones = np.array(flujos_electrones)
    
    # -----------------------------
    # Filtrado de outliers: Se utiliza el percentil 95 para descartar valores atípicos.
    # -----------------------------
    # Para ION_DIFF_ENERGY_FLUX integrado
    flujos_iones_clean = np.array([kf for kf in flujos_iones if not np.isnan(kf)])
    if len(flujos_iones_clean) == 0:
        raise ValueError("No hay datos válidos de flujo luego de limpiar NaN.")
    percentil_95 = np.percentile(flujos_iones_clean, 95)
    print(f"Percentil 95 (ION_DIFF): {percentil_95:.3f}")
    
    indices_diff = np.where((~np.isnan(flujos_iones)) & (flujos_iones <= percentil_95))[0]
    tiempo_final_filtrado = [tiempo_final[i] for i in indices_diff]
    flujos_iones_filtrado   = flujos_iones[indices_diff]
    
    # Para ION_TOTAL_ENERGY_FLUX
    ion_total_clean = np.array([val for val in ION_TOTAL_ENERGY_FLUX if not np.isnan(val)])
    ion_total_percentil95 = np.percentile(ion_total_clean, 95)
    print(f"Percentil 95 (ION_TOTAL): {ion_total_percentil95:.3f}")
    
    indices_total = np.where((~np.isnan(ION_TOTAL_ENERGY_FLUX)) & (ION_TOTAL_ENERGY_FLUX <= ion_total_percentil95))[0]
    tiempo_total_filtrado = [tiempo_final[i] for i in indices_total]
    ion_total_filtrado    = ION_TOTAL_ENERGY_FLUX[indices_total]
    
    # -----------------------------
    # Separación según latitud (SC_AACGM_LAT)
    # Se consideran “ajustadas” aquellas que estén en un rango definido.
    # -----------------------------
    # Separación según latitud (SC_AACGM_LAT)
    # Se consideran “ajustadas” aquellas que estén en un rango definido.
    adjust_SC_AACGM_LAT = []
    other_SC_AACGM_LAT  = []
    adjust_tiempo_final = []
    other_tiempo_final  = []
    comparador = []
    flag = False

    for i, elem in enumerate(SC_AACGM_LAT):
        if (-75 < elem < -50) or (50 < elem < 75):
            adjust_SC_AACGM_LAT.append(elem)
            adjust_tiempo_final.append(tiempo_final[i])
            if not flag:
                comparador.append((elem, tiempo_final[i]))
                print(f"Latitud: {elem}, Tiempo: {tiempo_final[i]}")
            flag = True
        else:
            if flag:
                comparador.append((elem, tiempo_final[i]))
                print(f"Latitud: {elem}, Tiempo: {tiempo_final[i]}")
            flag = False
            other_SC_AACGM_LAT.append(elem)
            other_tiempo_final.append(tiempo_final[i])
        
    
    
    
    # -----------------------------
    # Detección de extremos en los datos ajustados (cambio de signo)
    # -----------------------------
    extremos = []
    puntos_maximos = []
    extremos.append((adjust_tiempo_final[0], adjust_SC_AACGM_LAT[0]))
    for i in range(len(adjust_SC_AACGM_LAT) - 1):
        # Se detecta un cambio de signo en la latitud
        if (adjust_SC_AACGM_LAT[i] > 0 and adjust_SC_AACGM_LAT[i + 1] < 0) or \
           (adjust_SC_AACGM_LAT[i] < 0 and adjust_SC_AACGM_LAT[i + 1] > 0):
            extremos.append((adjust_tiempo_final[i], adjust_SC_AACGM_LAT[i]))
            extremos.append((adjust_tiempo_final[i + 1], adjust_SC_AACGM_LAT[i + 1]))
    extremos.append((adjust_tiempo_final[-1], adjust_SC_AACGM_LAT[-1]))

    
    # Agrupar extremos en pares (cada par define un segmento de interés)
    pares_extremos = [extremos[i:i + 2] for i in range(0, len(extremos), 2)]
    
    # -----------------------------
    # Selección de segmentos “válidos” según criterio: delta1 < delta2 
    # -----------------------------
    tol = 1  # Tolerancia en unidades de tiempo (por ejemplo, 1 segundo)

    valid_data = []
    for par in pares_extremos:
        # Cada par tiene la forma: [(t_in, lat_in), (t_out, lat_out)]
        t_in = par[0][0]
        t_out = par[1][0]
        count = 0
        # Contar los puntos en "comparador" (de la misma longitud que adjust_tiempo_final)
        # que NO sean 0 y que estén en el intervalo extendido [t_in - tol, t_out + tol]
        for (lat, t) in comparador:
            if (t >= t_in - tol) and (t <= t_out + tol) and (lat != 0):
                count += 1
        # Además, si existe un punto inmediatamente después de t_out en adjust_tiempo_final y 
        # la diferencia es <= tol, se lo suma.
        try:
            idx_out = adjust_tiempo_final.index(t_out)
        except ValueError:
            idx_out = None
        if idx_out is not None and idx_out < len(adjust_tiempo_final) - 1:
            t_next = adjust_tiempo_final[idx_out + 1]
            if (t_next - t_out) <= tol:
                count += 1
        # Si se tienen al menos 3 puntos (no cero) en el intervalo, se toma el par como válido.
        if count >= 3:
            valid_data.append(par)

    print("Valid data (pairs of extremes with >= 3 nonzero comparador points within tolerance):")
    for par in valid_data:
        print(par)
        
    # -----------------------------
    # Graficar series temporales filtradas de ION_DIFF, ION_TOTAL y latitud 
    # -----------------------------
    plt.figure(figsize=(12,6))
    plt.scatter(tiempo_final, SC_AACGM_LAT, s=1)
    plt.scatter(adjust_tiempo_final, adjust_SC_AACGM_LAT, s=1, color='red')

    # Graficar todos los segmentos válidos
    for segment in ( pares_extremos):
        plt.scatter(segment[0][0], segment[0][1], s=50, color='green', label='Valid Segment')
        plt.scatter(segment[1][0], segment[1][1], s=50, color='green')
    for elem in comparador:
        plt.scatter(elem[1],elem[0], s=5, color='black')
    for elem in valid_data:
        plt.scatter(elem[0][0],elem[0][1], s=50, color='yellow')
        plt.scatter(elem[1][0],elem[1][1], s=50, color='yellow')

    plt.xlabel('Time')
    plt.ylabel('Latitude')
    plt.title('Latitude vs Time')
    plt.grid()
    plt.show()

    plt.figure(figsize=(12, 6))
    plt.scatter(tiempo_final_filtrado, flujos_iones_filtrado, s=1)
    plt.xlabel('Time')
    plt.ylabel('Integrated Flux (eV) from ION_DIFF')
    plt.title('Integrated Flux vs Time (ION_DIFF, filtered to 95th percentile)')
    plt.grid()
    plt.show()
    
    plt.figure(figsize=(12, 6))
    plt.scatter(tiempo_total_filtrado, ion_total_filtrado, s=1, color='green')
    plt.xlabel('Time')
    plt.ylabel('Integrated Flux (eV) from ION_TOTAL')
    plt.title('Integrated Flux vs Time (ION_TOTAL, filtered to 95th percentile)')
    plt.grid()
    plt.show()
    
    # -----------------------------
    # Bucle sobre cada segmento válido para detectar b2i y graficar el ciclo
    # Se compara el ION_DIFF (integrado) con ION_TOTAL y se muestran varios paneles.
    # -----------------------------
    contador = 0
    for par in valid_data:
        # Se intenta ubicar los índices en el vector filtrado de tiempo para el par definido.
        try:
            idx_start = tiempo_final_filtrado.index(par[0][0])
            idx_end   = tiempo_final_filtrado.index(par[1][0])
        except ValueError:
            idx_start_array = np.where(np.array(tiempo_final_filtrado) == par[0][0])[0]
            idx_end_array   = np.where(np.array(tiempo_final_filtrado) == par[1][0])[0]
            if len(idx_start_array) == 0 or len(idx_end_array) == 0:
                print(f"Warning: No se encontró el índice para {par[0][0]} o {par[1][0]} en el ciclo {contador}. Se salta el ciclo.")
                continue
            idx_start = idx_start_array[0]
            idx_end   = idx_end_array[0]
        
        if idx_end < idx_start:
            print(f"Warning: idx_end ({idx_end}) < idx_start ({idx_start}) en el ciclo {contador}. Se salta el ciclo.")
            continue
        
        # Extraer el segmento del flujo integrado y tiempos correspondientes.
        segmento_flux = flujos_iones_filtrado[idx_start: idx_end + 1]
        segmento_time = tiempo_final_filtrado[idx_start: idx_end + 1]
        segmento_ion_total = ion_total_filtrado[idx_start: idx_end + 1]
    
        if len(segmento_flux) < (2 + 10):  # Se requiere un segmento suficientemente largo.
            print(f"Warning: Segmento en ciclo {contador} demasiado corto (longitud = {len(segmento_flux)}). Se salta el ciclo.")
            continue
    
        # Se detecta el candidato b2i en el segmento usando la técnica deslizante.
        candidate_local = detectar_b2i_sliding(
            np.array(segmento_flux), window_avg=2, lookahead=10, min_flux=10.5
        )
        if candidate_local is None or candidate_local >= len(segmento_flux):
            print(f"Warning: No se encontró candidato en el ciclo {contador}. Se toma el índice del máximo.")
            candidate_local = np.argmax(segmento_flux)
        if candidate_local < 0 or candidate_local >= len(segmento_flux):
            print(f"Warning: candidate_local = {candidate_local} fuera de rango en el ciclo {contador}. Se salta el ciclo.")
            continue
    
        t_candidate = segmento_time[candidate_local]
        flux_candidate = np.mean(segmento_flux[candidate_local: candidate_local + 2])
        print(f"Cycle {contador}: b2i candidate at time {t_candidate} with integrated Flux {flux_candidate:.3f}")
        
        # Se define la máscara para extraer el segmento en el conjunto completo (basado en tiempo)
        t_seg_start = segmento_time[0]
        t_seg_end   = segmento_time[-1]
        mask = np.array([ (t >= t_seg_start) and (t <= t_seg_end) for t in tiempo_final ])
        time_segment_full = np.array(tiempo_final)[mask]
        
        # Se extrae el segmento correspondiente a ION_TOTAL_ENERGY_FLUX
        ion_total_segment = ION_TOTAL_ENERGY_FLUX[mask]
        time_total_segment = np.array(tiempo_final)[mask]
        
        # -----------------------------
        # Creación de la figura con 2 columnas y 3 filas usando GridSpec.
        # -----------------------------
        fig = plt.figure(constrained_layout=True, figsize=(18, 12))
        gs = fig.add_gridspec(3, 2, height_ratios=[3, 3, 3])
        date_format = mdates.DateFormatter('%H:%M:%S')
        
        # --- Columna Izquierda: ION_DIFF_ENERGY_FLUX ---
        # Panel A (0,0): Gráfico del segmento de Integrated Flux y candidato b2i.
        axA = fig.add_subplot(gs[0, 0])
        axA.plot(segmento_time, segmento_flux, 'b-', label="Integrated Flux (ION_DIFF)")
        axA.scatter(t_candidate, flux_candidate, color='red', s=50, label="b2i Candidate")
        axA.set_ylabel("Integrated Flux (eV)")
        axA.set_title(f"Cycle {contador}: ION_DIFF (Integrated Flux & b2i)")
        axA.legend()
        axA.grid(True)
    
        # Panel B (1,0): Espectrograma de ION_DIFF_ENERGY_FLUX para el segmento.
        # Se utiliza imshow con escala logarítmica para representar el espectro.
        axB = fig.add_subplot(gs[1, 0])
        time_segment_num = mdates.date2num(time_segment_full)
        # Se define el extent:
        #   [xmin, xmax, ymin, ymax] en unidades numéricas de tiempo y energía.
        # Nota: Dependiendo de la orientación deseada se puede invertir el orden.
        extent = [time_segment_num[-1], time_segment_num[0], CHANNEL_ENERGIES[-1], CHANNEL_ENERGIES[0]]
        im = axB.imshow(ION_DIFF_ENERGY_FLUX[mask, :].T, aspect='auto', origin='lower',
                        extent=extent,
                        norm=plt.matplotlib.colors.LogNorm())
        axB.set_yscale('log')
        axB.set_title("Spectrogram - ION_DIFF_ENERGY_FLUX (eV)")
        axB.set_ylabel("Energy (eV)")
        cbar = fig.colorbar(im, ax=axB)
        cbar.set_label("Flux (arb. units)")
    
        # Panel C (2,0): Integrated Flux y Average Ion Energy vs Time (ION_DIFF)
        flux_sum_seg = np.sum(ION_DIFF_ENERGY_FLUX[mask, :canal], axis=1)
        ion_avg_energy_seg = np.zeros_like(flux_sum_seg)
        nonzero_seg = flux_sum_seg > 0
        ion_avg_energy_seg[nonzero_seg] = np.sum(ION_DIFF_ENERGY_FLUX[mask, :canal] * CHANNEL_ENERGIES[:canal],
                                                  axis=1)[nonzero_seg] / flux_sum_seg[nonzero_seg]
        axC = fig.add_subplot(gs[2, 0])
        axC.plot(time_segment_full, flux_sum_seg,
                 label="Integrated Flux (ION_DIFF)", color='blue', linestyle='--')
        axC.set_ylabel("Integrated Flux (eV)")
        axC.set_title("Integrated Flux & Avg Ion Energy (ION_DIFF)")
        # Segundo eje para la energía promedio
        ax_left2 = axC.twinx()
        ax_left2.plot(time_segment_full, ion_avg_energy_seg, label="Average Ion Energy", color='red', linestyle='--')
        ax_left2.set_ylabel("Average Ion Energy (eV)")
        lines1, labels1 = axC.get_legend_handles_labels()
        lines2, labels2 = ax_left2.get_legend_handles_labels()
        axC.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
        axC.grid(True)
        axC.xaxis_date()
        axC.xaxis.set_major_formatter(date_format)
        axC.set_xlabel("Time (UT)")
    
        # --- Columna Derecha: ION_TOTAL_ENERGY_FLUX ---
        # Panel A (0,1): Serie temporal de ION_TOTAL_ENERGY_FLUX para el intervalo.
        axA_total = fig.add_subplot(gs[0, 1])
        axA_total.plot(time_total_segment, ion_total_segment, color='green')
        axA_total.set_title("Time Series - ION_TOTAL_ENERGY_FLUX (eV)")
        axA_total.set_ylabel("Total Energy Flux (eV)")
        axA_total.xaxis_date()
        axA_total.xaxis.set_major_formatter(date_format)
    
        # Panel B (1,1): Serie repetida (se podría omitir o personalizar).
        axB_total = fig.add_subplot(gs[1, 1])
        axB_total.plot(time_total_segment, ion_total_segment, color='green')
        axB_total.set_title("Integrated ION_TOTAL_ENERGY_FLUX vs Time (eV)")
        axB_total.set_ylabel("Total Energy Flux (eV)")
        axB_total.xaxis_date()
        axB_total.xaxis.set_major_formatter(date_format)
    
        # Panel C (2,1): Comparación temporal con dos ejes y:
        # Se grafica el ION_DIFF (integrado) y ION_TOTAL_ENERGY_FLUX en el mismo panel.
        axC_total = fig.add_subplot(gs[2, 1])
        line1, = axC_total.plot(segmento_time, segmento_flux, color='blue', linestyle='--', marker='o', label='Integrated ION_DIFF')
        axC_total.set_ylabel("Integrated ION_DIFF Flux (eV)", color='blue')
        axC_total.tick_params(axis='y', labelcolor='blue')
    
        axC_total_right = axC_total.twinx()
        line2, = axC_total_right.plot(time_total_segment, ion_total_segment, color='green', linestyle='-', marker='x', label='ION_TOTAL')
        axC_total_right.set_ylabel("ION_TOTAL_ENERGY_FLUX (eV)", color='green')
        axC_total_right.tick_params(axis='y', labelcolor='green')
    
        axC_total.set_title("Temporal Comparison: ION_TOTAL vs Integrated ION_DIFF")
        axC_total.set_xlabel("Time (UT)")
        axC_total.xaxis_date()
        axC_total.xaxis.set_major_formatter(date_format)
    
        # Leyenda combinada en el panel
        lines = [line1, line2]
        labels = [l.get_label() for l in lines]
        axC_total.legend(lines, labels, loc='upper right')
    
        fig.align_labels()
        plt.show()
    
        contador += 1


# =============================================================================
# Ejecución del Script
# =============================================================================

if __name__ == '__main__':
    main()


