import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Setup path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

from src.data_loader import load_csv_data, load_metadata
from src.config import DATA_PATH_RAW, DATA_PATH_PROCESSED

def get_nearest_limit(speed_val):
    """
    Snaps the 85th percentile speed to the nearest standard limit (50, 70, 90).
    """
    limits = [50, 70, 90]
    # Find the limit with minimum difference
    return min(limits, key=lambda x: abs(x - speed_val))

def main():
    print("🚀 Iniciando Calibración de Límites de Velocidad (Regla del Percentil 85)...")
    
    # 1. Load Data
    # To get a good sample, we'll try to load a few months if available around 2018-2019
    # For now, let's stick to the ones we know exist reliably or iterate a list
    months_to_check = ["01-2019", "02-2019", "03-2019"]
    
    combined_dfs = []
    
    for m in months_to_check:
        csv_path = DATA_PATH_RAW / "trafico" / m / f"{m}.csv"
        if csv_path.exists():
            print(f"   > Cargando: {m}...")
            # We only need 'id', 'fecha', 'vmed' to save memory
            try:
                # Load only necessary columns would be ideal but load_csv_data might be simpler
                df = pd.read_csv(csv_path, sep=';', on_bad_lines='skip', usecols=['id', 'fecha', 'vmed'])
                df['fecha'] = pd.to_datetime(df['fecha'], errors='coerce')
                combined_dfs.append(df)
            except Exception as e:
                print(f"   x Error cargando {m}: {e}")
        else:
            print(f"   x Archivo no encontrado: {m}")
    
    if not combined_dfs:
        print("❌ No se encontraron datos para analizar.")
        return

    full_df = pd.concat(combined_dfs, ignore_index=True)
    print(f"📊 Total registros cargados: {len(full_df)}")
    
    # 2. Filter: Tuesdays (weekday=1) AND Hour == 4
    # timestamp.weekday(): Monday=0, Tuesday=1
    print("🧹 Filtrando por Martes a las 04:00 AM (Tráfico Libre)...")
    
    # Extract properties
    full_df['weekday'] = full_df['fecha'].dt.weekday
    full_df['hour'] = full_df['fecha'].dt.hour
    
    # Filter
    mask = (full_df['weekday'] == 1) & (full_df['hour'] == 4) & (full_df['vmed'] > 0)
    free_flow_df = full_df[mask]
    
    print(f"   > Registros de tráfico libre encontrados: {len(free_flow_df)}")
    
    if free_flow_df.empty:
        print("⚠️ No hay suficientes datos en ese horario.")
        return

    # 3. Calculate 85th Percentile per Sensor
    print("🧮 Calculando Percentil 85 por sensor...")
    
    results = []
    
    # Verify M-30 sensors only? 
    # User said "Filtra los datos históricos", implied for M-30 but maybe all?
    # Let's verify against metadata to be safe, filtering only M30?
    # Or just calculate for ALL sensors found in data, and let app filter later.
    # The user manual calculation implies "En cada uno de los sensores". 
    # Let's do all sensors present in the filtered data.
    
    grouped = free_flow_df.groupby('id')
    
    for sensor_id, group in grouped:
        if len(group) < 5: # Skip if too few samples
            continue
            
        v85 = np.percentile(group['vmed'], 85)
        inferred_limit = get_nearest_limit(v85)
        
        # Heuristic: If v85 is very low (e.g. < 40), it might be an exit ramp or error?
        # But we snap to 50 minimum.
        
        results.append({
            'id': sensor_id,
            'v85_observed': round(v85, 2),
            'inferred_limit': inferred_limit,
            'samples': len(group)
        })
        
    results_df = pd.DataFrame(results)
    
    # 4. Save
    output_dir = DATA_PATH_PROCESSED / "realvlimit"
    output_file = output_dir / "sensor_limits.csv"
    
    # Add metadata name if possible?
    # Let's load metadata just to have names for debugging, optional.
    metrics_path = DATA_PATH_RAW / "meta" / "pmed_ubicacion_10_2018.csv"
    if metrics_path.exists():
        meta = load_metadata(metrics_path)
        if not meta.empty and 'id' in meta.columns and 'nombre' in meta.columns:
            results_df = results_df.merge(meta[['id', 'nombre', 'tipo_elem']], on='id', how='left')
    
    # Sort by ID
    results_df = results_df.sort_values('id')
    
    # Ensure directory exists
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results_df.to_csv(output_file, index=False)
    print(f"✅ Resultados guardados en: {output_file}")
    print("\nEjemplo de resultados:")
    print(results_df.head(10))

if __name__ == "__main__":
    main()
