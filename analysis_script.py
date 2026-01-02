import sys
from pathlib import Path
import pandas as pd

# Setup path to import from src
BASE_DIR = Path(__file__).resolve().parent
sys.path.append(str(BASE_DIR))

from src.data_loader import load_csv_data, load_metadata
from src.preprocessor import DataPreprocessor
from src.config import DATA_PATH_RAW

def main():
    print("Iniciando análisis de sensores M-30...")
    
    # 1. Cargar datos (Usamos Marzo 2019 como muestra representativa)
    # Si falla, probar con Enero
    traffic_file = DATA_PATH_RAW / "trafico" / "03-2019" / "03-2019.csv"
    if not traffic_file.exists():
        traffic_file = DATA_PATH_RAW / "trafico" / "01-2019" / "01-2019.csv"
        
    meta_path = DATA_PATH_RAW / "meta" / "pmed_ubicacion_10_2018.csv"
    
    print(f"Cargando datos de tráfico: {traffic_file.name}")
    df = load_csv_data(traffic_file)
    meta_df = load_metadata(meta_path)
    
    # 2. Filtrar solo Sensores M-30
    if 'tipo_elem' in meta_df.columns:
        meta_df = meta_df[meta_df['tipo_elem'] == 'M30']
    
    m30_ids = meta_df['id'].unique()
    print(f"Total sensores M-30 encontrados en metadatos: {len(m30_ids)}")
    
    # 3. Preprocesar (Limpieza y calculo de densidad)
    # Pasar los IDs para que el preprocessor filtre los datos
    preprocessor = DataPreprocessor(sensor_ids=m30_ids)
    df_clean = preprocessor.clean_data(df)
    
    if df_clean.empty:
        print("No hay datos tras el filtrado.")
        return

    df_features = preprocessor.create_features(df_clean)
    
    # 4. Análisis
    # Agrupar por ID y calcular media de Densidad
    print("\nCalculando densidades medias...")
    ranking = df_features.groupby('id')['density'].mean().sort_values(ascending=False)
    
    if ranking.empty:
        print("No se pudieron calcular rankings.")
        return

    # Top 1
    top_id = ranking.index[0]
    top_val = ranking.iloc[0]
    
    # Buscar nombre
    try:
        top_name = meta_df[meta_df['id'] == top_id]['nombre'].values[0]
    except:
        top_name = "Nombre desconocido"
        
    print("\n" + "="*50)
    print(f"RESULTADO: SENSOR CON MAYOR DENSIDAD MEDIA")
    print("="*50)
    print(f"ID Sensor : {top_id}")
    print(f"Ubicación : {top_name}")
    print(f"Densidad  : {top_val:.2f} veh/km")
    print("="*50)
    
    print("\nTop 5 Sensores:")
    for i in range(min(5, len(ranking))):
        pid = ranking.index[i]
        val = ranking.iloc[i]
        try:
            name = meta_df[meta_df['id'] == pid]['nombre'].values[0]
        except:
            name = "N/A"
        print(f"{i+1}. [ID {pid}] {name}: {val:.2f} veh/km")

if __name__ == "__main__":
    main()
