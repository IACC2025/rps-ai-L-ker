"""
RPSAI - Modelo de IA para Piedra, Papel o Tijera
=================================================

SOLUCIÓN PRÁCTICA: Combinación de modelo + estrategia adaptativa
"""

import os
import pickle
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

# Configuracion de rutas
RUTA_PROYECTO = Path(__file__).parent.parent
RUTA_DATOS = RUTA_PROYECTO / "data" / "partidas.csv"
RUTA_MODELO = RUTA_PROYECTO / "models" / "modelo_entrenado.pkl"

# Mapeo de jugadas a numeros
JUGADA_A_NUM = {"piedra": 0, "papel": 1, "tijera": 2}
NUM_A_JUGADA = {0: "piedra", 1: "papel", 2: "tijera"}

# Que jugada gana a cual
GANA_A = {"piedra": "tijera", "papel": "piedra", "tijera": "papel"}
PIERDE_CONTRA = {"piedra": "papel", "papel": "tijera", "tijera": "piedra"}


def cargar_datos(ruta_csv: str = None) -> pd.DataFrame:
    if ruta_csv is None:
        ruta_csv = RUTA_DATOS
    return pd.read_csv(ruta_csv)

def preparar_datos(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # strip vacio para eliminar espacios en blanco y lower por si hay mayusculas
    df['p1'] = df['p1'].str.strip().str.lower()
    df['p2'] = df['p2'].str.strip().str.lower()

    # mapear las jugadas a numeros porque es lo que entiende el ordenador
    df["jugada_j1_num"] = df["p1"].map(JUGADA_A_NUM)
    df["jugada_j2_num"] = df["p2"].map(JUGADA_A_NUM)

    # ordenamos por partidas y rondas
    df = df.sort_values(["partida", "ronda"])

    # creamos la variable objetivo que es la siguiente jugada del otro jugador (victor)
    df["proxima_jugada_j2"] = df.groupby("partida")["jugada_j2_num"].shift(-1)

    # preparando datos de entrenamiento
    df = df.dropna(subset=["proxima_jugada_j2"]).reset_index(drop=True)
    df["proxima_jugada_j2"] = df["proxima_jugada_j2"].astype(int)
    return df

def crear_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    
    # Frecuencias básicas
    for jugada, num in JUGADA_A_NUM.items():
        df[f"freq_{jugada}_j2"] = (
            df.groupby("partida")["jugada_j2_num"]
            .transform(lambda x: (x.shift(1) == num).expanding().mean())
            .fillna(1/3)
        )
    
    # Última jugada (CRÍTICO)
    df["lag1_j2"] = df.groupby("partida")["jugada_j2_num"].shift(1).fillna(-1).astype(int)
    
    return df

def seleccionar_features(df: pd.DataFrame) -> tuple:
    feature_cols = [
        "freq_piedra_j2",
        "freq_papel_j2", 
        "freq_tijera_j2",
        "lag1_j2"
    ]
    
    df = df.dropna(subset=feature_cols + ["proxima_jugada_j2"])
    X = df[feature_cols]
    y = df["proxima_jugada_j2"]
    
    return X, y

def entrenar_modelo(X, y, test_size: float = 0.2):
    print("\n📊 Distribución de clases:")
    print(y.value_counts().sort_index())
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    
    # Solo Random Forest con configuración balanceada
    modelo = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=2,
        min_samples_leaf=1,
        class_weight='balanced',  # CRÍTICO
        random_state=42
    )
    
    modelo.fit(X_train, y_train)
    y_pred = modelo.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    
    print("\n" + "="*60)
    print("EVALUACIÓN DEL MODELO")
    print("="*60)
    print(f"Accuracy: {acc:.4f} ({acc*100:.2f}%)")
    print(f"\nReporte de Clasificación:")
    print(classification_report(y_test, y_pred, 
                               target_names=['piedra', 'papel', 'tijera'],
                               zero_division=0))
    
    return modelo

def guardar_modelo(modelo, ruta: str = None):
    if ruta is None:
        ruta = RUTA_MODELO
    os.makedirs(os.path.dirname(ruta), exist_ok=True)
    with open(ruta, "wb") as f:
        pickle.dump(modelo, f)
    print(f"✅ Modelo guardado en: {ruta}")

def cargar_modelo(ruta: str = None):
    if ruta is None:
        ruta = RUTA_MODELO
    if not os.path.exists(ruta):
        raise FileNotFoundError(f"No se encontro el modelo en: {ruta}")
    with open(ruta, "rb") as f:
        return pickle.load(f)

# =============================================================================
# JUGADOR IA - ESTRATEGIA HÍBRIDA
# =============================================================================

class JugadorIA:
    def __init__(self, ruta_modelo: str = None):
        self.modelo = None
        self.historial = []
        
        if ruta_modelo is not None:
            try:
                self.modelo = cargar_modelo(ruta_modelo)
                print(f"✅ Modelo cargado desde: {ruta_modelo}")
            except FileNotFoundError:
                print("⚠️ Modelo no encontrado. Jugando aleatoriamente.")
    
    def registrar_ronda(self, jugada_humano: str, jugada_ia: str):
        self.historial.append({
            "jugada_j1": jugada_ia,
            "jugada_j2": jugada_humano
        })
    
    def analizar_frecuencias(self) -> dict:
        """Analiza frecuencias de las últimas jugadas"""
        if len(self.historial) < 3:
            return None
            
        jugadas_humano = [h["jugada_j2"] for h in self.historial]
        
        # Últimas 10 jugadas (o todas si hay menos)
        recientes = jugadas_humano[-10:]
        
        conteo = {
            "piedra": recientes.count("piedra"),
            "papel": recientes.count("papel"),
            "tijera": recientes.count("tijera")
        }
        
        total = len(recientes)
        frecuencias = {k: v/total for k, v in conteo.items()}
        
        return frecuencias
    
    def estrategia_counter(self) -> str:
        """Estrategia: jugar lo que más frecuentemente vence al oponente"""
        freqs = self.analizar_frecuencias()
        if not freqs:
            return None
        
        # Encontrar la jugada más común
        jugada_comun = max(freqs, key=freqs.get)
        
        # Si hay una clara tendencia (>40%), countera
        if freqs[jugada_comun] > 0.40:
            return PIERDE_CONTRA[jugada_comun]
        
        return None
    
    def estrategia_anti_patron(self) -> str:
        """Detecta si el oponente repite jugadas"""
        if len(self.historial) < 3:
            return None
        
        ultimas_3 = [h["jugada_j2"] for h in self.historial[-3:]]
        
        # Si jugó lo mismo 2 veces seguidas, asume que cambiará
        if ultimas_3[-1] == ultimas_3[-2]:
            # Predice que cambiará, así que jugamos balanceado
            return np.random.choice(["piedra", "papel", "tijera"])
        
        return None
    
    def predecir_con_modelo(self) -> str:
        """Usa el modelo ML para predecir"""
        if self.modelo is None or len(self.historial) < 1:
            return None
        
        df_hist = pd.DataFrame(self.historial)
        df_hist["jugada_j2_num"] = df_hist["jugada_j2"].map(JUGADA_A_NUM)
        
        # Calcular features
        freq_piedra = (df_hist["jugada_j2_num"] == 0).mean()
        freq_papel = (df_hist["jugada_j2_num"] == 1).mean()
        freq_tijera = (df_hist["jugada_j2_num"] == 2).mean()
        lag1_j2 = df_hist["jugada_j2_num"].iloc[-1] if len(df_hist) >= 1 else -1
        
        features = np.array([[freq_piedra, freq_papel, freq_tijera, lag1_j2]])
        
        try:
            if hasattr(self.modelo, 'predict_proba'):
                probs = self.modelo.predict_proba(features)[0]
                # Usar probabilidades con algo de exploración
                prediccion_num = np.random.choice([0, 1, 2], p=probs)
            else:
                prediccion_num = self.modelo.predict(features)[0]
            
            return NUM_A_JUGADA[prediccion_num]
        except:
            return None
    
    def decidir_jugada(self) -> str:
        """
        ESTRATEGIA HÍBRIDA:
        1. Primeras 5 jugadas: aleatorio (exploración)
        2. Después: combina estrategias
           - 40% modelo ML
           - 30% counter de frecuencias
           - 20% anti-patrón
           - 10% aleatorio
        """
        # Fase 1: Exploración inicial
        if len(self.historial) < 5:
            return np.random.choice(["piedra", "papel", "tijera"])
        
        # Fase 2: Estrategia híbrida
        rand = np.random.random()
        
        if rand < 0.40:
            # 40%: Usa el modelo
            prediccion = self.predecir_con_modelo()
            if prediccion:
                return PIERDE_CONTRA[prediccion]
        
        elif rand < 0.70:
            # 30%: Counter de frecuencias
            jugada = self.estrategia_counter()
            if jugada:
                return jugada
        
        elif rand < 0.90:
            # 20%: Anti-patrón
            jugada = self.estrategia_anti_patron()
            if jugada:
                return jugada
        
        # 10%: Aleatorio (exploración continua)
        return np.random.choice(["piedra", "papel", "tijera"])

def main():
    print("\n" + "="*60)
    print(" 🎮 RPSAI - Entrenamiento del Modelo")
    print("="*60 + "\n")
    
    print("📂 Cargando datos...")
    df = cargar_datos()
    print(f"✅ Datos cargados: {len(df)} filas")
    
    print("\n🔧 Preparando datos...")
    df_preparado = preparar_datos(df)
    print(f"✅ Datos preparados: {len(df_preparado)} filas")
    
    print("\n🎯 Creando features...")
    df_features = crear_features(df_preparado)
    print(f"✅ Features creadas")
    
    print("\n📊 Seleccionando features...")
    X, y = seleccionar_features(df_features)
    print(f"✅ Features seleccionadas: {X.shape[1]} columnas, {X.shape[0]} muestras")
    
    print("\n🤖 Entrenando modelo...")
    modelo = entrenar_modelo(X, y)
    
    print("\n💾 Guardando modelo...")
    guardar_modelo(modelo)
    
    print("\n" + "="*60)
    print("✅ ENTRENAMIENTO FINALIZADO")
    print("🎮 Modelo listo para usar")
    print("\n💡 ESTRATEGIA: Híbrida (40% modelo + 30% counter + 30% adaptativo)")
    print("="*60 + "\n")

if __name__ == "__main__":
    main()