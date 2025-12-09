"""
RPSAI - Modelo de IA para Piedra, Papel o Tijera
=================================================

INSTRUCCIONES PARA EL ALUMNO:
-----------------------------
Este archivo contiene la plantilla para tu modelo de IA.
Debes completar las secciones marcadas con TODO.

El objetivo es crear un modelo que prediga la PROXIMA jugada del oponente
y responda con la jugada que le gana.

FORMATO DEL CSV (minimo requerido):
-----------------------------------
Tu archivo data/partidas.csv debe tener AL MENOS estas columnas:
    - partida: Numero de la partida
    - ronda: Numero de la ronda (1, 2, 3...)
    - p1: Jugada del jugador 1 (piedra/papel/tijera)
    - p2: Jugada del jugador 2/oponente (piedra/papel/tijera)
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

# =============================================================================
# PARTE 1: EXTRACCION DE DATOS
# =============================================================================

def cargar_datos(ruta_csv: str = None) -> pd.DataFrame:
    if ruta_csv is None:
        ruta_csv = RUTA_DATOS
    return pd.read_csv(ruta_csv)

def preparar_datos(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    
    # Limpiar espacios en blanco
    df['p1'] = df['p1'].str.strip().str.lower()
    df['p2'] = df['p2'].str.strip().str.lower()
    
    # Mapear jugadas a números
    df["jugada_j1_num"] = df["p1"].map(JUGADA_A_NUM)
    df["jugada_j2_num"] = df["p2"].map(JUGADA_A_NUM)
    
    # Ordenar por partida y ronda
    df = df.sort_values(["partida", "ronda"])
    
    # Crear la variable objetivo: proxima jugada de j2
    df["proxima_jugada_j2"] = df.groupby("partida")["jugada_j2_num"].shift(-1)
    
    # Eliminar filas sin valor objetivo (última ronda de cada partida)
    df = df.dropna(subset=["proxima_jugada_j2"]).reset_index(drop=True)
    df["proxima_jugada_j2"] = df["proxima_jugada_j2"].astype(int)
    
    return df

# =============================================================================
# PARTE 2: FEATURE ENGINEERING
# =============================================================================

def crear_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # =========================================================================
    # Feature 1 - Frecuencia de jugadas (con más peso a jugadas recientes)
    # =========================================================================
    for jugada, num in JUGADA_A_NUM.items():
        # Frecuencia histórica general
        df[f"freq_{jugada}_j2"] = (
            df.groupby("partida")["jugada_j2_num"]
            .transform(lambda x: (x.shift(1) == num).expanding().mean())
            .fillna(1/3)
        )
        
        # Frecuencia reciente (últimas 5 jugadas)
        df[f"freq_reciente_{jugada}_j2"] = (
            df.groupby("partida")["jugada_j2_num"]
            .transform(lambda x: (x.shift(1) == num).rolling(window=5, min_periods=1).mean())
            .fillna(1/3)
        )

    # =========================================================================
    # Feature 2 - Lag features (jugadas anteriores)
    # =========================================================================
    df["lag1_j2"] = df.groupby("partida")["jugada_j2_num"].shift(1).fillna(-1).astype(int)
    df["lag2_j2"] = df.groupby("partida")["jugada_j2_num"].shift(2).fillna(-1).astype(int)
    df["lag3_j2"] = df.groupby("partida")["jugada_j2_num"].shift(3).fillna(-1).astype(int)
    df["lag1_j1"] = df.groupby("partida")["jugada_j1_num"].shift(1).fillna(-1).astype(int)
    df["lag2_j1"] = df.groupby("partida")["jugada_j1_num"].shift(2).fillna(-1).astype(int)

    # =========================================================================
    # Feature 3 - Resultado anterior (1=j1 gana, -1=j2 gana, 0=empate)
    # =========================================================================
    def resultado_ronda(row):
        j1 = row["jugada_j1_num"]
        j2 = row["jugada_j2_num"]
        if j1 == j2:
            return 0
        if (j1 == 0 and j2 == 2) or (j1 == 1 and j2 == 0) or (j1 == 2 and j2 == 1):
            return 1  # j1 gana
        return -1  # j2 gana

    df["resultado"] = df.apply(resultado_ronda, axis=1)
    df["resultado_prev"] = df.groupby("partida")["resultado"].shift(1).fillna(0).astype(int)
    df["resultado_prev2"] = df.groupby("partida")["resultado"].shift(2).fillna(0).astype(int)

    # =========================================================================
    # Feature 4 - Empates (MEJORADO)
    # =========================================================================
    df["es_empate"] = (df["jugada_j1_num"] == df["jugada_j2_num"]).astype(int)
    
    # Empates consecutivos (últimas 5 rondas)
    df["empates_consecutivos"] = (
        df.groupby("partida")["es_empate"]
        .transform(lambda x: x.shift(1).rolling(window=5, min_periods=1).sum())
        .fillna(0)
        .astype(int)
    )
    
    # Jugada tras empate simple
    df["jugada_post_empate"] = -1
    empate_mask = df["es_empate"] == 1
    df.loc[empate_mask, "jugada_post_empate"] = (
        df.loc[empate_mask].groupby("partida")["jugada_j2_num"]
        .shift(-1)
        .fillna(-1)
        .astype(int)
    )
    
    # Tiene 2 o más empates en últimas 5 rondas
    df["tiene_empates_multiples"] = (df["empates_consecutivos"] >= 2).astype(int)

    # =========================================================================
    # Feature 5 - Racha de la misma jugada
    # =========================================================================
    df["racha_misma_jugada"] = 0
    for partida in df["partida"].unique():
        mask = df["partida"] == partida
        jugadas = df.loc[mask, "jugada_j2_num"].values
        racha = [0]
        for i in range(1, len(jugadas)):
            if jugadas[i] == jugadas[i-1]:
                racha.append(racha[-1] + 1)
            else:
                racha.append(0)
        df.loc[mask, "racha_misma_jugada"] = racha

    # =========================================================================
    # Feature 6 - Reacción del j2 a jugadas específicas de j1
    # =========================================================================
    df["respuesta_a_piedra"] = -1
    df["respuesta_a_papel"] = -1
    df["respuesta_a_tijera"] = -1
    
    lag1_j1_vals = df.groupby("partida")["jugada_j1_num"].shift(1)
    jugada_actual_j2 = df["jugada_j2_num"]
    
    df.loc[lag1_j1_vals == 0, "respuesta_a_piedra"] = jugada_actual_j2
    df.loc[lag1_j1_vals == 1, "respuesta_a_papel"] = jugada_actual_j2
    df.loc[lag1_j1_vals == 2, "respuesta_a_tijera"] = jugada_actual_j2

    # =========================================================================
    # Feature 7 - Comportamiento tras perder
    # =========================================================================
    df["perdio_j2"] = (df["resultado"] == 1).astype(int)
    perdio_prev = df.groupby("partida")["perdio_j2"].shift(1).fillna(0).astype(bool)
    
    df["jugada_tras_perder"] = -1
    df.loc[perdio_prev, "jugada_tras_perder"] = df.loc[perdio_prev, "jugada_j2_num"]

    # =========================================================================
    # Feature 8 - Cambió de jugada en la ronda anterior
    # =========================================================================
    lag1 = df.groupby("partida")["jugada_j2_num"].shift(1)
    lag2 = df.groupby("partida")["jugada_j2_num"].shift(2)
    df["cambio_jugada"] = (lag1 != lag2).fillna(False).astype(int)

    return df

def seleccionar_features(df: pd.DataFrame) -> tuple:
    feature_cols = [
        # Frecuencias
        "freq_piedra_j2", "freq_papel_j2", "freq_tijera_j2",
        "freq_reciente_piedra_j2", "freq_reciente_papel_j2", "freq_reciente_tijera_j2",
        
        # Lags
        "lag1_j2", "lag2_j2", "lag3_j2",
        "lag1_j1", "lag2_j1",
        
        # Resultados
        "resultado_prev", "resultado_prev2",
        
        # Empates
        "empates_consecutivos", "jugada_post_empate", "tiene_empates_multiples",
        
        # Patrones
        "racha_misma_jugada", "cambio_jugada",
        
        # Reacciones
        "respuesta_a_piedra", "respuesta_a_papel", "respuesta_a_tijera",
        
        # Comportamiento tras perder
        "jugada_tras_perder"
    ]
    
    df = df.dropna(subset=feature_cols + ["proxima_jugada_j2"])
    X = df[feature_cols]
    y = df["proxima_jugada_j2"]
    return X, y

# =============================================================================
# PARTE 3: ENTRENAMIENTO Y FUNCIONAMIENTO
# =============================================================================

def entrenar_modelo(X, y, test_size: float = 0.2):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )

    modelos = {
        'KNN': KNeighborsClassifier(n_neighbors=7),
        'DecisionTree': DecisionTreeClassifier(
            max_depth=10,
            min_samples_split=5,
            random_state=42
        ),
        'RandomForest': RandomForestClassifier(
            #n_estimators=200,
            #max_depth=10,
            #min_samples_split=5,
            #random_state=42
            n_estimators=100,  # Menos árboles
            max_depth=5,       # Más shallow
            min_samples_split=10,
            class_weight='balanced',
            random_state=42
        )
    }

    mejor_modelo = None
    mejor_accuracy = 0
    mejor_nombre = ""

    print("\n" + "="*60)
    print("EVALUACIÓN DE MODELOS")
    print("="*60)

    for nombre, modelo in modelos.items():
        modelo.fit(X_train, y_train)
        y_pred = modelo.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        
        print(f"\n{'─'*60}")
        print(f"Modelo: {nombre}")
        print(f"{'─'*60}")
        print(f"Accuracy: {acc:.4f} ({acc*100:.2f}%)")
        print(f"\nReporte de Clasificación:")
        print(classification_report(y_test, y_pred, 
                                   target_names=['piedra', 'papel', 'tijera'],
                                   zero_division=0))
        print(f"Matriz de Confusión:")
        print(confusion_matrix(y_test, y_pred))

        if acc > mejor_accuracy:
            mejor_accuracy = acc
            mejor_modelo = modelo
            mejor_nombre = nombre

    print("\n" + "="*60)
    print(f"🏆 MEJOR MODELO: {mejor_nombre}")
    print(f"📊 Accuracy: {mejor_accuracy:.4f} ({mejor_accuracy*100:.2f}%)")
    print("="*60 + "\n")
    
    return mejor_modelo

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
# PARTE 4: PREDICCION Y JUEGO
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
                print("⚠️  Modelo no encontrado. Jugando aleatoriamente.")

    def registrar_ronda(self, jugada_humano: str, jugada_ia: str):
        self.historial.append({
            "jugada_j1": jugada_ia,
            "jugada_j2": jugada_humano
        })

    def obtener_features_actuales(self) -> np.ndarray:
        # Si no hay historial, devolver features por defecto
        if not self.historial:
            return np.array([1/3, 1/3, 1/3,  # freq general
                           1/3, 1/3, 1/3,  # freq reciente
                           -1, -1, -1,      # lag j2
                           -1, -1,          # lag j1
                           0, 0,            # resultado prev
                           0, -1, 0,        # empates
                           0, 0,            # racha, cambio
                           -1, -1, -1,      # respuestas
                           -1])             # tras perder

        df_hist = pd.DataFrame(self.historial)
        df_hist["jugada_j1_num"] = df_hist["jugada_j1"].map(JUGADA_A_NUM)
        df_hist["jugada_j2_num"] = df_hist["jugada_j2"].map(JUGADA_A_NUM)

        # Frecuencias generales
        freq_piedra = (df_hist["jugada_j2_num"] == 0).mean()
        freq_papel = (df_hist["jugada_j2_num"] == 1).mean()
        freq_tijera = (df_hist["jugada_j2_num"] == 2).mean()

        # Frecuencias recientes (últimas 5)
        ultimas_5 = df_hist["jugada_j2_num"].tail(5)
        freq_rec_piedra = (ultimas_5 == 0).mean() if len(ultimas_5) > 0 else 1/3
        freq_rec_papel = (ultimas_5 == 1).mean() if len(ultimas_5) > 0 else 1/3
        freq_rec_tijera = (ultimas_5 == 2).mean() if len(ultimas_5) > 0 else 1/3

        # Lags
        lag1_j2 = df_hist["jugada_j2_num"].iloc[-1] if len(df_hist) >= 1 else -1
        lag2_j2 = df_hist["jugada_j2_num"].iloc[-2] if len(df_hist) >= 2 else -1
        lag3_j2 = df_hist["jugada_j2_num"].iloc[-3] if len(df_hist) >= 3 else -1
        lag1_j1 = df_hist["jugada_j1_num"].iloc[-1] if len(df_hist) >= 1 else -1
        lag2_j1 = df_hist["jugada_j1_num"].iloc[-2] if len(df_hist) >= 2 else -1

        # Resultados previos
        def calc_resultado(j1, j2):
            if j1 == j2:
                return 0
            if (j1 == 0 and j2 == 2) or (j1 == 1 and j2 == 0) or (j1 == 2 and j2 == 1):
                return 1
            return -1

        resultado_prev = 0
        resultado_prev2 = 0
        if len(df_hist) >= 2:
            j1_prev = df_hist["jugada_j1_num"].iloc[-2]
            j2_prev = df_hist["jugada_j2_num"].iloc[-2]
            resultado_prev = calc_resultado(j1_prev, j2_prev)
        if len(df_hist) >= 3:
            j1_prev2 = df_hist["jugada_j1_num"].iloc[-3]
            j2_prev2 = df_hist["jugada_j2_num"].iloc[-3]
            resultado_prev2 = calc_resultado(j1_prev2, j2_prev2)

        # Empates consecutivos (últimas 5 rondas)
        empates_ultimas_5 = 0
        if len(df_hist) >= 2:
            for i in range(max(0, len(df_hist)-5), len(df_hist)):
                if df_hist["jugada_j1_num"].iloc[i] == df_hist["jugada_j2_num"].iloc[i]:
                    empates_ultimas_5 += 1

        # Jugada post empate
        jugada_post_empate = -1
        if len(df_hist) >= 2:
            if df_hist["jugada_j1_num"].iloc[-2] == df_hist["jugada_j2_num"].iloc[-2]:
                jugada_post_empate = df_hist["jugada_j2_num"].iloc[-1]

        # Empates múltiples
        tiene_empates_multiples = 1 if empates_ultimas_5 >= 2 else 0

        # Racha misma jugada
        racha = 0
        if len(df_hist) >= 2:
            for i in range(len(df_hist)-1, 0, -1):
                if df_hist["jugada_j2_num"].iloc[i] == df_hist["jugada_j2_num"].iloc[i-1]:
                    racha += 1
                else:
                    break

        # Cambio de jugada
        cambio_jugada = 0
        if len(df_hist) >= 3:
            cambio_jugada = 1 if df_hist["jugada_j2_num"].iloc[-2] != df_hist["jugada_j2_num"].iloc[-3] else 0

        # Respuestas a jugadas específicas
        resp_piedra = resp_papel = resp_tijera = -1
        if len(df_hist) >= 2:
            if df_hist["jugada_j1_num"].iloc[-2] == 0:  # piedra
                resp_piedra = df_hist["jugada_j2_num"].iloc[-1]
            elif df_hist["jugada_j1_num"].iloc[-2] == 1:  # papel
                resp_papel = df_hist["jugada_j2_num"].iloc[-1]
            elif df_hist["jugada_j1_num"].iloc[-2] == 2:  # tijera
                resp_tijera = df_hist["jugada_j2_num"].iloc[-1]

        # Jugada tras perder
        jugada_tras_perder = -1
        if len(df_hist) >= 2 and resultado_prev == 1:  # j2 perdió
            jugada_tras_perder = df_hist["jugada_j2_num"].iloc[-1]

        return np.array([
            freq_piedra, freq_papel, freq_tijera,
            freq_rec_piedra, freq_rec_papel, freq_rec_tijera,
            lag1_j2, lag2_j2, lag3_j2,
            lag1_j1, lag2_j1,
            resultado_prev, resultado_prev2,
            empates_ultimas_5, jugada_post_empate, tiene_empates_multiples,
            racha, cambio_jugada,
            resp_piedra, resp_papel, resp_tijera,
            jugada_tras_perder
        ])

    def predecir_jugada_oponente(self) -> str:
        if self.modelo is None:
            return np.random.choice(["piedra", "papel", "tijera"])
        
        features = self.obtener_features_actuales()
        prediccion_num = self.modelo.predict([features])[0]
        return NUM_A_JUGADA[prediccion_num]

    def decidir_jugada(self) -> str:
        prediccion_oponente = self.predecir_jugada_oponente()
        # Jugamos lo que le gana a la predicción
        return PIERDE_CONTRA[prediccion_oponente]

# =============================================================================
# FUNCION PRINCIPAL
# =============================================================================

def main():
    print("\n" + "="*60)
    print("   🎮 RPSAI - Entrenamiento del Modelo")
    print("="*60 + "\n")

    # 1. Cargar datos
    print("📂 Cargando datos...")
    df = cargar_datos()
    print(f"✅ Datos cargados: {len(df)} filas")

    # 2. Preparar datos
    print("\n🔧 Preparando datos...")
    df_preparado = preparar_datos(df)
    print(f"✅ Datos preparados: {len(df_preparado)} filas")

    # 3. Crear features
    print("\n🎯 Creando features...")
    df_features = crear_features(df_preparado)
    print(f"✅ Features creadas")

    # 4. Seleccionar features
    print("\n📊 Seleccionando features...")
    X, y = seleccionar_features(df_features)
    print(f"✅ Features seleccionadas: {X.shape[1]} columnas, {X.shape[0]} muestras")

    # 5. Entrenar modelo
    print("\n🤖 Entrenando modelos...")
    modelo = entrenar_modelo(X, y)

    # 6. Guardar modelo
    print("\n💾 Guardando modelo...")
    guardar_modelo(modelo)
    
    print("\n" + "="*60)
    print("✅ ENTRENAMIENTO FINALIZADO")
    print("🎮 Modelo listo para usar")
    print("="*60 + "\n")

if __name__ == "__main__":
    main()