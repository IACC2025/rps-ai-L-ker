"""
RPSAI - Modelo de IA para Piedra, Papel o Tijera
=================================================

Reflexión:
Tras muchas pruebas esta es mi versión definitiva pero me al parecerme un tanto vacía preferí incluir
dentro elementos que había borrado para que vieses el trabajo que he tenido y se debe a lo siguiente.

Al principio confiaba ciegamente en los datos de los compañeros ya que tenian unas 75 partidas más que
yo, el problema que he acabado viendo es que el jugador 2 (contra el que estaba preparando la IA)
jugó mucho piedra y esto junto a todos los patrones que yo intentaba ver generaba un gran overfitting
y mucho ruido así que mi decisión final tras varias pruebas (cada version de la rama dev la pusheo
despues de hacer muchos cambios cuando llego a estar satisfecho con los avances) fue reducir en la
medida de lo posible las features para evitar dichos problemas.

Siento que con las ideas que he tenido habría preferido mucho intentar
hacer un modelo general pero a la altura a la que me encontraba no me veía con el tiempo
suficiente para intentar recolectar datos con suficiente calidad ya que cuanto más avanzaba
en el desarrollo más pude ver el peso real que tienen estos datos sobre el modelo y la calidad de 
estos ya que hasta los proporcionados por los compañeros que han sido recolectados con un buen
método generan grandes problemas de overfitting por las formas de jugar que tienen.
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
    """
    Carga los datos del CSV con las partidas jugadas.
    
    Args:
        ruta_csv: Ruta al archivo CSV. Si es None, usa RUTA_DATOS por defecto.
    
    Returns:
        DataFrame con los datos cargados
    """
    if ruta_csv is None:
        ruta_csv = RUTA_DATOS
    return pd.read_csv(ruta_csv)


def preparar_datos(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepara los datos para el entrenamiento:

    Args:
        df: DataFrame con los datos crudos del CSV
    
    Returns:
        DataFrame preparado para crear features
    """
    df = df.copy()
    
    # Limpiamos espacios en blanco y convertimos a mayúsculas
    df['p1'] = df['p1'].str.strip().str.lower()
    df['p2'] = df['p2'].str.strip().str.lower()
    
    # Mapeamos jugadas a numeros para el entendimiento del modelo
    df["jugada_j1_num"] = df["p1"].map(JUGADA_A_NUM)
    df["jugada_j2_num"] = df["p2"].map(JUGADA_A_NUM)
    
    # Ordenamos por partida y ronda
    df = df.sort_values(["partida", "ronda"])
    
    # Creamos la variable objetivo (próxima jugada de victor)
    df["proxima_jugada_j2"] = df.groupby("partida")["jugada_j2_num"].shift(-1)
    
    # Eliminamos las rondas sin valor objetivo
    df = df.dropna(subset=["proxima_jugada_j2"]).reset_index(drop=True)
    df["proxima_jugada_j2"] = df["proxima_jugada_j2"].astype(int)
    
    return df

# =============================================================================
# PARTE 2: FEATURE ENGINEERING
# =============================================================================

def crear_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Crea features (características) para el modelo de ML.

    Args:
        df: DataFrame preparado con jugadas numéricas
    
    Returns:
        DataFrame con todas las features calculadas
    """
    df = df.copy()
    
    # =========================================================================
    # Feature 1 - Frecuencia de jugadas (GENERAL)
    # Mide que porcentaje de veces el oponente ha jugado cada opcion
    # =========================================================================
    for jugada, num in JUGADA_A_NUM.items():
        # Frecuencia histórica acumulada (expanding = ventana creciente)
        df[f"freq_{jugada}_j2"] = (
            df.groupby("partida")["jugada_j2_num"]
            .transform(lambda x: (x.shift(1) == num).expanding().mean())
            .fillna(1/3)  # Si no hay datos, asumir distribución uniforme
        )
        
        # Frecuencia reciente (últimas 5 jugadas)
        # Captura cambios de estrategia más recientes
        df[f"freq_reciente_{jugada}_j2"] = (
            df.groupby("partida")["jugada_j2_num"]
            .transform(lambda x: (x.shift(1) == num).rolling(window=5, min_periods=1).mean())
            .fillna(1/3)
        )
    
    # =========================================================================
    # Feature 2 - Lag features (jugadas anteriores)
    # Ultima jugada del oponente
    # =========================================================================
    df["lag1_j2"] = df.groupby("partida")["jugada_j2_num"].shift(1).fillna(-1).astype(int)
    
    # Features eliminadas por overfitting:
    # df["lag2_j2"] = ...  # Penúltima jugada - Memorizaba secuencias demasiado especificas
    # df["lag3_j2"] = ...  # Antepenúltima jugada - Producía overfitting
    # df["lag1_j1"] = ...  # Nuestra última jugada - No era muy relevante y podía producir ruido
    # df["lag2_j1"] = ...  # Nuestra penúltima - Lo mismo que la anterior
    
    # =========================================================================
    # Feature 3 - Resultado anterior
    # Era demasiado específica para cada partida por separado
    # =========================================================================
    # def resultado_ronda(row):
    #     j1 = row["jugada_j1_num"]
    #     j2 = row["jugada_j2_num"]
    #     if j1 == j2: return 0
    #     if (j1 == 0 and j2 == 2) or (j1 == 1 and j2 == 0) or (j1 == 2 and j2 == 1):
    #         return 1  # j1 gana
    #     return -1  # j2 gana
    # df["resultado"] = df.apply(resultado_ronda, axis=1)
    # df["resultado_prev"] = ...   # Demasiado específica
    # df["resultado_prev2"] = ...  # Producia overfitting
    
    # =========================================================================
    # Feature 4 - Empates
    # Dependía demasiado del jugador y con la cantidad de datos recoletada producía más ruido que una ayuda
    # =========================================================================
    # df["es_empate"] = ...
    # df["empates_consecutivos"] = ...      # Era poco relevante
    # df["jugada_post_empate"] = ...        # Buscaba patrones muy específicos para la cantidad de datos
    # df["tiene_empates_multiples"] = ...   # Redundante
    
    # =========================================================================
    # Feature 5 - Racha de la misma jugada
    # Repetir jugadas ganadoras
    # =========================================================================
    df["racha_misma_jugada"] = 0
    for partida in df["partida"].unique():
        mask = df["partida"] == partida
        jugadas = df.loc[mask, "jugada_j2_num"].values
        racha = [0]
        for i in range(1, len(jugadas)):
            if jugadas[i] == jugadas[i-1]:
                racha.append(racha[-1] + 1)  # Incrementar racha
            else:
                racha.append(0)  # Resetear racha
        df.loc[mask, "racha_misma_jugada"] = racha
    
    # =========================================================================
    # Feature 6 - Reacción a jugadas específicas
    # Aprenden comportamiento individual del oponente de entrenamiento, producia mucho overfitting
    # y con los datos que tengo (explicado al inicio del documento un poco) producía mucho overfitting
    # =========================================================================
    # df["respuesta_a_piedra"] = ...   
    # df["respuesta_a_papel"] = ...    
    # df["respuesta_a_tijera"] = ...   
    
    # =========================================================================
    # Feature 7 - Comportamiento tras perder
    # Demasiado ajustado al jugador y con esta cantidad de datos muchas veces producía errores
    # =========================================================================
    # df["perdio_j2"] = ...
    # df["jugada_tras_perder"] = ... 
    
    # =========================================================================
    # Feature 8 - Cambió de jugada en la ronda anterior
    # Mide adaptabilidad general del oponente
    # =========================================================================
    lag1 = df.groupby("partida")["jugada_j2_num"].shift(1)
    lag2 = df.groupby("partida")["jugada_j2_num"].shift(2)
    df["cambio_jugada"] = (lag1 != lag2).fillna(False).astype(int)
    
    return df


def seleccionar_features(df: pd.DataFrame) -> tuple:
    """
    Selecciona las features finales para el modelo.
    
    Args:
        df: DataFrame con todas las features calculadas
    
    Returns:
        X: Features para entrenar (matriz)
        y: Variable objetivo (próxima jugada)
    """
    feature_cols = [
        # Frecuencias generales
        "freq_piedra_j2",
        "freq_papel_j2", 
        "freq_tijera_j2",
        
        # Frecuencias recientes
        "freq_reciente_piedra_j2",
        "freq_reciente_papel_j2",
        "freq_reciente_tijera_j2",
        
        # Contexto inmediato (lag)
        "lag1_j2", 
        
        # Patrones de comportamiento
        "racha_misma_jugada", 
        "cambio_jugada"
    ]
    
    # Eliminar filas con valores faltantes en features o objetivo
    df = df.dropna(subset=feature_cols + ["proxima_jugada_j2"])
    
    X = df[feature_cols]
    y = df["proxima_jugada_j2"]
    
    return X, y

# =============================================================================
# PARTE 3: ENTRENAMIENTO Y EVALUACION
# =============================================================================

def entrenar_modelo(X, y, test_size: float = 0.2):
    """
    Entrena el modelo de Random Forest optimizado para datasets pequeños.
    
    CONFIGURACIÓN ANTI-OVERFITTING:
    Args:
        X: Features de entrenamiento
        y: Variable objetivo
        test_size: Porcentaje de datos para test (default 20%)
    
    Returns:
        Modelo entrenado
    """
    # Mostrar distribución de clases (diagnóstico de desbalance)
    print("\nDistribución de clases en el dataset:")
    print(y.value_counts().sort_index())
    print(f"Proporción: {y.value_counts(normalize=True).sort_index().to_dict()}")
    
    # Dividir en train (80%) y test (20%)
    # stratify=y mantiene la misma proporción de clases en ambos conjuntos
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    
    # Random Forest optimizado para dataset pequeño
    modelo = RandomForestClassifier(
        # Tras probar con muchas combinaciones esta es la que mejor resultados me ha dado
        # para enfrentar la excesiva cantidad de piedras que se repiten en los datos de prueba
        # y evitar overfitting
        n_estimators=100,
        max_depth=10,
        min_samples_split=2,
        min_samples_leaf=1,
        class_weight='balanced',
        random_state=42
    )
    
    # Entrenar modelo
    modelo.fit(X_train, y_train)
    
    # Evaluar en conjunto de test
    y_pred = modelo.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    
    # Mostrar resultados
    print("\n" + "="*60)
    print("EVALUACIÓN DEL MODELO")
    print("="*60)
    print(f"Accuracy: {acc:.4f} ({acc*100:.2f}%)")
    print(f"\nReporte de Clasificación:")
    print(classification_report(y_test, y_pred, 
                               target_names=['piedra', 'papel', 'tijera'],
                               zero_division=0))
    print(f"\nMatriz de Confusión:")
    print(confusion_matrix(y_test, y_pred))
    
    return modelo


def guardar_modelo(modelo, ruta: str = None):
    """
    Guarda el modelo entrenado en un archivo .pkl
    
    Args:
        modelo: Modelo entrenado de sklearn
        ruta: Ruta donde guardar. Si es None, usa RUTA_MODELO
    """
    if ruta is None:
        ruta = RUTA_MODELO
    os.makedirs(os.path.dirname(ruta), exist_ok=True)
    with open(ruta, "wb") as f:
        pickle.dump(modelo, f)
    print(f"Modelo guardado en: {ruta}")


def cargar_modelo(ruta: str = None):
    """
    Carga un modelo entrenado desde archivo .pkl
    
    Args:
        ruta: Ruta del modelo. Si es None, usa RUTA_MODELO
    
    Returns:
        Modelo cargado
    """
    if ruta is None:
        ruta = RUTA_MODELO
    if not os.path.exists(ruta):
        raise FileNotFoundError(f"No se encontro el modelo en: {ruta}")
    with open(ruta, "rb") as f:
        return pickle.load(f)

# =============================================================================
# PARTE 4: JUGADOR IA - ESTRATEGIA HÍBRIDA
# =============================================================================

class JugadorIA:
    """
    Jugador IA que combina modelo ML con estrategias adaptativas.
    """
    
    def __init__(self, ruta_modelo: str = None):
        """
        Inicializa el jugador IA.
        
        Args:
            ruta_modelo: Ruta al modelo entrenado. Si es None, juega aleatorio.
        """
        self.modelo = None
        self.historial = []
        
        if ruta_modelo is not None:
            try:
                self.modelo = cargar_modelo(ruta_modelo)
                print(f"Modelo cargado desde: {ruta_modelo}")
            except FileNotFoundError:
                print("Modelo no encontrado. Jugando aleatoriamente.")
    
    def registrar_ronda(self, jugada_humano: str, jugada_ia: str):
        """
        Registra una ronda jugada (se llama después de cada turno).
        
        Args:
            jugada_humano: Lo que jugó el oponente humano
            jugada_ia: Lo que jugó la IA (nosotros)
        """
        self.historial.append({
            "jugada_j1": jugada_ia,
            "jugada_j2": jugada_humano
        })
    
    def analizar_frecuencias(self) -> dict:
        """
        Analiza las frecuencias de las últimas jugadas del oponente.
        
        Returns:
            Dict con frecuencias de cada jugada (piedra/papel/tijera)
            None si hay menos de 3 jugadas
        """
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
        """
        Estrategia: jugar lo que vence a la jugada más común del oponente.
        
        Si el oponente tiene una tendencia clara (>40% de una jugada),
        countera jugando lo que le gana.
        
        Returns:
            Jugada que countera la tendencia, o None si no hay tendencia clara
        """
        freqs = self.analizar_frecuencias()
        if not freqs:
            return None
        
        # Encontrar la jugada más común
        jugada_comun = max(freqs, key=freqs.get)
        
        # Si hay una clara tendencia (>40%) countera
        if freqs[jugada_comun] > 0.40:
            return PIERDE_CONTRA[jugada_comun]
        
        return None
    
    def estrategia_anti_patron(self) -> str:
        """
        Detecta si el oponente repite jugadas consecutivas.
        
        Si jugó lo mismo 2 veces seguidas, asume que podría cambiar,
        así que juega balanceado (aleatorio).
        
        Returns:
            Jugada aleatoria si detecta patrón, None en caso contrario
        """
        if len(self.historial) < 3:
            return None
        
        ultimas_3 = [h["jugada_j2"] for h in self.historial[-3:]]
        
        # Si jugo lo mismo 2 veces seguidas, asume que cambiará
        if ultimas_3[-1] == ultimas_3[-2]:
            # Predice que cambiará, así que jugamos balanceado
            return np.random.choice(["piedra", "papel", "tijera"])
        
        return None
    
    def predecir_con_modelo(self) -> str:
        """
        Usa el modelo ML para predecir la próxima jugada del oponente.
        
        Returns:
            Predicción del modelo, o None si no hay modelo o datos suficientes
        """
        if self.modelo is None or len(self.historial) < 1:
            return None
        
        df_hist = pd.DataFrame(self.historial)
        df_hist["jugada_j2_num"] = df_hist["jugada_j2"].map(JUGADA_A_NUM)
        
        # Calcular features (mismo proceso que en entrenamiento)
        
        # 1. Frecuencias generales
        freq_piedra = (df_hist["jugada_j2_num"] == 0).mean()
        freq_papel = (df_hist["jugada_j2_num"] == 1).mean()
        freq_tijera = (df_hist["jugada_j2_num"] == 2).mean()
        
        # 2. Frecuencias recientes (últimas 5)
        ultimas_5 = df_hist["jugada_j2_num"].tail(5)
        freq_rec_piedra = (ultimas_5 == 0).mean() if len(ultimas_5) > 0 else 1/3
        freq_rec_papel = (ultimas_5 == 1).mean() if len(ultimas_5) > 0 else 1/3
        freq_rec_tijera = (ultimas_5 == 2).mean() if len(ultimas_5) > 0 else 1/3
        
        # 3. Última jugada
        lag1_j2 = df_hist["jugada_j2_num"].iloc[-1] if len(df_hist) >= 1 else -1
        
        # 4. Racha
        racha = 0
        if len(df_hist) >= 2:
            for i in range(len(df_hist)-1, 0, -1):
                if df_hist["jugada_j2_num"].iloc[i] == df_hist["jugada_j2_num"].iloc[i-1]:
                    racha += 1
                else:
                    break
        
        # 5. Cambio de jugada
        cambio_jugada = 0
        if len(df_hist) >= 3:
            cambio_jugada = 1 if df_hist["jugada_j2_num"].iloc[-2] != df_hist["jugada_j2_num"].iloc[-3] else 0
        
        # Construir array de features (mismo orden que en entrenamiento)
        features = np.array([[
            freq_piedra, freq_papel, freq_tijera,
            freq_rec_piedra, freq_rec_papel, freq_rec_tijera,
            lag1_j2,
            racha,
            cambio_jugada
        ]])
        
        try:
            # Usar probabilidades para añadir variabilidad
            if hasattr(self.modelo, 'predict_proba'):
                probs = self.modelo.predict_proba(features)[0]
                # Muestreo probabilístico (no siempre el máximo)
                prediccion_num = np.random.choice([0, 1, 2], p=probs)
            else:
                prediccion_num = self.modelo.predict(features)[0]
            
            return NUM_A_JUGADA[prediccion_num]
        except:
            return None
    
    def decidir_jugada(self) -> str:
        """
        FUNCIÓN PRINCIPAL: Decide qué jugada hacer usando estrategia híbrida.

        Returns:
            La jugada que debe hacer la IA (piedra/papel/tijera)
        """
        # Fase 1: Exploración inicial (primeras 5 jugadas)
        if len(self.historial) < 5:
            return np.random.choice(["piedra", "papel", "tijera"])
        
        # Fase 2: Estrategia híbrida
        rand = np.random.random()
        
        if rand < 0.40:
            # 40%: Usar el modelo ML
            prediccion = self.predecir_con_modelo()
            if prediccion:
                return PIERDE_CONTRA[prediccion]  # Jugar lo que le gana
        
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
        
        # 10%: Aleatorio (mantener impredecibilidad)
        return np.random.choice(["piedra", "papel", "tijera"])

# =============================================================================
# FUNCION PRINCIPAL
# =============================================================================

def main():
    """
    Pipeline completo de entrenamiento:
    1. Cargar datos del CSV
    2. Preparar datos (limpiar, crear objetivo)
    3. Crear features
    4. Seleccionar features finales
    5. Entrenar modelo
    6. Guardar modelo entrenado
    """    
    # 1. Cargar datos
    print("Cargando datos...")
    df = cargar_datos()
    print(f"Datos cargados: {len(df)} filas")
    
    # 2. Preparar datos
    print("\nPreparando datos...")
    df_preparado = preparar_datos(df)
    print(f"Datos preparados: {len(df_preparado)} filas")
    
    # 3. Crear features
    print("\nCreando features...")
    df_features = crear_features(df_preparado)
    print(f"Features creadas")
    
    # 4. Seleccionar features
    print("\nSeleccionando features...")
    X, y = seleccionar_features(df_features)
    print(f"Features seleccionadas: {X.shape[1]} columnas, {X.shape[0]} muestras")
    
    # 5. Entrenar modelo
    print("\nEntrenando modelo...")
    modelo = entrenar_modelo(X, y)
    
    # 6. Guardar modelo
    print("\nGuardando modelo...")
    guardar_modelo(modelo)
    
    print("\n" + "="*60)
    print("ENTRENAMIENTO FINALIZADO")
    print("Modelo listo para usar")
    print("\nESTRATEGIA: Híbrida (40% modelo + 30% counter + 30% adaptativo)")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()