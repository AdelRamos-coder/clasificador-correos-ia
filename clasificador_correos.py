"""
=============================================================
  CLASIFICADOR DE CORREOS UNIVERSITARIOS
  Institución Universitaria Pascual Bravo
  Tecnologías: Python, Pandas, Scikit-learn

  Categorías:
    tarea, nota, anuncio, evento, logro, intercambio, otro
=============================================================
"""

# ──────────────────────────────────────────────
# 1. IMPORTAR LIBRERÍAS
# ──────────────────────────────────────────────
import re
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
)


# ──────────────────────────────────────────────
# 2. CONFIGURACIÓN
# ──────────────────────────────────────────────
DATASET_PATH = "correos.csv"
MODEL_PATH = "clasificador_correos.pkl"
TEST_SIZE = 0.20
RANDOM_STATE = 42

# Stop-words en español + palabras de "ruido" típicas en correos
STOPWORDS_ES = [
    "de", "la", "el", "los", "las", "un", "una", "y", "o", "a",
    "en", "que", "del", "para", "por", "con", "se", "su", "sus",
    "lo", "les", "le", "es", "ser", "este", "esta", "estos", "estas",
    "al", "muy", "más", "como", "pero", "ya", "fue", "ha", "han",
    "asunto", "estimado", "apreciado", "atentamente", "cordialmente",
    "buenas", "tardes", "días", "noches", "señor", "señora",
]


# ──────────────────────────────────────────────
# 3. LIMPIEZA DE TEXTO
# ──────────────────────────────────────────────
def limpiar_texto(texto: str) -> str:
    """Normaliza un correo eliminando ruido (URLs, emails, símbolos)."""
    if not isinstance(texto, str):
        return ""
    texto = texto.lower()
    texto = re.sub(r"http\S+|www\S+", " ", texto)
    texto = re.sub(r"\S+@\S+", " ", texto)
    texto = re.sub(r"[^a-záéíóúñü\s]", " ", texto)
    texto = re.sub(r"\s+", " ", texto).strip()
    return texto


# ──────────────────────────────────────────────
# 4. FUNCIÓN DE PREDICCIÓN (con regla de negocio)
# ──────────────────────────────────────────────
PALABRAS_INTERCAMBIO = [
    "intercambio", "movilidad", "erasmus", "fulbright",
    "beca internacional", "exterior", "extranjero",
    "convocatoria internacional", "daad", "doble titulación",
    "internacionalización",
]


def clasificar_correo(texto: str, pipeline, umbral_intercambio: float = 0.20) -> dict:
    """Clasifica un correo aplicando reglas + modelo."""
    texto_limpio = limpiar_texto(texto)

    prob = pipeline.predict_proba([texto_limpio])[0]
    clases = pipeline.classes_
    probs_dict = dict(zip(clases, prob))

    # Regla: si hay palabras clave fuertes de intercambio y el modelo
    # le da al menos el umbral, forzamos la categoría intercambio.
    texto_lower = texto.lower()
    for palabra in PALABRAS_INTERCAMBIO:
        if palabra in texto_lower:
            if probs_dict.get("intercambio", 0) >= umbral_intercambio:
                return {
                    "categoria": "intercambio",
                    "confianza": f"{probs_dict['intercambio']:.2%}",
                    "metodo": "regla + modelo",
                    "probabilidades": {
                        k: f"{v:.2%}"
                        for k, v in sorted(probs_dict.items(), key=lambda x: -x[1])
                    },
                }

    categoria = max(probs_dict, key=probs_dict.get)
    return {
        "categoria": categoria,
        "confianza": f"{probs_dict[categoria]:.2%}",
        "metodo": "modelo",
        "probabilidades": {
            k: f"{v:.2%}"
            for k, v in sorted(probs_dict.items(), key=lambda x: -x[1])
        },
    }


# ──────────────────────────────────────────────
# 5. FUNCIÓN PRINCIPAL
# ──────────────────────────────────────────────
def main():

    # ── Cargar datos ─────────────────────────
    df = pd.read_csv(DATASET_PATH)
    df = df[df["categoria"] != "categoria"].reset_index(drop=True)
    df["texto_limpio"] = df["texto"].apply(limpiar_texto)

    print("=" * 60)
    print("  DATASET - CLASIFICADOR PASCUAL BRAVO")
    print("=" * 60)
    print(df["categoria"].value_counts())
    print(f"\nTotal de ejemplos: {len(df)}")
    print()

    print("Ejemplo de limpieza:")
    print(f"  Original:  {df['texto'].iloc[0][:100]}...")
    print(f"  Limpio:    {df['texto_limpio'].iloc[0][:100]}...")
    print()

    # ── Separar train/test ────────────────────
    X = df["texto_limpio"]
    y = df["categoria"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    print(f"Datos de entrenamiento: {len(X_train)} ejemplos")
    print(f"Datos de prueba:        {len(X_test)} ejemplos")
    print()

    # ── Pipeline ──────────────────────────────
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=10000,
            min_df=2,
            max_df=0.85,
            sublinear_tf=True,
            stop_words=STOPWORDS_ES,
        )),
        ("modelo", LogisticRegression(
            max_iter=2000,
            C=1.0,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        )),
    ])

    print("=" * 60)
    print("  ENTRENANDO MODELO...")
    print("=" * 60)
    pipeline.fit(X_train, y_train)
    print("✓ Modelo entrenado exitosamente")
    print()

    # ── Evaluación ───────────────────────────
    y_pred = pipeline.predict(X_test)

    print("=" * 60)
    print("  EVALUACIÓN EN DATOS DE PRUEBA")
    print("=" * 60)
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.2%}\n")

    print("Reporte por categoría:")
    print("-" * 60)
    print(classification_report(y_test, y_pred, zero_division=0))

    print("Matriz de confusión:")
    print("-" * 60)
    cm = confusion_matrix(y_test, y_pred, labels=pipeline.classes_)
    cm_df = pd.DataFrame(cm, index=pipeline.classes_, columns=pipeline.classes_)
    print(cm_df)
    print()

    # ── Validación cruzada ────────────────────
    print("=" * 60)
    print("  VALIDACIÓN CRUZADA (5 folds)")
    print("=" * 60)
    scores = cross_val_score(pipeline, X, y, cv=5, scoring="accuracy")
    print(f"Accuracy promedio: {scores.mean():.2%} ± {scores.std():.2%}")
    print(f"Scores: {[f'{s:.2%}' for s in scores]}")
    print()

    # ── Pruebas con ejemplos reales ────────────
    print("=" * 60)
    print("  PRUEBAS CON EJEMPLOS")
    print("=" * 60)

    ejemplos = [
        "Asunto: Nueva tarea en Moodle - Cálculo II. El profesor publicó el taller 3, entregar el viernes.",
        "Asunto: Calificación publicada Bases de Datos. Tu nota del parcial 2 ya está disponible en SICAU.",
        "Asunto: Convocatoria intercambio académico España 2026. Oficina de Internacionalización del Pascual Bravo.",
        "Asunto: Suspensión clases lunes. Por jornada institucional no hay actividades académicas.",
        "Asunto: Felicitaciones excelencia académica. Mejor promedio de Ingeniería de Sistemas.",
        "Asunto: Beca DAAD para Alemania. Postulaciones abiertas hasta el 30 de mayo.",
        "Asunto: Hackathon Pascual Bravo 2026. Inscripciones abiertas, premios para los mejores equipos.",
        "Asunto: Renovación carné estudiantil. Acércate a la oficina de Bienestar.",
    ]

    for correo in ejemplos:
        resultado = clasificar_correo(correo, pipeline)
        preview = correo[:75] + "..." if len(correo) > 75 else correo
        print(f"\n📧 {preview}")
        print(f"   → Categoría: {resultado['categoria'].upper()}")
        print(f"   → Confianza: {resultado['confianza']} ({resultado['metodo']})")

    # ── Guardar modelo ───────────────────────
    joblib.dump(pipeline, MODEL_PATH)
    print(f"\n\n✓ Modelo guardado en: {MODEL_PATH}")

    # ── Comparar modelos ─────────────────────
    print("\n" + "=" * 60)
    print("  COMPARACIÓN DE MODELOS")
    print("=" * 60)

    modelos = {
        "Regresión Logística": LogisticRegression(max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE),
        "SVM Lineal":          LinearSVC(max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE),
        "Random Forest":       RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=RANDOM_STATE),
    }

    vectorizador = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=10000,
        min_df=2,
        sublinear_tf=True,
        stop_words=STOPWORDS_ES,
    )
    X_vec = vectorizador.fit_transform(X)

    for nombre, modelo in modelos.items():
        scores = cross_val_score(modelo, X_vec, y, cv=5, scoring="accuracy")
        print(f"  {nombre:25s}: {scores.mean():.2%} ± {scores.std():.2%}")

    print("\n" + "=" * 60)
    print("  PROCESO COMPLETADO")
    print("=" * 60)


# ──────────────────────────────────────────────
# PUNTO DE ENTRADA
# ──────────────────────────────────────────────
if __name__ == "__main__":
    main()