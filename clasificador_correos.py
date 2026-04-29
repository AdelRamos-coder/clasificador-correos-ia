"""
=============================================================
  CLASIFICADOR DE CORREOS UNIVERSITARIOS
  Tecnologías: Python, Pandas, Scikit-learn
  Categorías: tarea, nota, anuncio, evento, logro,
               intercambio, otro
=============================================================
"""

# ──────────────────────────────────────────────
# 1. IMPORTAR LIBRERÍAS
# ──────────────────────────────────────────────
import pandas as pd
import numpy as np
import re
import joblib

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score
)


# ──────────────────────────────────────────────
# 2. DATASET DE ENTRENAMIENTO
#    Mínimo ~10 ejemplos por categoría para empezar.
#    Más datos = mejor precisión.
# ──────────────────────────────────────────────
datos = [
    # ── TAREA ──────────────────────────────────
    ("Se publicó una nueva tarea en Classroom para el lunes", "tarea"),
    ("Recordatorio: entrega de tarea de cálculo mañana", "tarea"),
    ("Nueva actividad disponible en la plataforma educativa", "tarea"),
    ("Debes subir tu proyecto antes del viernes", "tarea"),
    ("Tarea obligatoria: leer capítulos 3 y 4 para el miércoles", "tarea"),
    ("Se asignó trabajo práctico de programación orientada a objetos", "tarea"),
    ("Entrega de práctica de laboratorio este jueves sin excepción", "tarea"),
    ("Actividad grupal publicada en Google Classroom", "tarea"),
    ("Nuevo ejercicio de algoritmos disponible en Moodle", "tarea"),
    ("Recuerda entregar tu informe de proyecto final esta semana", "tarea"),

    # ── NOTA ───────────────────────────────────
    ("Notas del parcial ya disponibles en la plataforma", "nota"),
    ("Se publicaron las calificaciones del examen final", "nota"),
    ("Tu calificación del primer parcial ya está disponible", "nota"),
    ("Resultados del examen de cálculo publicados en el sistema", "nota"),
    ("Puedes consultar tu nota del quiz en el portal estudiantil", "nota"),
    ("Las notas del laboratorio fueron actualizadas hoy", "nota"),
    ("Calificaciones del semestre publicadas, revisa tu situación académica", "nota"),
    ("Resultado del examen recuperatorio ya está en el sistema", "nota"),
    ("Tu promedio fue actualizado luego de la corrección del parcial", "nota"),
    ("Se corrigió tu nota de la exposición oral del proyecto", "nota"),

    # ── ANUNCIO ────────────────────────────────
    ("La facultad comunica cambio en horario de clases", "anuncio"),
    ("Aviso importante: el campus estará cerrado el próximo lunes", "anuncio"),
    ("Comunicado oficial sobre el nuevo reglamento académico", "anuncio"),
    ("Información importante sobre el proceso de matrícula 2026", "anuncio"),
    ("La universidad informa suspensión de clases por paro docente", "anuncio"),
    ("Cambio de aula para la materia de estructuras de datos", "anuncio"),
    ("Aviso: el sistema de notas estará en mantenimiento el viernes", "anuncio"),
    ("Comunicado: nuevas políticas de asistencia para este semestre", "anuncio"),
    ("La dirección académica informa sobre modificaciones al pensum", "anuncio"),
    ("Importante: requisitos actualizados para graduación 2026", "anuncio"),

    # ── EVENTO ─────────────────────────────────
    ("Evento de bienvenida para estudiantes de primer año", "evento"),
    ("Invitación al hackathon universitario este fin de semana", "evento"),
    ("Conferencia sobre inteligencia artificial el próximo jueves", "evento"),
    ("Feria de prácticas profesionales abierta a todos los estudiantes", "evento"),
    ("Semana cultural universitaria del 5 al 9 de mayo", "evento"),
    ("Taller de liderazgo y habilidades blandas, cupos limitados", "evento"),
    ("Ceremonia de graduación programada para el 20 de junio", "evento"),
    ("Workshop de machine learning con casos reales, este sábado", "evento"),
    ("Charla sobre emprendimiento con egresados exitosos", "evento"),
    ("Torneo deportivo universitario, inscripciones abiertas", "evento"),

    # ── LOGRO ──────────────────────────────────
    ("Felicitaciones, obtuviste el mejor promedio de tu carrera", "logro"),
    ("Tu equipo ganó el primer lugar en el concurso de programación", "logro"),
    ("Reconocimiento por excelencia académica del semestre", "logro"),
    ("Fuiste seleccionado para la lista de honor universitaria", "logro"),
    ("Felicitaciones por completar con distinción tu proyecto de grado", "logro"),
    ("Tu artículo fue aceptado en la conferencia internacional de ingeniería", "logro"),
    ("Premio al mejor estudiante del departamento de sistemas", "logro"),
    ("Certificado de excelencia académica disponible para descarga", "logro"),
    ("Tu proyecto fue elegido entre los mejores del semestre", "logro"),
    ("Distinción académica otorgada por el consejo universitario", "logro"),

    # ── INTERCAMBIO ────────────────────────────  ← MUY IMPORTANTE
    ("Convocatoria abierta para intercambio académico en España 2026", "intercambio"),
    ("Programa de movilidad estudiantil hacia universidades de Canadá", "intercambio"),
    ("Beca completa para estudiar un semestre en Alemania, postúlate ya", "intercambio"),
    ("Oportunidad de intercambio internacional en Francia, plazo hasta mayo", "intercambio"),
    ("Convocatoria Erasmus+ para movilidad académica en Europa", "intercambio"),
    ("Programa de intercambio con Universidad de Toronto, cupos disponibles", "intercambio"),
    ("Beca Fulbright para estudios en Estados Unidos, aplica antes del 30", "intercambio"),
    ("Convocatoria para intercambio en universidades latinoamericanas", "intercambio"),
    ("Movilidad académica internacional hacia Brasil y Argentina", "intercambio"),
    ("Abierta convocatoria para programa de doble titulación en México", "intercambio"),
    ("Becas DAAD para estudiar posgrado en Alemania 2026-2027", "intercambio"),
    ("Programa de intercambio con MIT, selección de candidatos abierta", "intercambio"),
    ("Convocatoria para pasantía de investigación en universidades de Asia", "intercambio"),
    ("Beca gobierno de Chile para movilidad estudiantil internacional", "intercambio"),
    ("Oportunidades en el exterior: programa con universidades de Australia", "intercambio"),

    # ── OTRO ───────────────────────────────────
    ("Por favor actualiza tu foto en el portal estudiantil", "otro"),
    ("Encuesta de satisfacción del semestre, tu opinión es importante", "otro"),
    ("Nuevo menú disponible en la cafetería universitaria", "otro"),
    ("Recordatorio: renovar tu carné estudiantil antes de junio", "otro"),
    ("El parqueadero estará habilitado solo hasta las 8pm esta semana", "otro"),
    ("Se habilita nueva red WiFi en el bloque de ingenierías", "otro"),
    ("Encuesta sobre uso de plataformas digitales, toma 5 minutos", "otro"),
    ("Actualización de datos personales requerida en el sistema", "otro"),
    ("La biblioteca amplía su horario durante temporada de exámenes", "otro"),
    ("Nuevo reglamento de uso de laboratorios de computación", "otro"),
]

# Convertir a DataFrame
df = pd.DataFrame(datos, columns=["texto", "categoria"])

print("=" * 55)
print("  DISTRIBUCIÓN DEL DATASET")
print("=" * 55)
print(df["categoria"].value_counts())
print(f"\nTotal de ejemplos: {len(df)}")
print()


# ──────────────────────────────────────────────
# 3. LIMPIEZA DE TEXTO
#    Pasos:
#    a) Convertir a minúsculas
#    b) Eliminar caracteres especiales y números
#    c) Quitar espacios extra
#
#    NOTA: Para español NO eliminamos tildes ni 'ñ'
#    porque son parte del vocabulario y ayudan
#    a distinguir palabras.
# ──────────────────────────────────────────────
def limpiar_texto(texto: str) -> str:
    """
    Limpia un correo electrónico para su procesamiento.

    Args:
        texto: El texto original del correo
    Returns:
        Texto limpio y normalizado
    """
    # a) Minúsculas
    texto = texto.lower()

    # b) Eliminar URLs
    texto = re.sub(r'http\S+|www\S+', '', texto)

    # c) Eliminar correos electrónicos
    texto = re.sub(r'\S+@\S+', '', texto)

    # d) Eliminar caracteres especiales pero conservar letras
    #    españolas (á, é, í, ó, ú, ñ, ü)
    texto = re.sub(r'[^a-záéíóúñü\s]', '', texto)

    # e) Eliminar espacios múltiples
    texto = re.sub(r'\s+', ' ', texto).strip()

    return texto


# Aplicar limpieza al dataset
df["texto_limpio"] = df["texto"].apply(limpiar_texto)

print("Ejemplo de limpieza:")
print(f"  Original:  {df['texto'][0]}")
print(f"  Limpio:    {df['texto_limpio'][0]}")
print()


# ──────────────────────────────────────────────
# 4. SEPARAR DATOS EN ENTRENAMIENTO Y PRUEBA
#    80% para entrenar, 20% para evaluar
#    stratify=y asegura que cada categoría
#    tenga representación en ambos conjuntos
# ──────────────────────────────────────────────
X = df["texto_limpio"]
y = df["categoria"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print(f"Datos de entrenamiento: {len(X_train)} ejemplos")
print(f"Datos de prueba:        {len(X_test)} ejemplos")
print()


# ──────────────────────────────────────────────
# 5. VECTORIZACIÓN CON TF-IDF
#
#    TF-IDF = Term Frequency × Inverse Document Frequency
#
#    - TF: qué tan frecuente es una palabra en UN correo
#    - IDF: qué tan rara es esa palabra en TODOS los correos
#    - Palabras muy comunes ("el", "la") obtienen peso bajo
#    - Palabras específicas ("intercambio", "beca") peso alto
#
#    ngram_range=(1,2): considera palabras solas Y pares
#    "intercambio académico" será una característica propia
# ──────────────────────────────────────────────

# ──────────────────────────────────────────────
# 6. MODELO: Regresión Logística con Pipeline
#
#    Pipeline combina preprocesamiento + modelo en un solo objeto.
#    Ventajas:
#    - Evita errores de data leakage
#    - Más fácil de guardar y reutilizar
#    - Una sola llamada para predecir
#
#    ¿Por qué Logresión Logística?
#    - Funciona muy bien con texto y TF-IDF
#    - Rápido de entrenar
#    - Fácil de interpretar
#    - Buena precisión en clasificación de texto
# ──────────────────────────────────────────────
pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(
        ngram_range=(1, 2),   # unigramas y bigramas
        max_features=5000,    # máximo 5000 características
        min_df=1,             # al menos 1 documento (dataset pequeño)
        sublinear_tf=True,    # escala logarítmica para TF
    )),
    ("modelo", LogisticRegression(
        max_iter=1000,
        C=1.0,                # regularización
        random_state=42
    ))
])


# ──────────────────────────────────────────────
# 7. ENTRENAMIENTO
# ──────────────────────────────────────────────
print("=" * 55)
print("  ENTRENANDO MODELO...")
print("=" * 55)
pipeline.fit(X_train, y_train)
print("✓ Modelo entrenado exitosamente")
print()


# ──────────────────────────────────────────────
# 8. EVALUACIÓN DEL MODELO
# ──────────────────────────────────────────────
y_pred = pipeline.predict(X_test)

print("=" * 55)
print("  EVALUACIÓN EN DATOS DE PRUEBA")
print("=" * 55)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2%}\n")

print("Reporte completo por categoría:")
print("-" * 55)
print(classification_report(y_test, y_pred))

# Validación cruzada (más confiable con dataset pequeño)
print("=" * 55)
print("  VALIDACIÓN CRUZADA (5 folds)")
print("=" * 55)
scores = cross_val_score(pipeline, X, y, cv=5, scoring="accuracy")
print(f"Accuracy promedio: {scores.mean():.2%} ± {scores.std():.2%}")
print(f"Scores individuales: {[f'{s:.2%}' for s in scores]}")
print()


# ──────────────────────────────────────────────
# 9. FUNCIÓN DE PREDICCIÓN
#    Esta es la función que usarás en producción
# ──────────────────────────────────────────────
def clasificar_correo(texto: str, umbral_intercambio: float = 0.20) -> dict:
    """
    Clasifica un correo universitario en su categoría.

    Args:
        texto: El texto del correo a clasificar
        umbral_intercambio: probabilidad mínima para forzar
                            la categoría intercambio cuando
                            hay palabras clave claras

    Returns:
        dict con 'categoria', 'confianza' y 'probabilidades'
    """
    # Palabras clave que indican intercambio con alta certeza
    palabras_intercambio = [
        "intercambio", "movilidad", "erasmus", "fulbright",
        "beca internacional", "exterior", "extranjero",
        "convocatoria internacional", "daad", "doble titulación"
    ]

    texto_limpio = limpiar_texto(texto)

    # Verificar palabras clave de intercambio (regla de negocio)
    texto_lower = texto.lower()
    for palabra in palabras_intercambio:
        if palabra in texto_lower:
            prob = pipeline.predict_proba([texto_limpio])[0]
            clases = pipeline.classes_
            probs_dict = dict(zip(clases, prob))
            # Solo forzar si el modelo no está seguro de otra categoría
            if probs_dict.get("intercambio", 0) >= umbral_intercambio:
                return {
                    "categoria": "intercambio",
                    "confianza": f"{probs_dict['intercambio']:.2%}",
                    "metodo": "regla + modelo",
                    "probabilidades": {k: f"{v:.2%}" for k, v in
                                       sorted(probs_dict.items(),
                                              key=lambda x: -x[1])}
                }

    # Predicción normal del modelo
    prob = pipeline.predict_proba([texto_limpio])[0]
    clases = pipeline.classes_
    probs_dict = dict(zip(clases, prob))

    categoria = max(probs_dict, key=probs_dict.get)
    confianza = probs_dict[categoria]

    return {
        "categoria": categoria,
        "confianza": f"{confianza:.2%}",
        "metodo": "modelo",
        "probabilidades": {k: f"{v:.2%}" for k, v in
                           sorted(probs_dict.items(),
                                  key=lambda x: -x[1])}
    }


# ──────────────────────────────────────────────
# 10. PRUEBAS CON EJEMPLOS REALES
# ──────────────────────────────────────────────
print("=" * 55)
print("  PRUEBAS CON EJEMPLOS")
print("=" * 55)

ejemplos = [
    "Se publicó una nueva tarea en Classroom",
    "Notas del parcial disponibles en la plataforma",
    "Convocatoria abierta para intercambio académico en España",
    "Evento de bienvenida para estudiantes nuevos",
    "Felicitaciones, obtuviste el mejor promedio del semestre",
    "Beca Fulbright para estudios en Estados Unidos, aplica ya",
    "El campus estará cerrado el próximo lunes festivo",
    "Programa Erasmus+ 2026, postulaciones abiertas hasta marzo",
]

for correo in ejemplos:
    resultado = clasificar_correo(correo)
    print(f"\nCorreo: \"{correo}\"")
    print(f"  → Categoría: {resultado['categoria'].upper()}")
    print(f"  → Confianza: {resultado['confianza']}")


# ──────────────────────────────────────────────
# 11. GUARDAR EL MODELO
#     Guarda el pipeline completo (vectorizador + modelo)
#     para reutilizarlo sin reentrenar
# ──────────────────────────────────────────────
joblib.dump(pipeline, "clasificador_correos.pkl")
print("\n\n✓ Modelo guardado en: clasificador_correos.pkl")


# ──────────────────────────────────────────────
# 12. CÓMO CARGAR Y USAR EL MODELO GUARDADO
# ──────────────────────────────────────────────
"""
# En otro script o en tu API:
import joblib

pipeline = joblib.load("clasificador_correos.pkl")

def predecir(texto):
    texto_limpio = limpiar_texto(texto)
    return pipeline.predict([texto_limpio])[0]
"""


# ──────────────────────────────────────────────
# 13. COMPARAR DIFERENTES MODELOS
#     (opcional, para elegir el mejor)
# ──────────────────────────────────────────────
print("\n" + "=" * 55)
print("  COMPARACIÓN DE MODELOS")
print("=" * 55)

modelos = {
    "Regresión Logística": LogisticRegression(max_iter=1000, random_state=42),
    "SVM Lineal":          LinearSVC(max_iter=1000, random_state=42),
    "Random Forest":       RandomForestClassifier(n_estimators=100, random_state=42),
}

vectorizador = TfidfVectorizer(ngram_range=(1, 2), max_features=5000, sublinear_tf=True)
X_vec = vectorizador.fit_transform(X)

for nombre, modelo in modelos.items():
    scores = cross_val_score(modelo, X_vec, y, cv=5, scoring="accuracy")
    print(f"{nombre:25s}: {scores.mean():.2%} ± {scores.std():.2%}")


print("\n" + "=" * 55)
print("  PROCESO COMPLETADO")
print("=" * 55)