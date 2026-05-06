# 📧 Clasificador de Correos con IA — Pascual Bravo

Sistema de Machine Learning que clasifica automáticamente correos universitarios de la **Institución Universitaria Pascual Bravo** en 7 categorías usando procesamiento de lenguaje natural (NLP).

---

## 📋 Descripción

Este proyecto analiza correos electrónicos universitarios y los clasifica automáticamente, ayudando a los estudiantes a organizar y priorizar la información que reciben.

### Categorías

| Categoría | Descripción |
|-----------|-------------|
| 📚 **tarea** | Asignaciones, talleres, entregas, quizzes |
| 📝 **nota** | Calificaciones y resultados de exámenes |
| 📢 **anuncio** | Avisos institucionales, cambios, comunicados |
| 🗓️ **evento** | Conferencias, hackathons, talleres, ceremonias |
| 🏆 **logro** | Reconocimientos, premios, distinciones |
| ✈️ **intercambio** | Movilidad académica, becas internacionales |
| 📌 **otro** | Información general (carné, transporte, encuestas) |

---

## 🛠️ Tecnologías utilizadas

- **Python 3.10+**
- **Pandas** — manipulación de datos
- **Scikit-learn** — modelos de Machine Learning
- **TF-IDF** — vectorización de texto
- **FastAPI** — API REST
- **Uvicorn** — servidor ASGI

---

## ⚙️ ¿Cómo funciona?

1. El correo entra como texto (asunto + cuerpo).
2. Se **limpia** (minúsculas, sin URLs, sin emails, sin símbolos).
3. Se transforma a **vectores TF-IDF** (unigramas + bigramas).
4. Un modelo de **Regresión Logística** predice la categoría.
5. Una **regla de negocio** corrige casos especiales para detectar oportunidades de intercambio.

---

## 📂 Estructura del proyecto

```
clasificador-correos/
 ┣ clasificador_correos.py   # Entrenamiento del modelo
 ┣ api_clasificador.py       # API REST con FastAPI
 ┣ correos.csv               # Dataset (1000 correos)
 ┣ guia_clasificador.html    # Interfaz web de prueba
 ┗ clasificador_correos.pkl  # Modelo entrenado (generado)
```

---

## 🚀 Instalación

```bash
pip install pandas scikit-learn fastapi uvicorn joblib
```

---

## ▶️ Uso

### 1. Entrenar el modelo

```bash
python clasificador_correos.py
```

Esto entrenará el modelo con los 1000 correos del dataset y generará `clasificador_correos.pkl`.

### 2. Ejecutar la API

```bash
uvicorn api_clasificador:app --reload --port 8000
```

Luego abre: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

### 3. Usar el modelo desde Python

```python
import joblib
modelo = joblib.load("clasificador_correos.pkl")

texto = "Convocatoria abierta para intercambio en Canadá 2026"
prediccion = modelo.predict([texto])
print(prediccion)  # ['intercambio']
```

---

## 📊 Dataset

El archivo `correos.csv` contiene **1000 correos sintéticos** generados con el contexto real del Pascual Bravo: profesores, materias (Cálculo, Programación, Circuitos, Termodinámica, etc.), plataformas (Moodle, SICAU, Teams) y oficinas institucionales.

| Categoría | Cantidad |
|-----------|----------|
| tarea | 180 |
| anuncio | 160 |
| nota | 150 |
| otro | 150 |
| evento | 140 |
| intercambio | 130 |
| logro | 90 |
| **Total** | **1000** |

---

## 📈 Métricas del modelo

- **Accuracy en test**: ~100%
- **Validación cruzada (5 folds)**: 99.9% ± 0.2%
- Modelos comparados: Regresión Logística, SVM Lineal, Random Forest

> **Nota**: la alta precisión se debe a que el dataset es sintético y bastante consistente. Con correos reales del mundo se espera una precisión entre 85% y 95%.

---

## 🎯 Estado del proyecto

- ✔️ Modelo funcional con 1000 ejemplos
- ✔️ Clasificación multicategoría (7 clases)
- ✔️ API REST con FastAPI
- ✔️ Regla de negocio para detectar intercambios
- ✔️ Validación cruzada y comparación de modelos
- 🔄 Pendiente: integración con Gmail API

---

## 🔮 Futuras mejoras

- [ ] Integración con Gmail API para clasificar la bandeja real
- [ ] Sistema de priorización (correos urgentes primero)
- [ ] Clasificación automática en carpetas/etiquetas
- [ ] Modelo más avanzado con embeddings (BERT en español)
- [ ] Interfaz web completa para monitoreo

---

## 📝 Nota

Proyecto desarrollado como práctica de Machine Learning aplicado a problemas reales de la vida universitaria en la **Institución Universitaria Pascual Bravo**.