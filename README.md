#  Clasificador de Correos con IA

Sistema de Machine Learning que clasifica correos universitarios en diferentes categorías usando procesamiento de lenguaje natural (NLP).

---

##  Descripción

Este proyecto permite analizar automáticamente correos electrónicos y clasificarlos en categorías relevantes para estudiantes, como:

* Tareas
* Notas
* Anuncios
* Eventos
* Logros
* Intercambios académicos
* Otros

El objetivo es organizar y priorizar información importante de manera automática.

---

## Tecnologías utilizadas

* Python
* Pandas
* Scikit-learn
* TF-IDF (Procesamiento de texto)
* Machine Learning (Regresión Logística, SVM, Random Forest)
* FastAPI (API REST)

---

## ¿Cómo funciona?

1. Se limpia el texto del correo
2. Se transforma a vectores numéricos (TF-IDF)
3. El modelo de Machine Learning analiza patrones
4. Se predice la categoría del correo

---

## Ejemplo

Entrada:

```
Convocatoria abierta para intercambio en Canadá
```

Salida:

```
Categoría: intercambio
```

---

## Estructura del proyecto

```
clasificador-correos
 ┣ clasificador_correos.py   # Modelo ML
 ┣ api_clasificador.py       # API con FastAPI
 ┣ correos.csv               # Dataset
 ┣ guia_clasificador.html    # Interfaz de prueba
 ┣ clasificador_correos.pkl  # Modelo entrenado
```

---

## Instalación

```bash
pip install pandas scikit-learn fastapi uvicorn joblib
```

---

## Ejecutar el modelo

```bash
python clasificador_correos.py
```

---

## Ejecutar la API

```bash
uvicorn api_clasificador:app --reload
```

Luego abrir:

```
http://127.0.0.1:8000/docs
```

---

## Ejemplo de uso (Python)

```python
import joblib

modelo = joblib.load("clasificador_correos.pkl")

texto = "Se publicó una nueva tarea en Classroom"
print(modelo.predict([texto]))
```

---

## Estado del proyecto

* ✔️ Modelo funcional
* ✔️ Clasificación multicategoría
* ✔️ API REST
* 🔄 Mejora continua con más datos (75 ejemplos datos, objetivo 1000 datos)

---

## Futuras mejoras

* Integración con Gmail API
* Mejorar precisión del modelo
* Interfaz web completa
* Sistema de priorización de correos

---

## Nota

Este proyecto fue desarrollado como práctica de Machine Learning aplicado a problemas reales.
