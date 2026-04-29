"""
=============================================================
  API REST PARA EL CLASIFICADOR DE CORREOS
  Tecnología: FastAPI
  
  Instalación:
    pip install fastapi uvicorn joblib scikit-learn
  
  Ejecutar:
    uvicorn api_clasificador:app --reload --port 8000
  
  Probar en browser:
    http://localhost:8000/docs
=============================================================
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import re
import os
from typing import Optional

# ── Cargar modelo entrenado ───────────────────
MODEL_PATH = "clasificador_correos.pkl"

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"No se encontró el modelo en '{MODEL_PATH}'.\n"
        "Ejecuta primero: python clasificador_correos.py"
    )

pipeline = joblib.load(MODEL_PATH)

# ── Crear aplicación FastAPI ──────────────────
app = FastAPI(
    title="Clasificador de Correos Universitarios",
    description="API para clasificar correos en: tarea, nota, anuncio, evento, logro, intercambio, otro",
    version="1.0.0"
)


# ── Modelos de datos (esquemas) ───────────────
class CorreoRequest(BaseModel):
    texto: str
    asunto: Optional[str] = None  # opcional: asunto del correo

    class Config:
        json_schema_extra = {
            "example": {
                "texto": "Convocatoria abierta para intercambio en España 2026",
                "asunto": "IMPORTANTE: Programa de movilidad académica"
            }
        }


class CorreoResponse(BaseModel):
    categoria: str
    confianza: str
    metodo: str
    probabilidades: dict


# ── Función de limpieza ───────────────────────
def limpiar_texto(texto: str) -> str:
    texto = texto.lower()
    texto = re.sub(r'http\S+|www\S+', '', texto)
    texto = re.sub(r'\S+@\S+', '', texto)
    texto = re.sub(r'[^a-záéíóúñü\s]', '', texto)
    texto = re.sub(r'\s+', ' ', texto).strip()
    return texto


PALABRAS_INTERCAMBIO = [
    "intercambio", "movilidad", "erasmus", "fulbright",
    "beca internacional", "exterior", "extranjero",
    "convocatoria internacional", "daad", "doble titulación"
]


def clasificar(texto: str, asunto: str = "") -> dict:
    texto_completo = f"{asunto} {texto}".strip()
    texto_limpio = limpiar_texto(texto_completo)

    texto_lower = texto_completo.lower()
    for palabra in PALABRAS_INTERCAMBIO:
        if palabra in texto_lower:
            prob = pipeline.predict_proba([texto_limpio])[0]
            clases = pipeline.classes_
            probs_dict = dict(zip(clases, prob))
            if probs_dict.get("intercambio", 0) >= 0.15:
                return {
                    "categoria": "intercambio",
                    "confianza": f"{probs_dict['intercambio']:.2%}",
                    "metodo": "regla + modelo",
                    "probabilidades": {k: f"{v:.2%}" for k, v in
                                       sorted(probs_dict.items(), key=lambda x: -x[1])}
                }

    prob = pipeline.predict_proba([texto_limpio])[0]
    clases = pipeline.classes_
    probs_dict = dict(zip(clases, prob))
    categoria = max(probs_dict, key=probs_dict.get)

    return {
        "categoria": categoria,
        "confianza": f"{probs_dict[categoria]:.2%}",
        "metodo": "modelo",
        "probabilidades": {k: f"{v:.2%}" for k, v in
                           sorted(probs_dict.items(), key=lambda x: -x[1])}
    }


# ── Endpoints ─────────────────────────────────

@app.get("/")
def raiz():
    return {
        "mensaje": "Clasificador de Correos Universitarios",
        "version": "1.0.0",
        "endpoints": {
            "POST /clasificar": "Clasificar un correo",
            "POST /clasificar/lote": "Clasificar múltiples correos",
            "GET /categorias": "Ver categorías disponibles",
            "GET /health": "Estado del servicio"
        }
    }


@app.get("/health")
def health_check():
    return {"estado": "activo", "modelo_cargado": True}


@app.get("/categorias")
def obtener_categorias():
    return {
        "categorias": list(pipeline.classes_),
        "descripcion": {
            "tarea":        "Asignaciones y entregas académicas",
            "nota":         "Calificaciones y resultados de exámenes",
            "anuncio":      "Avisos institucionales y cambios",
            "evento":       "Actividades, conferencias y ceremonias",
            "logro":        "Reconocimientos y premios académicos",
            "intercambio":  "Movilidad académica y becas internacionales ⭐",
            "otro":         "Información general no clasificada"
        }
    }


@app.post("/clasificar", response_model=CorreoResponse)
def clasificar_correo(correo: CorreoRequest):
    if not correo.texto.strip():
        raise HTTPException(status_code=400, detail="El texto del correo no puede estar vacío")
    
    resultado = clasificar(correo.texto, correo.asunto or "")
    return resultado


@app.post("/clasificar/lote")
def clasificar_lote(correos: list[CorreoRequest]):
    if len(correos) > 100:
        raise HTTPException(status_code=400, detail="Máximo 100 correos por lote")
    
    resultados = []
    for i, correo in enumerate(correos):
        resultado = clasificar(correo.texto, correo.asunto or "")
        resultados.append({
            "indice": i,
            "texto_original": correo.texto[:80] + "..." if len(correo.texto) > 80 else correo.texto,
            **resultado
        })
    
    return {
        "total": len(resultados),
        "resultados": resultados,
        "resumen": {
            cat: sum(1 for r in resultados if r["categoria"] == cat)
            for cat in pipeline.classes_
        }
    }


# ── Para ejecutar directamente ────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)