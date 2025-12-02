import os
import io
import logging
from typing import Optional, Dict, Any

import numpy as np
import joblib
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image
from tensorflow.keras.models import load_model

# -----------------------
# Config / Logging
# -----------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("tb-api")

# configurable model paths (useful on Render / Heroku / etc)
IMG_MODEL_PATH = os.getenv("IMG_MODEL_PATH", "models/MD_ETB.h5")
RF_MODEL_PATH = os.getenv("RF_MODEL_PATH", "models/MD_RF.joblib")

# model image size expected by MD_ETB  (adjust if different)
IMG_SIZE = (300, 300)  # (width, height)

app = FastAPI(title="TB Diagnostic API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # em produção restrinja ao domínio do front
    allow_methods=["*"],
    allow_headers=["*"],
)


# -----------------------
# Load models (on startup)
# -----------------------
try:
    logger.info(f"Loading image model from {IMG_MODEL_PATH} ...")
    modelo_imagem = load_model(IMG_MODEL_PATH)
    logger.info("Image model loaded.")
except Exception as e:
    logger.exception("Failed to load image model.")
    modelo_imagem = None

try:
    logger.info(f"Loading symptoms model from {RF_MODEL_PATH} ...")
    modelo_sintomas = joblib.load(RF_MODEL_PATH)
    logger.info("Symptoms model loaded.")
except Exception as e:
    logger.exception("Failed to load symptoms model.")
    modelo_sintomas = None


# -----------------------
# Helper functions
# -----------------------
def process_image_file(file_bytes: bytes) -> np.ndarray:
    """
    Read bytes, convert to RGB, resize to IMG_SIZE and normalize to [0,1].
    Returns a batch of shape (1, H, W, 3).
    """
    try:
        img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail="Invalid image file.")
    img = img.resize(IMG_SIZE)
    arr = np.array(img).astype("float32") / 255.0
    # ensure shape (1, H, W, 3)
    arr = np.expand_dims(arr, axis=0)
    return arr


def build_symptoms_array(symptoms: list[int]) -> np.ndarray:
    """
    Convert list of binary/int symptoms into 2D numpy array for scikit-learn predict.
    """
    arr = np.array(symptoms, dtype=np.float32).reshape(1, -1)
    return arr


def combine_probs(prob_symptoms: float, prob_image: float, w_symptoms: float = 0.4, w_image: float = 0.6) -> float:
    return float(prob_symptoms * w_symptoms + prob_image * w_image)


# -----------------------
# Pydantic model for JSON endpoint (optional)
# -----------------------
class SymptomsPayload(BaseModel):
    febre_duas_semanas: int
    tosse_com_sangue: int
    escarro_com_sangue: int
    suores_noturnos: int
    dor_no_peito: int
    dor_nas_costas: int
    falta_de_ar: int
    perda_de_peso: int
    cansaco: int
    carocos_axila_pescoco: int
    tosse_catarro_2_4_semanas: int
    linfonodos_inchados: int
    perda_de_apetite: int


# -----------------------
# Endpoints
# -----------------------

@app.get("/")
def home():
    return {"status": "API funcionando!"}


@app.post("/sintomas-e-imagem")
async def sintomas_e_imagem(
    febre_duas_semanas: int = Form(...),
    tosse_com_sangue: int = Form(...),
    escarro_com_sangue: int = Form(...),
    suores_noturnos: int = Form(...),
    dor_no_peito: int = Form(...),
    dor_nas_costas: int = Form(...),
    falta_de_ar: int = Form(...),
    perda_de_peso: int = Form(...),
    cansaco: int = Form(...),
    carocos_axila_pescoco: int = Form(...),
    tosse_catarro_2_4_semanas: int = Form(...),
    linfonodos_inchados: int = Form(...),
    perda_de_apetite: int = Form(...),
    raiox: UploadFile = File(...)
) -> Dict[str, Any]:
    """
    Returns separate outputs for the symptoms model and the image model (no combination).
    """
    # validate models
    if modelo_sintomas is None:
        raise HTTPException(status_code=500, detail="Symptoms model not loaded.")
    if modelo_imagem is None:
        raise HTTPException(status_code=500, detail="Image model not loaded.")

    # build symptoms vector
    symptoms_list = [
        febre_duas_semanas,
        tosse_com_sangue,
        escarro_com_sangue,
        suores_noturnos,
        dor_no_peito,
        dor_nas_costas,
        falta_de_ar,
        perda_de_peso,
        cansaco,
        carocos_axila_pescoco,
        tosse_catarro_2_4_semanas,
        linfonodos_inchados,
        perda_de_apetite
    ]

    X_sym = build_symptoms_array(symptoms_list)
    try:
        # try predict_proba, fallback to predict if not available
        if hasattr(modelo_sintomas, "predict_proba"):
            prob_sym = float(modelo_sintomas.predict_proba(X_sym)[0][1])
        else:
            pred = modelo_sintomas.predict(X_sym)[0]
            prob_sym = float(pred)
    except Exception as e:
        logger.exception("Error predicting symptoms model.")
        raise HTTPException(status_code=500, detail="Error predicting symptoms model.")

    # process image
    content = await raiox.read()
    img_batch = process_image_file(content)
    try:
        prob_img = float(modelo_imagem.predict(img_batch)[0][0])
    except Exception:
        logger.exception("Error predicting image model.")
        raise HTTPException(status_code=500, detail="Error predicting image model.")

    return {
        "probabilidade_sintomas": prob_sym,
        "probabilidade_imagem": prob_img
    }


@app.post("/diagnostico")
async def diagnostico(
    febre_duas_semanas: int = Form(...),
    tosse_com_sangue: int = Form(...),
    escarro_com_sangue: int = Form(...),
    suores_noturnos: int = Form(...),
    dor_no_peito: int = Form(...),
    dor_nas_costas: int = Form(...),
    falta_de_ar: int = Form(...),
    perda_de_peso: int = Form(...),
    cansaco: int = Form(...),
    carocos_axila_pescoco: int = Form(...),
    tosse_catarro_2_4_semanas: int = Form(...),
    linfonodos_inchados: int = Form(...),
    perda_de_apetite: int = Form(...),
    raiox: UploadFile = File(...),
    weight_symptoms: Optional[float] = Form(0.4),
    weight_image: Optional[float] = Form(0.6),
) -> Dict[str, Any]:
    """
    Returns combined diagnosis (probability + explanation).
    """
    # validate models
    if modelo_sintomas is None or modelo_imagem is None:
        raise HTTPException(status_code=500, detail="Models not loaded.")

    symptoms_list = [
        febre_duas_semanas,
        tosse_com_sangue,
        escarro_com_sangue,
        suores_noturnos,
        dor_no_peito,
        dor_nas_costas,
        falta_de_ar,
        perda_de_peso,
        cansaco,
        carocos_axila_pescoco,
        tosse_catarro_2_4_semanas,
        linfonodos_inchados,
        perda_de_apetite
    ]

    X_sym = build_symptoms_array(symptoms_list)
    try:
        if hasattr(modelo_sintomas, "predict_proba"):
            prob_sym = float(modelo_sintomas.predict_proba(X_sym)[0][1])
        else:
            prob_sym = float(modelo_sintomas.predict(X_sym)[0])
    except Exception:
        logger.exception("Error predicting symptoms model.")
        raise HTTPException(status_code=500, detail="Error predicting symptoms model.")

    # image
    content = await raiox.read()
    img_batch = process_image_file(content)
    try:
        prob_img = float(modelo_imagem.predict(img_batch)[0][0])
    except Exception:
        logger.exception("Error predicting image model.")
        raise HTTPException(status_code=500, detail="Error predicting image model.")

    prob_final = combine_probs(prob_sym, prob_img, w_symptoms=weight_symptoms, w_image=weight_image)

    explanation = {
        "why": "Combined weighted average of symptoms model and image model.",
        "weights": {"symptoms": weight_symptoms, "image": weight_image},
        "interpretation": [
            f"Symptoms model probability: {prob_sym:.3f}",
            f"Image model probability: {prob_img:.3f}"
        ]
    }

    return {"probabilidade_final": prob_final, "explanation": explanation}


@app.post("/analisar-tb")
async def analisar_tb(
    febre_duas_semanas: int = Form(...),
    tosse_com_sangue: int = Form(...),
    escarro_com_sangue: int = Form(...),
    suores_noturnos: int = Form(...),
    dor_no_peito: int = Form(...),
    dor_nas_costas: int = Form(...),
    falta_de_ar: int = Form(...),
    perda_de_peso: int = Form(...),
    cansaco: int = Form(...),
    carocos_axila_pescoco: int = Form(...),
    tosse_catarro_2_4_semanas: int = Form(...),
    linfonodos_inchados: int = Form(...),
    perda_de_apetite: int = Form(...),
    raiox: UploadFile = File(...),
    threshold: Optional[float] = Form(0.5)
) -> Dict[str, Any]:
    """
    Quick decision endpoint: returns 'TB' or 'Normal' and the probabilities.
    """
    # validate models
    if modelo_sintomas is None or modelo_imagem is None:
        raise HTTPException(status_code=500, detail="Models not loaded.")

    symptoms_list = [
        febre_duas_semanas,
        tosse_com_sangue,
        escarro_com_sangue,
        suores_noturnos,
        dor_no_peito,
        dor_nas_costas,
        falta_de_ar,
        perda_de_peso,
        cansaco,
        carocos_axila_pescoco,
        tosse_catarro_2_4_semanas,
        linfonodos_inchados,
        perda_de_apetite
    ]

    X_sym = build_symptoms_array(symptoms_list)
    try:
        if hasattr(modelo_sintomas, "predict_proba"):
            prob_sym = float(modelo_sintomas.predict_proba(X_sym)[0][1])
        else:
            prob_sym = float(modelo_sintomas.predict(X_sym)[0])
    except Exception:
        logger.exception("Error predicting symptoms model.")
        raise HTTPException(status_code=500, detail="Error predicting symptoms model.")

    content = await raiox.read()
    img_batch = process_image_file(content)
    try:
        prob_img = float(modelo_imagem.predict(img_batch)[0][0])
    except Exception:
        logger.exception("Error predicting image model.")
        raise HTTPException(status_code=500, detail="Error predicting image model.")

    prob_final = combine_probs(prob_sym, prob_img)
    label = "TB" if prob_final >= threshold else "Normal"

    return {
        "label": label,
        "prob_symptoms": prob_sym,
        "prob_image": prob_img,
        "prob_final": prob_final
    }
