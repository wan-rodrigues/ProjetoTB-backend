import numpy as np
import joblib
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from tensorflow.keras.models import load_model
from PIL import Image
import io

app = FastAPI()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


modelo_imagem = load_model("models/MD_ETB.h5")
modelo_sintomas = joblib.load("models/MD_RF.joblib")



def processar_imagem(file):
    img = Image.open(io.BytesIO(file)).convert("RGB")
    img = img.resize((300, 300))
    img = np.array(img) / 255.0
    img = img.reshape(1, 300, 300, 3)
    return img



@app.post("/analisar")
async def analisar(
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
):

    
    entrada_sintomas = np.array([[
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
    ]])

    
    prob_sintomas = float(modelo_sintomas.predict_proba(entrada_sintomas)[0][1])

    
    conteudo_img = await raiox.read()
    img = processar_imagem(conteudo_img)

    
    prob_imagem = float(modelo_imagem.predict(img)[0][0])

    
    prob_final = (prob_sintomas * 0.4) + (prob_imagem * 0.6)

    return {
        "probabilidade_sintomas": prob_sintomas,
        "probabilidade_imagem": prob_imagem,
        "probabilidade_final": prob_final
    }


@app.get("/")
def home():
    return {"status": "API funcionando!"}