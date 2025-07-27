# api/route.py
from fastapi import FastAPI
import joblib
import ast
import re
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
)

# Carrega modelos (menos de 50MB)
vectorizer = joblib.load('model/vectorizer.joblib')
nn = joblib.load('model/nn_model.joblib')
metadata = joblib.load('model/metadata.joblib')

@app.post("/suggest")
async def suggest(ingredients: str):
    # Sua lógica Python aqui
    return {"recipes": []}  # Retorno de exemplo