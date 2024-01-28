from fastapi import FastAPI 
from pydantic import BaseModel 
import pandas as pd 
from typing import List 

from model import Model 

app = FastAPI()

class InputData(BaseModel): 
    dependentes: int 
    educacao: int 
    empregado: int 
    renda_anual: float 
    valor: float 
    pagamento: int 
    score: int 
    patrimonio_residencial: float 
    patrimonio_comercial: float 
    patrimonio_luxo: float 
    patrimonio_bancario: float 

@app.get("/")
def home(): 
    return {"FastAPI": "Hello"}


@app.post("/predict")
async def predict(data: InputData):
    status = Model().predict_loan(data=data.dict())

    return {'response': status}
