from fastapi import FastAPI 
from pydantic import BaseModel 
from typing import List 

from model import Model 

app = FastAPI()

class InputData(BaseModel): 
    params: List[float]

@app.post("/predict")
def predict(data: InputData):
    return {'response': True if Model().predict_loan(data.params) == 1 else 0}
