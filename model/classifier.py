import pickle 
import pandas as pd 

class Model: 
    def __init__(self) -> None:
        with open("./model/classifier.pkl", "rb") as f: 
            self.model = pickle.load(f)

        with open("./model/scaler.pkl", 'rb') as f: 
            self.scaler = pickle.load(f)

    def scaler_data(self, data: dict): 
        df = pd.DataFrame([data])

        try: 
            data_scaled = self.scaler.transform(df.values)
            return data_scaled 
        
        except: return False  
    
    def predict_loan(self, data: dict): 
        data_scaled = self.scaler_data(data)

        try: 
            predicted = self.model.predict(data_scaled)

            return True if predicted[0] == 1 else False 
        except: 
            return None #self.model.predict(data_scaled)
    
    def __str__(self) -> str:
        return self.model.__class__.__name__