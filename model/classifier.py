import pickle 

class Model: 
    def __init__(self) -> None:
        with open("./model/classifier.pkl", "rb") as f: 
            self.model = pickle.load(f)

        with open("./model/scaler.pkl", 'rb') as f: 
            self.scaler = pickle.load(f)

    def scaler_data(self, data: list): 
        return self.scaler.transform([data])
    
    def predict_loan(self, data: list): 
        data_scaled = self.scaler_data(data)

        return self.model.predict(data_scaled)
    
    def __str__(self) -> str:
        return self.model.__class__.__name__