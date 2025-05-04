import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
import numpy
import pickle
import pandas

app = FastAPI()

try:
    with open("model_v_0_1.pkl", "rb") as pickle_in:
        classifier = pickle.load(pickle_in)
except FileNotFoundError:
    classifier = None  # Handle the case where the model file is not found
    print("Error: model_v_0_1.pkl not found!")

class BankNote(BaseModel):
    variance: float
    skewness: float
    curtosis: float
    entropy: float

@app.get('/')
def index():
    return {'message': 'hello, stranger'}

@app.get('/{name}')
def get_name(name: str):
    return {'message': f'Hello, {name}'}

@app.post('/predict')
def predict_banknote(data: BankNote):
    if classifier is None:
        return {"error": "Model not loaded. Please ensure model_v_0_1.pkl exists."}

    data_dict = data.dict()
    print(data_dict)
    print("hello")
    variance = data_dict['variance']
    print(variance)
    skewness = data_dict['skewness']
    print(skewness)
    curtosis = data_dict['curtosis']
    print(curtosis)
    entropy = data_dict['entropy']
    print(entropy)
    prediction = classifier.predict([[variance, skewness, curtosis, entropy]])
    print(f"Prediction: {prediction}")
    print("hello")

    # Assuming your classifier predicts 1 for "Fake Note" and 0 for "Genuine Bank Note"
    if prediction[0] == 1:
        result = "Fake Note"
    else:
        result = "Genuine Bank Note"

    return {
        'prediction': result
    }

if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)