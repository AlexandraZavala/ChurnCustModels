from fastapi import FastAPI
import pickle
import pandas as pd

app = FastAPI()

with open('xgb_model.pkl', 'rb') as file:
    loaded_model = pickle.load(file)

def preprocess_data(data):
    input_dict = {
        'Credit_Score': data['CreditScore'],
        'Age': data['Age'],
        'Tenure': data['Tenure'],
        'Balance': data['Balance'],
        'NumOfProducts': data['NumOfProducts'],
        'HasCrCard': data['HasCrCard'],
        'IsActiveMember': data['IsActiveMember'],
        'EstimatedSalary': data['EstimatedSalary'],
        'Geography_France': 1 if data['Geography'] == 'France' else 0,
        'Geography_Germany': 1 if data['Geography'] == 'Germany' else 0,
        'Geography_Spain': 1 if data['Geography'] == 'Spain' else 0,
        'Gender_Male': 1 if data['Gender'] == 'Male' else 0,
        'Gender_Female': 1 if data['Gender'] == 'Female' else 0
    }

    customer_df = pd.DataFrame([input_dict])
    print("Customer DataFrame:")
    print(customer_df)

    return pd.DataFrame([input_dict])

def get_prediction(data):
    preprocessed_data = preprocess_data(data)
    prediction = loaded_model.predict(preprocessed_data)
    probability = loaded_model.predict_proba(preprocessed_data)
    return prediction, probability

@app.post("/predict")
async def predict(data: dict):
    prediction, probability = get_prediction(data)
    return {"prediction": prediction.tolist(), "probabilities": probability.tolist()}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=10000)
