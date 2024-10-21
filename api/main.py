from fastapi import FastAPI
import pickle
import pandas as pd

app = FastAPI()


with open('svm-SMOTE.pkl', 'rb') as file:
    svm_model = pickle.load(file)

with open('voting_clf_final.pkl', 'rb') as file:
    voting_clf_model = pickle.load(file)

with open('xgboost-SMOTE.pkl', 'rb') as file:
    xgb_model = pickle.load(file)

with open('kn-SMOTE.pkl', 'rb') as file:
    knn_model = pickle.load(file)

with open('rf_model.pkl', 'rb') as file:
    random_forest_model = pickle.load(file)

with open('dt-SMOTE.pkl', 'rb') as file:
    decision_tree_model = pickle.load(file)


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

    svm_prediction = svm_model.predict(preprocessed_data)
    svm_probability = svm_model.predict_proba(preprocessed_data)

    voting_clf_prediction = voting_clf_model.predict(preprocessed_data)
    voting_clf_probability = voting_clf_model.predict_proba(preprocessed_data)

    dt_prediction = decision_tree_model.predict(preprocessed_data)
    dt_probability = decision_tree_model.predict_proba(preprocessed_data)

    xgb_prediction = xgb_model.predict(preprocessed_data)
    xgb_probability = xgb_model.predict_proba(preprocessed_data)

    knn_prediction = knn_model.predict(preprocessed_data)
    knn_probability = knn_model.predict_proba(preprocessed_data)

    rf_prediction = random_forest_model.predict(preprocessed_data)
    rf_probability = random_forest_model.predict_proba(preprocessed_data)
    
    probabilities = {
        'SVM': float(svm_probability[0][1]),
        'Voting Classifier': float(voting_clf_probability[0][1]),
        'XGBoost': float(xgb_probability[0][1]),
        'Decision Tree': float(dt_probability[0][1]),
        'Random Forest': float(rf_probability[0][1]),
        'K-nearest Neighbors': float(knn_probability[0][1])
    }

    return probabilities

@app.post("/predict")
async def predict(data: dict):
    probabilities = get_prediction(data)
    return {"probabilities": probabilities}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=10000)
