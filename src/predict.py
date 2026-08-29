import joblib
import json
import os
import pandas as pd

def predict_house(sqft, bedrooms, age, location_score):
    # Load models dynamically inside function
    scaler = joblib.load('models/scaler.pkl')
    reg_model = joblib.load('models/regression_model.pkl')
    cls_model = joblib.load('models/classification_model.pkl')
    
    # Safely load metadata if it exists
    reg_algo = "Regression Model"
    cls_algo = "Classification Model"
    if os.path.exists('models/metadata.json'):
        with open('models/metadata.json', 'r') as f:
            metadata = json.load(f)
            reg_algo = metadata.get('regression_algorithm', reg_algo)
            cls_algo = metadata.get('classification_algorithm', cls_algo)
            
    input_df = pd.DataFrame([[sqft, bedrooms, age, location_score]], 
                            columns=['square_feet', 'bedrooms', 'age_years', 'location_score'])
    
    input_scaled = scaler.transform(input_df)
    
    predicted_price = reg_model.predict(input_scaled)[0]
    is_high_val = cls_model.predict(input_scaled)[0]
    
    label = "High Value Property" if is_high_val == 1 else "Standard Value Property"
    
    return round(predicted_price), label, reg_algo, cls_algo