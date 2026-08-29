import pandas as pd
import numpy as np
import os

np.random.seed(42)
n_samples = 500

data = pd.DataFrame({
    'square_feet': np.random.randint(800, 3500, n_samples),
    'bedrooms': np.random.randint(1, 6, n_samples),
    'age_years': np.random.randint(1, 40, n_samples),
    'location_score': np.random.uniform(1.0, 10.0, n_samples)
})


data['price'] = (
    data['square_feet'] * 3500 
    + data['bedrooms'] * 300000 
    - data['age_years'] * 30000 
    + data['location_score'] * 500000 
    + np.random.normal(0, 200000, n_samples)
)


data['is_high_value'] = (data['price'] > data['price'].median()).astype(int)


os.makedirs('data', exist_ok=True)
data.to_csv('data/housing_data.csv', index=False)
print("Updated dataset created with higher pricing scale at data/housing_data.csv!")