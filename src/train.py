import os
import json
import joblib
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import r2_score, f1_score
from .preprocessing import load_and_preprocess_data

def train_and_evaluate():
    X_train, X_test, y_reg_train, y_reg_test, y_cls_train, y_cls_test, scaler = load_and_preprocess_data('data/housing_data.csv')
    
    os.makedirs('models', exist_ok=True)
    joblib.dump(scaler, 'models/scaler.pkl')

    # 1. REGRESSION
    lr = LinearRegression().fit(X_train, y_reg_train)
    lr_score = r2_score(y_reg_test, lr.predict(X_test))
    
    rf_reg = RandomForestRegressor(n_estimators=100, random_state=42).fit(X_train, y_reg_train)
    rf_reg_score = r2_score(y_reg_test, rf_reg.predict(X_test))

    if lr_score > rf_reg_score:
        best_regressor = lr
        reg_name = "Linear Regression"
    else:
        best_regressor = rf_reg
        reg_name = "Random Forest Regressor"
        
    joblib.dump(best_regressor, 'models/regression_model.pkl')

    # 2. CLASSIFICATION
    log_reg = LogisticRegression().fit(X_train, y_cls_train)
    log_score = f1_score(y_cls_test, log_reg.predict(X_test))

    rf_cls = RandomForestClassifier(n_estimators=100, random_state=42).fit(X_train, y_cls_train)
    rf_score = f1_score(y_cls_test, rf_cls.predict(X_test))

    if log_score > rf_score:
        best_classifier = log_reg
        cls_name = "Logistic Regression"
    else:
        best_classifier = rf_cls
        cls_name = "Random Forest Classifier"

    joblib.dump(best_classifier, 'models/classification_model.pkl')

    # Save model names metadata
    metadata = {
        "regression_algorithm": reg_name,
        "classification_algorithm": cls_name
    }
    with open('models/metadata.json', 'w') as f:
        json.dump(metadata, f)

    print("\nSaved best models and metadata to 'models/' folder!")

if __name__ == '__main__':
    train_and_evaluate()