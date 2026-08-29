import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_and_preprocess_data(data_path):
    df=pd.read_csv(data_path)

    x=df[['square_feet','bedrooms','age_years','location_score']]
    y_reg=df['price']
    y_cls=df['is_high_value']

    x_train,x_test,y_reg_train,y_reg_test,y_cls_train,y_cls_test=train_test_split(
        x,y_reg,y_cls,test_size=0.2,
        random_state=40
    )

    scaler=StandardScaler()
    x_train_scaled=scaler.fit_transform(x_train)
    x_test_scaled=scaler.transform(x_test)

    return(x_train_scaled,x_test_scaled,
           y_reg_train,y_reg_test,
           y_cls_train,y_cls_test,
           scaler)
