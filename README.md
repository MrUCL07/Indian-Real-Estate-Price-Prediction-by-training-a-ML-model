----Indian Real Estate Price Prediction----

1.(src/create_data.py)
Firstly, for creating the data I am using numpy's random function to generate few random values for the square feet, bedrooms, age of the house and the rating of the location where the house is located. For predicting the price of the house using all this information, I am using the formulae Price=(sqftx3500)+(bedroomsx300,000)-(agex30,000)+(locationx500,000). Then as we get the price of every house, I am using a function to define it, if its price is higher than the median market price then it's stored as 1 or otherwise it's stored as 0. All of the data is stored in data/ folder named as housing_data.csv .

2.(src/preprocessing.py)
Now as we have the data, I am going to train the ML Model using train_test_split() and StandardSclar(). train_test_split() function splits the 500 records into 80% training data and 20% testing data. StandardScaler() function standardizes the feature to have a mean of 0 and standard deviation of 1. Then we use scaler.fit_transform() and scaler.transform() .

3.(src/train.py)
In this we train the ML model using LinearRegression() and RandomForestRegressor() for predicting the price. r2_score evaluates the accuracy of the prediction. I also use LogisticRegression() and RandomForestClassifier() for categorize properties. f1_score evaluates the classification balance. 

4.(src/predict.py)
In this codepage, I have written the code for the interference engine to accept the user inputs from the UI. I am using joblib.load(), pd.DataFrame(), scaler.transform() and .predict(). joblib.load() imports the trained models into memory instantly. pd.DataFrame() reads the data and returns it in a 1-row DataFrame. scaler.transform() scales the raw input values using the saved training standard deviation and mean parameters. .predict() returns the price prediction in Rupees.

5.(app.py)
This is the UI of the project. I am using Streamlit which provides web GUI interface. I am using st.number_input(), st.slider(), st.button(), format_indian_currency() and st.metric(). st_number_input and st.slider() renders the input form for user parameters. st.button() triggers the predict_house execution on user interface. format_indian_currency converts the raw large data into Lakhs and Crores. st.metric() renders the high contrast metric visual cards showing prices and categories directly on the screen.