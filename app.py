import streamlit as st
from src.predict import predict_house

st.set_page_config(page_title="Indian Real Estate Evaluator", page_icon="🏡", layout="centered")

st.title("🏡 Indian Real Estate Price & Value Evaluator")
st.write("Enter property details below to generate predictions from both Regression and Classification models.")

st.markdown("---")

col1, col2 = st.columns(2)

with col1:
    sqft = st.number_input("Square Feet", min_value=500, max_value=10000, value=1800, step=50)
    bedrooms = st.number_input("Bedrooms", min_value=1, max_value=10, value=2, step=1)

with col2:
    age = st.number_input("Property Age (Years)", min_value=0, max_value=100, value=1, step=1)
    location_score = st.slider("Location Rating (1 - 10)", min_value=1.0, max_value=10.0, value=9.0, step=0.1)

st.markdown("---")

def format_indian_currency(amount):
    if amount >= 10000000:
        return f"₹{amount / 10000000:.2f} Crores"
    elif amount >= 100000:
        return f"₹{amount / 100000:.2f} Lakhs"
    else:
        return f"₹{amount:,.0f}"

if st.button("Evaluate Property", use_container_width=True):
    price, category, reg_algo, cls_algo = predict_house(sqft, bedrooms, age, location_score)
    
    st.subheader("Model Predictions")
    
    res_col1, res_col2 = st.columns(2)
    with res_col1:
        st.metric(label="Predicted Price", value=format_indian_currency(price))
        st.caption(f" Algorithm used: **{reg_algo}**")
        
    with res_col2:
        st.metric(label="Investment Tag", value=category)
        st.caption(f" Algorithm used: **{cls_algo}**")