import streamlit as st
import joblib

# Load model and encoders
model = joblib.load('outfit_predictor.pkl')
le_size = joblib.load('le_size.pkl')
le_genre = joblib.load('le_genre.pkl')
le_body = joblib.load('le_body.pkl')
le_outfit = joblib.load('le_outfit.pkl')

st.title("👗 Outfit Recommender")

size = st.selectbox("Select your Size:", le_size.classes_)
genre = st.selectbox("Select your Gender:", le_genre.classes_)
body_type = st.selectbox("Select your Body Type:", le_body.classes_)

if st.button("Get Outfit Suggestion"):
    # Encode inputs
    input_data = [
        le_size.transform([size])[0],
        le_genre.transform([genre])[0],
        le_body.transform([body_type])[0]
    ]

    prediction = model.predict([input_data])
    outfit = le_outfit.inverse_transform(prediction)

    st.success(f"Recommended Outfit: **{outfit[0]}**")
