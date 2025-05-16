import streamlit as st
import numpy as np
import tensorflow as tf
import pickle

# Load TFLite model
interpreter = tf.lite.Interpreter(model_path="fake_news_detector_model.tflite")
interpreter.allocate_tensors()

# Load TF-IDF Vectorizer
with open('tfidf_vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Streamlit UI
st.set_page_config(page_title="Fake News Detector Using TinyML", page_icon="📰", layout="centered")

st.title("📰 Fake News Detector Using TinyML")  # Updated heading

# Input text
news_text = st.text_area("Paste a news headline or article here 👇", height=200)

# Add button with vibrant animations
button_style = """
    <style>
        .stButton>button {
            background-color: #4CAF50;
            color: white;
            padding: 10px 20px;
            font-size: 18px;
            border-radius: 5px;
            border: none;
            cursor: pointer;
            transition: 0.3s ease-in-out;
        }
        .stButton>button:hover {
            background-color: #45a049;
            box-shadow: 0 5px 15px rgba(0, 0, 0, 0.2);
        }
    </style>
"""
st.markdown(button_style, unsafe_allow_html=True)

if st.button("✨ Check News"):
    if news_text.strip() == "":
        st.warning("⚠️ Please enter some text.")
    else:
        # Vectorize and pad
        X = vectorizer.transform([news_text])
        X = X.toarray().astype(np.float32)
        if X.shape[1] < 1000:
            X = np.pad(X, ((0, 0), (0, 1000 - X.shape[1])), mode='constant')
        elif X.shape[1] > 1000:
            X = X[:, :1000]

        # Run inference
        interpreter.set_tensor(input_details[0]['index'], X)
        interpreter.invoke()
        prediction = interpreter.get_tensor(output_details[0]['index'])[0][0]

        # Display results with colorful styles
        result = "🟢 Real News" if prediction > 0.5 else "🔴 Fake News"
        progress_color = "green" if prediction > 0.5 else "red"
        st.markdown(f"""
            <div style="text-align: center; margin-top: 20px; font-size: 1.5rem; font-weight: bold; color: {progress_color};">
                Prediction: {result}
            </div>
        """, unsafe_allow_html=True)
        st.progress(float(prediction if prediction > 0.5 else 1 - prediction))

# Footer with updated text
st.markdown("""
    ---
    <div style="text-align: center; margin-top: 20px; font-size: 0.9rem;">
        Crafted with 💡 innovation and 💻 technology using <span style="color: #0D6EFD;">TinyML</span>, <span style="color: #dc3545;">TFLite</span>, and Streamlit
    </div>
""", unsafe_allow_html=True)

