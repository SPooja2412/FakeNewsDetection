from flask import Flask, request, jsonify
import pickle
import numpy as np
import tensorflow as tf

# Load vectorizer and model
with open('tfidf_vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

interpreter = tf.lite.Interpreter(model_path="fake_news_detector_model.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    text = request.json['text']
    sample_vector = vectorizer.transform([text]).toarray()

    # Adjust shape if needed
    if sample_vector.shape[1] < 1000:
        padding = np.zeros((1, 1000 - sample_vector.shape[1]))
        sample_vector = np.hstack((sample_vector, padding))
    elif sample_vector.shape[1] > 1000:
        sample_vector = sample_vector[:, :1000]

    sample_vector = sample_vector.astype('float32')
    interpreter.set_tensor(input_details[0]['index'], sample_vector)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])
    prediction = 'Real' if output[0][0] > 0.5 else 'Fake'
    return jsonify({'prediction': prediction})

if __name__ == '__main__':
    app.run(debug=True)
