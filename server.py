import os
import numpy as np
from keras.models import load_model
from flask import Flask, request, jsonify
from io import BytesIO
from tensorflow.keras.preprocessing import image

class_labels = [
    'Apple___Apple_scab','Apple___Black_rot','Apple___Cedar_apple_rust','Apple___healthy','Blueberry___healthy',
    'Cherry_(including_sour)___Powdery_mildew','Cherry_(including_sour)___healthy','Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
    'Corn_(maize)___Common_rust_','Corn_(maize)___Northern_Leaf_Blight','Corn_(maize)___healthy','Grape___Black_rot',
    'Grape___Esca_(Black_Measles)','Grape___Leaf_blight_(Isariopsis_Leaf_Spot)','Grape___healthy','Orange___Haunglongbing_(Citrus_greening)',
    'Peach___Bacterial_spot','Peach___healthy','Pepper,_bell___Bacterial_spot','Pepper,_bell___healthy','Potato___Early_blight',
    'Potato___Late_blight','Potato___healthy','Raspberry___healthy','Soybean___healthy','Squash___Powdery_mildew',
    'Strawberry___Leaf_scorch','Strawberry___healthy','Tomato___Bacterial_spot','Tomato___Early_blight',
    'Tomato___Late_blight','Tomato___Leaf_Mold','Tomato___Septoria_leaf_spot','Tomato___Spider_mites Two-spotted_spider_mite',
    'Tomato___Target_Spot','Tomato___Tomato_Yellow_Leaf_Curl_Virus','Tomato___Tomato_mosaic_virus','Tomato___healthy'
]
#load model
mdl = load_model(r'plant_disease_model.h5')
# Initialize Flask application
app = Flask(__name__)

# Endpoint for image classification
@app.route('/classify', methods=['POST'])
def classify_image():
    # Get image file from the request
    file = request.files['image']
    # Convert FileStorage object to file-like object
    img_stream = BytesIO(file.read())
    # Load image
    img = image.load_img(img_stream, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    # Scale the data
    img_array = img_array/255
    # Predict
    y_pred = mdl.predict(img_array)
    # Get predicted class label
    predicted_class = class_labels[np.argmax(y_pred)]
    # Return the result
    return jsonify({'class': predicted_class})

if __name__ == '__main__':
    app.run(debug=True)
 
