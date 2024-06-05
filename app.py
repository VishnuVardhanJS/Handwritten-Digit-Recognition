import numpy as np
import tensorflow as tf
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
import os
import io

# Define a flask app
app = Flask(__name__)

# Model saved with Keras model.save()
MODEL_PATH = './model/handwritten_digits.keras'


print('Model loaded. Check http://127.0.0.1:5000/')

@app.route('/', methods=['GET'])
def index():
    
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        f = request.files['file']
        image_data = f.read()
        inp_image = Image.open(io.BytesIO(image_data))
        model = tf.keras.models.load_model(MODEL_PATH)
        # print(type(inp_image))
        cv2_img = np.array(inp_image)
        image = cv2.cvtColor(cv2_img, cv2.COLOR_RGB2BGR)
        grey = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2GRAY) #Image to Black and White
        ret, thresh = cv2.threshold(grey.copy(), 75, 255, cv2.THRESH_BINARY_INV) #Adding Threshold
        contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) #Finding Bounding Contours

        #Cropping Image
        for c in contours:
            x,y,w,h = cv2.boundingRect(c)
            digit = thresh[y:y+h, x:x+w]
            resized_digit = cv2.resize(digit, (18,18))
            padded_digit = np.pad(resized_digit, ((5,5),(5,5)), "constant", constant_values=0)
        reshaped_image = padded_digit.reshape(1, 28, 28, 1)
        prediction = model.predict(reshaped_image)  
        # print("The given Number is : ", np.argmax(prediction))
    return str(np.argmax(prediction))        # Convert to string
    # return None


if __name__ == '__main__':
    app.run(debug=True)

