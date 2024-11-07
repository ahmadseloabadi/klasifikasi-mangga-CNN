from flask import Flask, render_template, request
from tensorflow import keras
from keras import models
from keras.models import load_model
from tensorflow.keras.preprocessing import image

import numpy as np
import tensorflow_hub as hub

import os

app = Flask(__name__)
model = load_model('P1.hdf5', custom_objects={'KerasLayer': hub.KerasLayer})
disease_names = ['Anwar Ratool','Chaunsa(Black)', 'Chaunsa(Summer Bashist)','Chaunsa (White)','Dosehri', 'Fajri','Langra', 'Sindri']
uploaded_folder="static/images/uploaded"

# function to process image and predict results
def process_predict(image_path, model):
    # read image
    img = image.load_img(image_path, target_size=(224, 224))
    # preprocess image
    img = image.img_to_array(img)
    # now divide image and expand dims
    img = np.expand_dims(img, axis=0) / 255
    # Make prediction
    pred_probs = model.predict(img)
    # Get name from prediction
    pred = disease_names[np.argmax(pred_probs)]
    pred_probs = round(np.max(pred_probs)*100, 2)
    return pred, pred_probs


@app.route('/', methods=['GET', 'POST'])
def home_page():
  if request.method == 'POST':
        # name inside files and in html input should match
        image_file = request.files['file']
        if image_file:
                filename = image_file.filename
                file_path = os.path.join( uploaded_folder, filename)
                image_file.save(file_path)
                
                # prediction
                pred, pred_proba = process_predict(file_path, model)
                if pred_proba > 45:
                  return render_template('prediction.html',gambar=file_path, prediction=pred, prediction_probability=pred_proba)
                else: 
                    return render_template('false_pred.html')  
  return render_template("index.html")


@app.route('/Categories')
def categories_page():
    return render_template('categories.html')

@app.route('/About')
def about_page():
    return render_template("about.html")


if __name__ == '__main__':
    app.run()
