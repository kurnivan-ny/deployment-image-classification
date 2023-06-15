from flask import Flask, request, jsonify, make_response #flask fast ai api
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np
import os

varians_dict = {'cats': 0, 'dogs': 1}

# load model
model = tf.keras.models.load_model("model_rps.h5")

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = './image_request/'

# predit model
def predict_image(img_path):
    img = load_img(img_path, target_size=(150,150))

    img_array = img_to_array(img)
    img_array = img_array/255.
    img_array = tf.expand_dims(img_array, 0) # 1,150,150,3

    varians_list = list(varians_dict.keys())
    prediction = model(img_array)
    pred_idx = np.round(model.predict(img_array)[0][0]).astype('int') #0 1
    pred_varian = varians_list[pred_idx]
    return pred_varian

@app.route('/predict', methods=['POST'])
def API():
    # request method using POST
    if request.method == "POST":
        if request.json is None:
            return jsonify({"error" : "no image"})
        
        try:
            filejson = request.get_json()
            imageName = filejson["image"]
            img_path = os.path.join(app.config['UPLOAD_FOLDER'], imageName)
            output_image = predict_image(img_path)
            return make_response(jsonify({"predict_image":output_image}),201)
        except FileNotFoundError as e:
            return make_response(jsonify({"error":str(e)}), 400)
        except Exception as e:
            print(e)
            return make_response(jsonify({"error": str(e)}), 500)
    
    return "OK"

if __name__ == "__main__":
    app.run(debug=True)