from flask import Flask, render_template, request
import imageio
import scipy
from PIL import Image
import re
import os
import sys
import numpy as np
from keras.models import load_model
import logging

logging.basicConfig(level=logging.DEBUG)

import base64

app = Flask(__name__)
sys.path.append(os.path.abspath("./model"))
# load the model
mnist_model = load_model("model.h5")
mnist_model._make_predict_function()

def image(imageData):
    img = re.search(r'base64,(.*)', str(imageData)).group(1)
    with open("output.png", 'wb') as output:
        output.write(base64.b64decode(img))
    


@app.route('/')
def index():
    return render_template("index.html")

@app.route('/predict/', methods=['GET'])
def predict():
    # get the query from the user
    img = request.args.to_dict()['data']
    # data cleaning on user argument
    x = image(img) 
    x = imageio.imread('output.png')
    x = np.array(Image.fromarray(x).resize((28,28)))
    scipy.misc.imsave("outfile.png", x)
    x = x[:,:,-1]
    x = x.astype(float)
    x = x.reshape([28, 28, 1])
    x /= 255. 

    # get prediction
    prediction = mnist_model.predict(np.array([x]))
    num = np.argmax(prediction)
    
    # just want probability of that single prediction
    prediction = round(prediction[0][num], 3)

    output = str(num) + "\nConfidence: " +str(prediction)
    return str(output)


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)








