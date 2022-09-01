import tensorflow as tf
from tensorflow.keras.models import load_model
from flask import Flask, render_template, request, jsonify
import numpy as np
import pandas as pd
import librosa as lib
from sklearn import preprocessing
from io import StringIO




MODEL_PATH = "gru-attention_V.0.tf"
model = load_model(MODEL_PATH)

app = Flask(__name__)


df = pd.DataFrame()
predictions = list()
activities = ["Human_Movement", "Vehicle_Movement", "No_Movement", "Animal_Movement"]

def compute_mfcc(audio, rate):
    mfcc_feature = lib.feature.mfcc(y= audio, sr= rate, n_mfcc = 20 ,hop_length = 256 ,n_fft = 1024)
    #n_mfcc gives no. of mfcc coefficients required,hop_length gives no. of samples between successive frames,n_fft gives length of the fft operation
    mfcc_feature = preprocessing.scale(mfcc_feature , axis = 1)
    mfcc_feature = mfcc_feature.T
#     f = plt.figure(1, figsize = (25,10))
#     f.set_figwidth(8)
#     f.set_figheight(4)
#     librosa.display.specshow(mfcc_feature , x_axis = 'time' , sr = (0.025 * rate))
#     plt.colorbar(format='%+2f')
#     plt.title("MFCC")
#     plt.show()
#     plt.pause(1)
#     plt.close()
    #print(mfcc_feature.shape)
    return mfcc_feature



@app.route('/model_predict' , methods = ["GET" , "POST"])
def model_predict():
    
    if not predictions:
        return jsonify(result = "No Movement :)")

    else:
        result = predictions.pop(0)
        return jsonify(result = result)



@app.route('/get_file' , methods = ["GET" , "POST"])
def get_file():
    if request.method == 'POST':
        if request.files:
            f = request.files['file']
            data = f.read()
            data = str(data , 'utf-8')
            data = StringIO(data)
            data = pd.read_csv(data, skiprows=22, header = None).iloc[: , 2]
            data = data.astype(float)
            #print(data)
            mfcc = compute_mfcc(data.to_numpy() , 8000)
            mfcc = mfcc.reshape((1,32,20))
            predict = model.predict(mfcc)
            result = activities[np.argmax(predict[0])]
            print(result)
            predictions.append(result)
            #return jsonify(result = result)
            return "received"

           
    else:
        return 'Error occured :('

@app.route('/' , methods = ["GET" , "POST"])
def index():

    return render_template('home.html')



if __name__ == '__main__':
    app.debug = False
    app.run(host='0.0.0.0' , port = 5000)  

