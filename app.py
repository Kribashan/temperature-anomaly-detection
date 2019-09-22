from flask import Flask, request, jsonify
import joblib
import traceback
import pandas as pd
import os

#creating instance of the class
app=Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    if loaded_model:
        try:
            json_ = request.json
            query = pd.get_dummies(pd.DataFrame(json_))

            prediction = list(loaded_model.predict(query))

            return jsonify({'prediction': str(prediction)})

        except:

            return jsonify({'trace': traceback.format_exc()})
    else:
        print ('Train the model first')
        return ('No model here to use')


if __name__ == '__main__':
   cur_dir = os.path.dirname(__file__)

   with open(os.path.join(cur_dir,'model','model.pkl'), 'rb') as model_file:
       loaded_model = joblib.load(model_file) # Load "model.pkl"

   app.run(port=8051, debug=True)