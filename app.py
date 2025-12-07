import numpy as np
import pickle
from flask import Flask, request, render_template

app = Flask(__name__, template_folder="template", static_folder="staticFiles")

model = pickle.load(open('build.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        values = []

        for v in request.form.values():
            if v == "" or v is None:          
                return render_template('index.html',
                                       prediction_text="Please fill all fields!")
            values.append(int(v))

        final_features = [np.array(values)]
        prediction = model.predict(final_features)[0]  

        if prediction == 0:
            result = "Will Cancel"
        else:
            result = "Will Not Cancel"

        return render_template('index.html', prediction_text=result)

    except Exception as e:
        return render_template('index.html',
                               prediction_text=f"ERROR: {str(e)}")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
