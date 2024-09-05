# app.py
from flask import Flask, render_template, request
from model import predict_iris

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Lấy dữ liệu từ form
        sepal_length = float(request.form['sepal_length'])
        sepal_width = float(request.form['sepal_width'])
        petal_length = float(request.form['petal_length'])
        petal_width = float(request.form['petal_width'])

        # Dự đoán
        features = [sepal_length, sepal_width, petal_length, petal_width]
        prediction = predict_iris(features)
        
        
        species = ['Setosa', 'Versicolor', 'Virginica']
        result = species[prediction]

        return render_template('result.html', prediction=result)
    except Exception as e:
        return str(e)

if __name__ == "__main__":
    app.run(debug=True)
