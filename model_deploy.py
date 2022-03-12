from flask import Flask, render_template, make_response, request
from flask_cors import CORS
from src.text_inference_deploy import select_model
app = Flask(__name__)
CORS(app)


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/predict', methods=['POST'])
def predict():
    '''Post method for arabic dialect classification'''
    data = request.form
    input_text = data['InputText']
    algorithm = data['Algorithm']
    pred_class = select_model(algorithm, input_text)
    print(pred_class)
    headers = {'Content-Type': 'text/html'}
    return make_response(render_template('result.html', result=pred_class), 200, headers)


if __name__ == '__main__':
    app.run()
