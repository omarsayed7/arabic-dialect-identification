from flask import Flask, render_template
from endpoints import blueprint as classification_endpoint
from flask_ngrok import run_with_ngrok
from flask_cors import CORS

app = Flask(__name__)
CORS(app)
app.register_blueprint(classification_endpoint)


@app.route('/')
def home():
    return render_template('home.html')


if __name__ == '__main__':
    app.run()
