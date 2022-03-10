from flask import Flask
from endpoints import blueprint as classification_endpoint
from flask_ngrok import run_with_ngrok
from flask_cors import CORS

app = Flask(__name__)
CORS(app)
run_with_ngrok(app)
app.register_blueprint(classification_endpoint)

if __name__ == '__main__':
    app.run()
