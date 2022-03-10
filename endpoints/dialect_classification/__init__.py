from http.client import HTTPResponse
from socket import socket
from flask import request, Response, send_file, jsonify
from flask_restplus import Namespace, Resource, fields
from http import HTTPStatus
from numpy import require
from src.text_inference import dense_nn_inference, classif_ml_inference

namespace = Namespace('classification-model', 'Classification APIs')

classification_model = namespace.model('TextModel', {
    'InputText': fields.String(
        required=True,
        description="Text to be classified"
    ),
    'Algorithm': fields.String(
        required=True,
        description="Algorithm of the dialect classification"
    )
})


@namespace.route('')
class Classification(Resource):
    @namespace.response(500, 'Internal Server error')
    @namespace.expect(classification_model)
    @namespace.marshal_with(classification_model, code=HTTPStatus.CREATED)
    def post(self):
        '''Post method for arabic dialect classification'''
        data = request.json
        input_text = data['InputText']
        algorithm = data['Algorithm']
        if algorithm == 'Dense_NN':
            pred_class = dense_nn_inference(
                input_text, 'src/models/tokenizer.pickle', "src/models/dense_model.h5")
        if algorithm == 'SGD':
            pred_class = classif_ml_inference(
                input_text, "src/models/SGD_model.pkl")
        if algorithm == 'RF':
            pred_class = classif_ml_inference(
                input_text, "src/models/RF_model.pkl")
        print(pred_class)
        return jsonify({'Class': pred_class})
