from http.client import HTTPResponse
from socket import socket
from flask import request, Response, send_file, jsonify
from flask_restplus import Namespace, Resource, fields
from http import HTTPStatus
from numpy import require
# from utilis.inference import inference

namespace = Namespace('classification-model', 'Classification APIs')

classification_model = namespace.model('TextModel', {
    'InputText': fields.String(
        required=True,
        description="Text to be classified"
    )
})


@namespace.route('')
class Classification(Resource):
    @namespace.response(500, 'Internal Server error')
    @namespace.expect(classification_model)
    @namespace.marshal_with(classification_model, code=HTTPStatus.CREATED)
    def post(self):
        '''Post method for segmentation of an given lat/long bounding box'''
        data = request.json
        input_text = data['InputText']
        return jsonify({'status': str("testtt")})
