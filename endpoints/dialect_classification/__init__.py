from flask import request
from flask_restplus import Namespace, Resource, fields
from http import HTTPStatus
from flask import Flask, render_template, make_response
from src.text_inference_deploy import select_model

namespace = Namespace('classification_model', 'Classification APIs')

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
    @namespace.expect(classification_model)
    @namespace.marshal_with(classification_model, code=HTTPStatus.CREATED)
    def post(self):
        '''Post method for arabic dialect classification'''
        data = request.form
        input_text = data['InputText']
        algorithm = data['Algorithm']
        pred_class = select_model(algorithm, input_text)
        print(pred_class)
        headers = {'Content-Type': 'text/html'}
        return make_response(render_template('result.html', result=pred_class), 200, headers)
