from flask import Blueprint
from flask_restplus import Api
from endpoints.dialect_classification import namespace as classification_ns

blueprint = Blueprint('api', __name__, url_prefix='/api')
api_extension = Api(blueprint, title='Arabic Dialect Classification',
                    version='1.0',
                    description='Arab dialect classification web API using Python Flask',
                    doc='/Swagger')

api_extension.add_namespace(classification_ns)
