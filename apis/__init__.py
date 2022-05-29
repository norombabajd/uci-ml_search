from flask_restx import Api
from .search import api as ns1

api = Api(
    title='Search for dataset',
    version='1.0'
)

api.add_namespace(ns1)