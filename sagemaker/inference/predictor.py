"""
Flask application that implements the /ping and /invocations endpoints
required by SageMaker for spam classification.
"""
import os
import json
import flask
import joblib
import logging
from marshmallow import Schema, fields, ValidationError

app = flask.Flask(__name__)
model = None
vectorizer = None
model_path = '/opt/ml/model/model.joblib'
vectorizer_path = '/opt/ml/model/vectorizer.joblib'

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class InstanceSchema(Schema):
    text = fields.Str(required=True)


class InputSchema(Schema):
    instances = fields.List(fields.Nested(InstanceSchema), required=True)


def load_model():
    """Load the model and vectorizer from the model directory"""
    global model, vectorizer
    
    if os.path.exists(model_path) and os.path.exists(vectorizer_path):
        model = joblib.load(model_path)
        vectorizer = joblib.load(vectorizer_path)
        logger.info(f"Model and vectorizer loaded successfully")
        return True
    else:
        logger.error(f"Model files not found")
        return False


if not load_model():
    raise RuntimeError("Model files missing. Cannot start inference service.")


@app.route('/ping', methods=['GET'])
def ping():
    """
    Health check endpoint - required by SageMaker.
    Returns 200 if model is loaded and ready.
    """
    health = model is not None and vectorizer is not None
    status = 200 if health else 500
    return flask.Response(response='\n', status=status, mimetype='application/json')


@app.route('/invocations', methods=['POST'])
def invocations():
    """
    Inference endpoint - required by SageMaker.
    Expects: {"instances": [{"text": "email content"}, ...]}
    Returns: {"predictions": [0, 1, ...], "labels": ["ham", "spam", ...]}
    """
    if model is None or vectorizer is None:
        return flask.Response(
            response=json.dumps({'error': 'Model not loaded'}),
            status=500,
            mimetype='application/json'
        )
    
    try:
        if flask.request.content_type != 'application/json':
            return flask.Response(
                response=json.dumps({'error': 'Unsupported content type'}),
                status=415,
                mimetype='application/json'
            )
        
        input_json = flask.request.get_json()
        if not input_json:
            return flask.Response(
                response=json.dumps({'error': 'Empty request body'}),
                status=400,
                mimetype='application/json'
            )
        
        schema = InputSchema()
        try:
            validated = schema.load(input_json)
        except ValidationError as err:
            logger.warning(f"Validation error: {err.messages}")
            return flask.Response(
                response=json.dumps({'error': 'Invalid input format', 'details': err.messages}),
                status=400,
                mimetype='application/json'
            )
        
        # Extract texts from instances
        texts = [instance['text'] for instance in validated['instances']]
        
        # Vectorize and predict
        X = vectorizer.transform(texts)
        predictions = model.predict(X).tolist()
        labels = ['ham' if p == 0 else 'spam' for p in predictions]
        
        result = {
            'predictions': predictions,
            'labels': labels
        }
        
        # Add probabilities if available
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(X).tolist()
            result['probabilities'] = probabilities
        
        return flask.Response(
            response=json.dumps(result),
            status=200,
            mimetype='application/json'
        )
        
    except Exception as e:
        logger.error("Error during inference", exc_info=True)
        return flask.Response(
            response=json.dumps({'error': 'Internal server error'}),
            status=500,
            mimetype='application/json'
        )
