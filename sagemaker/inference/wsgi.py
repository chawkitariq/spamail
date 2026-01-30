"""
WSGI application wrapper for the Flask app.
This is invoked by gunicorn.
"""
import predictor as myapp

# This is the Flask application
app = myapp.app
