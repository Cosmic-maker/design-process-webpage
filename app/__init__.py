import os
from flask import Flask

def create_app():
    app = Flask(__name__)
    app.secret_key=os.urandom(24) # flask needs that for sessions and cookies
    from .routes import setup_routes
    setup_routes(app)

    return app
