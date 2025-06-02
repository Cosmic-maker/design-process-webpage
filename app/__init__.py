import os
import shutil
from flask import Flask

def reset_directories():
    folders_to_reset = [
        os.path.join("app", "uploads"),
        os.path.join("app", "static", "diagrams"),
    ]

    for folder in folders_to_reset:
        if os.path.exists(folder):
            shutil.rmtree(folder)
        os.makedirs(folder, exist_ok=True)

def create_app():
    app = Flask(__name__)
    app.secret_key = os.urandom(24)

    reset_directories()  # Ordner beim Start löschen & neu erstellen

    from .routes import setup_routes
    setup_routes(app)

    return app
