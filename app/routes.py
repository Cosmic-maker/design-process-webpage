from flask import render_template, request, redirect, flash
import os
from werkzeug.utils import secure_filename

from app.logic import process_excel_file

UPLOAD_FOLDER = "uploads"  # Ordner für Uploads
ALLOWED_EXTENSIONS = {'xlsx'}  # Nur Excel-Dateien erlaubt

# Funktion zur Überprüfung der Dateiendung
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def setup_routes(app):
    app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)  # Ordner für Uploads erstellen, falls nicht vorhanden

    @app.route("/")
    def index():
        return render_template("index.html")

    @app.route("/correspondence_analysis")
    def correspondence_analysis():
        return render_template("correspondence_analysis.html")

    @app.route('/cumulative_occurence_analysis')
    def cumulative_occurence_analysis():
        return render_template("cumulative_occurence_analysis.html")

    @app.route("/markov_chain_analysis")
    def markov_chain_analysis():
        return render_template("markov_chain_analysis.html")

    # Route für das Hochladen der Excel-Dateien
    @app.route("/upload", methods=["POST"])
    def upload():
        files = request.files.getlist("files")

        # Überprüfen, ob Dateien vorhanden sind
        if not files or files[0].filename == "":
            flash("Keine Dateien hochgeladen.")
            return redirect("/")

        for file in files:
            if file and allowed_file(file.filename):  # Überprüfen, ob Datei eine erlaubte Excel-Datei ist
                filename = secure_filename(file.filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)  # Speichern der Datei auf dem Server

                try:
                    # Verarbeite die Datei mit der Logik-Funktion und speichere das Diagramm
                    process_excel_file(filepath)
                    flash(f"Die Datei {filename} wurde hochgeladen.")
                except ValueError as e:
                    flash(f"Fehler bei der Verarbeitung der Datei {filename}: {e}")
                except Exception as e:
                    flash(f"Unerwarteter Fehler bei der Verarbeitung der Datei {filename}: {e}")
            else:
                flash(f"{file.filename} ist keine gültige Excel-Datei. Bitte eine .xlsx-Datei hochladen.")

        return redirect("/")  # Nach dem Upload zurück zur Startseite