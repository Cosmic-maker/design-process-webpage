import os

import pandas as pd
from werkzeug.utils import secure_filename
from flask import render_template, request, redirect, flash, send_file, current_app
from app.logic import process_excel_file, create_combined_excel, perform_cumulative_occurence_analysis, perform_markov_chain_analysis

BASE_DIR = os.path.abspath(os.path.dirname(__file__))

# Konfiguration
ALLOWED_EXTENSIONS = {'xlsx'}
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")  # ABSOLUTER Pfad
COMBINED_FILENAME = os.path.join(UPLOAD_FOLDER, "combined_output.xlsx")  # ABSOLUTER Pfad

# Sicherstellen, dass das Verzeichnis existiert
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def setup_routes(app):
    app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

    @app.route("/")
    def index():
        # Prüfen, ob die kombinierte Datei existiert
        combined_file_exists = os.path.exists(COMBINED_FILENAME)
        return render_template("index.html", combined_file_exists=combined_file_exists)

    @app.route("/upload", methods=["POST"])
    def upload():
        files = request.files.getlist("files")
        if not files or files[0].filename == "":
            flash("❌ Keine Dateien hochgeladen.")
            return redirect("/")

        valid_sheets = {}  # Hier speichern wir die verarbeiteten Sheets

        for file in files:
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)
                print(f"📥 Datei gespeichert unter: {filepath}")

                try:
                    sheet_dfs = process_excel_file(filepath)

                    for name, df in sheet_dfs:
                        original_name = name
                        i = 1
                        while name in valid_sheets:
                            name = f"{original_name}_{i}"
                            i += 1
                        valid_sheets[name] = df

                    flash(f"✅ Datei {filename} wurde erfolgreich verarbeitet.")
                except ValueError as e:
                    flash(f"❌ Fehler bei {filename}: {e}")
                except Exception as e:
                    flash(f"❌ Unerwarteter Fehler bei {filename}: {e}")
            else:
                flash(f"❌ {file.filename} ist keine gültige Excel-Datei.")

        if valid_sheets:
            print(f"📁 Erstelle kombinierte Datei: {COMBINED_FILENAME}")
            create_combined_excel(valid_sheets, COMBINED_FILENAME)
            flash("📄 Kombinierte Excel-Datei wurde erstellt: combined_output.xlsx")

        return redirect("/")

    @app.route("/download_combined_file")
    def download_combined():
        if not os.path.exists(COMBINED_FILENAME):
            print("❌ Datei existiert nicht!")
            flash("❌ Kombinierte Datei nicht gefunden.")
            return redirect("/")

        try:
            return send_file(
                COMBINED_FILENAME,
                as_attachment=True,
                download_name=f"kombinierte_designprozesse.xlsx",
                mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
            )
        except Exception as e:
            print(f"❌ Download-Fehler: {str(e)}")
            flash(f"❌ Fehler beim Download: {str(e)}")
            return redirect("/")

    @app.route('/cumulative_occurence_analysis', methods=["GET", "POST"])
    def cumulative_occurence_analysis():
        uploads_dir = app.config['UPLOAD_FOLDER']

        # Excel-Dateien auflisten
        if os.path.exists(uploads_dir):
            registers = [f for f in os.listdir(uploads_dir) if f.endswith('.xlsx')]
            print(f"Gefundene Register für Cumulative Analysis: {registers}")
        else:
            registers = []
            print(f"❌ Das Verzeichnis {uploads_dir} existiert nicht.")

        selected_register = None
        image_filenames = []
        fbs_results = {}
        characterizations = {}

        # Analyse ausführen, falls Button gedrückt wurde
        if request.method == "POST":
            selected_register = request.form.get("register")
            fbs_threshold = int(request.form.get("threshold", 5))  # Schwelle aus dem Formular
            min_occurrences_char = int(request.form.get("min_occurrences_char", 5))  # Mindestanzahl für Charakterisierung
            min_occurrences_slope = int(request.form.get("min_occurrences_slope", 5))  # Mindestanzahl für Steigung

            if selected_register:
                selected_file = os.path.join(uploads_dir, selected_register)
                try:
                    df = pd.read_excel(selected_file)
                    filename_base = os.path.splitext(selected_register)[0]

                    # ACHTUNG: sheet_name muss gesetzt sein oder als Parameter kommen
                    results = perform_cumulative_occurence_analysis(
                        df,
                        sheet_name=None,  # ← falls du keinen bestimmten Sheet brauchst
                        filename_base=filename_base,
                        min_occurrences_char=min_occurrences_char,
                        min_occurrences_slope=min_occurrences_slope,
                        fbs_threshold=fbs_threshold
                    )

                    fbs_results = results["fbs_results"]
                    characterizations = results["characterizations"]

                    flash(f"✅ Cumulative Occurrence Analysis für {selected_register} erfolgreich durchgeführt.")

                    diagram_folder = os.path.join(app.static_folder, 'diagrams', 'cumulative_occurrence_analysis')
                    image_filenames = [
                        f for f in os.listdir(diagram_folder)
                        if f.startswith(filename_base) and f.endswith('.png')
                    ]
                except Exception as e:
                    flash(f"❌ Fehler bei der Analyse der Datei: {str(e)}")
            else:
                flash("❌ Kein Register ausgewählt.")

        # Sowohl für GET als auch POST wird gerendert
        return render_template(
            "cumulative_occurence_analysis.html",
            selected_register=selected_register,
            registers=registers,
            image_filenames=image_filenames,
            fbs_results=fbs_results,
            characterizations=characterizations
        )


    @app.route('/markov_analysis', methods=["GET", "POST"])
    def markov_analysis():
        uploads_dir = os.path.join('app', 'uploads')
        selected_register = request.form.get('register')

        if os.path.exists(uploads_dir):
            registers = [f for f in os.listdir(uploads_dir) if f.endswith('.xlsx')]
            print(f"Gefundene Register: {registers}")
        else:
            registers = []
            print(f"Das Verzeichnis {uploads_dir} existiert nicht.")

        if request.method == "POST":
            selected_register = request.form.get("register")
            if selected_register:
                selected_file = os.path.join(uploads_dir, selected_register)
                try:
                    df = pd.read_excel(selected_file)
                    perform_markov_chain_analysis(df, selected_register, selected_register)
                    flash(f"Markov-Analyse für {selected_register} erfolgreich durchgeführt.")
                except Exception as e:
                    flash(f"Fehler bei der Analyse der Datei: {str(e)}")
            else:
                flash("Kein Register ausgewählt.")

        uploads_dir = os.path.join(app.root_path, 'uploads')
        registers = [f for f in os.listdir(uploads_dir) if f.endswith('.xlsx')]

        return render_template("markov_chain_analysis.html",
                       selected_register=selected_register,
                       registers=registers)