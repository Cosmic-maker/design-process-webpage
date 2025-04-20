import os

import pandas as pd
from werkzeug.utils import secure_filename
from flask import render_template, request, redirect, flash, send_file
from app.logic import process_excel_file, create_combined_excel, perform_markov_chain_analysis

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

    @app.route("/markov_chain_analysis")
    def generate_markov():
        if not os.path.exists(COMBINED_FILENAME):
            flash("❌ Kombinierte Datei nicht gefunden.")
            return redirect("/")

        try:
            # read excel file
            xls = pd.ExcelFile(COMBINED_FILENAME)

            for sheet_name in xls.sheet_names:
                df = xls.parse(sheet_name)
                perform_markov_chain_analysis(df, sheet_name=sheet_name, filename_base="combined_output")

            flash("📊 Markov-Diagramme erfolgreich erstellt.")
        except Exception as e:
            print(f"❌ Fehler bei der Markov-Analyse: {str(e)}")
            flash(f"❌ Fehler bei der Markov-Analyse: {str(e)}")

        return redirect("/")
