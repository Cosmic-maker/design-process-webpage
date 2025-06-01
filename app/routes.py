import os

import pandas as pd
from werkzeug.utils import secure_filename
from flask import render_template, request, redirect, flash, send_file, current_app
from app.logic import process_excel_file, create_combined_excel, perform_cumulative_occurence_analysis, \
    perform_markov_chain_analysis, perform_correspondence_analysis

BASE_DIR = os.path.abspath(os.path.dirname(__file__))

# Konfiguration
ALLOWED_EXTENSIONS = {'xlsx'}
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")  # ABSOLUTER Pfad
COMBINED_FILENAME = os.path.join(UPLOAD_FOLDER, "combined_output.xlsx")  # ABSOLUTER Pfad

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def setup_routes(app):
    app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

    @app.route("/")
    def index():

        combined_file_exists = os.path.exists(COMBINED_FILENAME)
        return render_template("index.html", combined_file_exists=combined_file_exists)

    @app.route("/upload", methods=["POST"])
    def upload():
        files = request.files.getlist("files")
        if not files or files[0].filename == "":
            flash("❌ No files not uploaded.")
            return redirect("/")

        valid_sheets = {}

        for file in files:
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)
                print(f"📥 File saved as: {filepath}")

                try:
                    sheet_dfs = process_excel_file(filepath)

                    for name, df in sheet_dfs:
                        original_name = name
                        i = 1
                        while name in valid_sheets:
                            name = f"{original_name}_{i}"
                            i += 1
                        valid_sheets[name] = df

                    flash(f"✅ File {filename} was successfully processed.")
                except ValueError as e:
                    flash(f"❌ Error {filename}: {e}")
                except Exception as e:
                    flash(f"❌ Unexpected Error {filename}: {e}")
            else:
                flash(f"❌ {file.filename} is not a valid xlsx file.")

        if valid_sheets:
            print(f"📁 Generate combined file: {COMBINED_FILENAME}")
            create_combined_excel(valid_sheets, COMBINED_FILENAME)
            flash("📄 Combined file is created: combined_output.xlsx")

        return redirect("/")

    @app.route("/download_combined_file")
    def download_combined():
        if not os.path.exists(COMBINED_FILENAME):
            print("❌ File does not exist!")
            flash("❌ Combined file not found.")
            return redirect("/")

        try:
            return send_file(
                COMBINED_FILENAME,
                as_attachment=True,
                download_name=f"kombinierte_designprozesse.xlsx",
                mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
            )
        except Exception as e:
            print(f"❌ Download-Error: {str(e)}")
            flash(f"❌ Download-Error: {str(e)}")
            return redirect("/")


    @app.route('/cumulative_occurence_analysis', methods=["GET", "POST"])
    def cumulative_occurence_analysis():
        uploads_dir = app.config['UPLOAD_FOLDER']
        combined_file = os.path.join(uploads_dir, 'combined_output.xlsx')

        selected_registers = []
        image_filenames = []
        fbs_results = {}
        characterizations = {}

        if os.path.exists(combined_file):
            try:
                xls = pd.ExcelFile(combined_file)
                registers = xls.sheet_names
                print(f"Register in combined_output.xlsx gefunden: {registers}")
            except Exception as e:
                flash(f"❌ Error reading merged file: {str(e)}")
                registers = []
        else:
            flash("❌ File missing: 'combined_output.xlsx'")
            registers = []

        if request.method == "POST":
            selected_registers = request.form.getlist("register")
            fbs_threshold = int(request.form.get("threshold", 5))
            min_occurrences_char = int(request.form.get("min_occurrences_char", 5))
            min_occurrences_slope = int(request.form.get("min_occurrences_slope", 5))

            if selected_registers:
                for selected_register in selected_registers:
                    try:
                        df = pd.read_excel(combined_file, sheet_name=selected_register)
                        filename_base = selected_register

                        results = perform_cumulative_occurence_analysis(
                            df,
                            sheet_name=selected_register,
                            filename_base=filename_base,
                            min_occurrences_char=min_occurrences_char,
                            min_occurrences_slope=min_occurrences_slope,
                            fbs_threshold=fbs_threshold
                        )

                        # Ergebnisse zusammenführen
                        fbs_results.update(results["fbs_results"])
                        characterizations.update(results["characterizations"])

                        flash(f"✅ Analysis for '{selected_register}' successfully carried out.")

                        diagram_folder = os.path.join(app.static_folder, 'diagrams', 'cumulative_occurrence_analysis')
                        found_images = [
                            f for f in os.listdir(diagram_folder)
                            if f.startswith(filename_base) and f.endswith('.png')
                        ]
                        image_filenames.extend(found_images)

                    except Exception as e:
                        flash(f"❌ Error '{selected_register}': {str(e)}")
            else:
                flash("❌ No register selected.")

        return render_template(
            "cumulative_occurence_analysis.html",
            selected_register=selected_registers,
            registers=registers,
            image_filenames=image_filenames,
            fbs_results=fbs_results,
            characterizations=characterizations
        )

    @app.route('/correspondence_analysis', methods=["GET", "POST"])
    def correspondence_analysis():
        combined_file = os.path.join(current_app.root_path, 'uploads', 'combined_output.xlsx')

        selected_registers = []
        image_filename = None
        registers = []
        analysis_type = 'process_phases'  # Default

        # Prüfe, ob Datei existiert
        if not os.path.exists(combined_file):
            flash("❌ Combined file 'combined_output.xlsx' not found.", 'error')
            return render_template("correspondence_analysis.html",
                                   registers=[],
                                   selected_registers=[],
                                   image_filename=None,
                                   analysis_type=analysis_type)

        # Lese Register (Blätter)
        try:
            xls = pd.ExcelFile(combined_file)
            registers = xls.sheet_names
        except Exception as e:
            flash(f"❌ Error reading Excel file: {str(e)}", 'error')
            return render_template("correspondence_analysis.html",
                                   registers=[],
                                   selected_registers=[],
                                   image_filename=None,
                                   analysis_type=analysis_type)

        # Formularabsendung
        if request.method == "POST":
            selected_registers = request.form.getlist("register")
            analysis_type = request.form.get("analysis_type", "process_phases")

            if not selected_registers:
                flash("❌ Please select at least one register.", 'error')
            else:
                try:
                    if analysis_type == 'whole_process' and len(selected_registers) < 2:
                        flash("❌ Whole process analysis requires at least 2 processes.", 'error')
                    else:
                        image_filename = perform_correspondence_analysis(
                            combined_file,
                            selected_registers,
                            analysis_type
                        )
                        flash(f"✅ {analysis_type.replace('_', ' ').title()} completed successfully for: {', '.join(selected_registers)}", 'success')
                except ValueError as e:
                    flash(f"❌ {str(e)}", 'error')
                except Exception as e:
                    flash(f"❌ Analysis error: {str(e)}", 'error')

        return render_template("correspondence_analysis.html",
                               selected_registers=selected_registers,
                               registers=registers,
                               image_filename=image_filename,
                               analysis_type=analysis_type)


    @app.route('/markov_analysis', methods=["GET", "POST"])
    def markov_analysis():
        uploads_dir = os.path.join(app.root_path, 'uploads')
        combined_file = os.path.join(uploads_dir, 'combined_output.xlsx')

        selected_registers = []
        threshold = 0.0

        if request.method == "POST":
            selected_registers = request.form.getlist("register")
            threshold = float(request.form.get("threshold", 0))
            action = request.form.get("action")

            if not selected_registers:
                flash("No register selected.")
            else:
                try:
                    xls = pd.ExcelFile(combined_file)
                    if action == "generate" or action == "apply_threshold":
                        for register in selected_registers:
                            df = pd.read_excel(combined_file, sheet_name=register)
                            perform_markov_chain_analysis(df, register, register, threshold)
                        flash("Diagram updated." if action == "apply_threshold"
                              else "Diagram generated.")
                except Exception as e:
                    flash(f"Analysis error: {e}")

        # Sheetnamen laden
        if os.path.exists(combined_file):
            try:
                xls = pd.ExcelFile(combined_file)
                registers = xls.sheet_names
            except Exception as e:
                flash(f"Error reading merged file: {str(e)}")
                registers = []
        else:
            flash("Combined file not found.")
            registers = []

        return render_template("markov_chain_analysis.html",
                               selected_registers=selected_registers,
                               registers=registers,
                               threshold=threshold)


