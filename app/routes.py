from flask import render_template, request, redirect, flash
import pandas as pd
import os
from werkzeug.utils import secure_filename

UPLOAD_FOLDER = "uploads" # folder for uploads
ALLOWED_EXTENSIONS = {'xlsx'} # only excel

# function to check the extension
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def setup_routes(app):
    app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER # set folder on app configuration
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)  # creating folder if it does not exist

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

    # route for upload just POST requests
    @app.route("/upload", methods=["POST"])
    def upload():
        # get the files
        files = request.files.getlist("files")
        # if no files
        if not files:
            flash("Keine Dateien hochgeladen.")
            return redirect("/") # back to starting page
        # going through files
        for file in files:
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                # save file on server
                file.save(filepath)
                # processing file
                process_excel_file(filepath) # method call
            else:
                flash(f"{file.filename} ist keine gültige Excel-Datei.")

        flash("Dateien erfolgreich hochgeladen und verarbeitet.")
        return redirect("/")

# processing files
def process_excel_file(filepath):
    # open with pandas
    xls = pd.ExcelFile(filepath)
    print(f" Verarbeite Datei: {filepath}")
    # go through all tables
    for sheet_name in xls.sheet_names:
        df = xls.parse(sheet_name) # load current sheet as dataframe
        # checking format of columns
        if list(df.columns[:2]) != ["Segment", "Code"]:
            print(f" Fehler: Tab '{sheet_name}' hat falsche Spalten in Datei {filepath}")
            continue # next sheet

        print(f"✅ Tab '{sheet_name}' OK in Datei {filepath}")
        print(df.head())
