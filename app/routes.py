from flask import render_template

def setup_routes(app):
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
