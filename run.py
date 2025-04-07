from flask import Flask

app = Flask(__name__)

# Eine einfache Route
@app.route('/')
def home():
    return 'Hallo, Welt!'

if __name__ == '__main__':
    app.run(debug=True)
