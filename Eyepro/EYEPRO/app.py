from flask import Flask, render_template
from main_controller import run_mode
import webbrowser
from threading import Timer
import os

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/start/<mode>')
def start_mode(mode):
    run_mode(mode)
    return f"{mode.capitalize()} mode started!"

def open_browser():
    webbrowser.open_new("http://127.0.0.1:5000/")

if __name__ == '__main__':
    # Only open browser in the main process, not the reloader
    if os.environ.get("WERKZEUG_RUN_MAIN") == "true":
        Timer(1, open_browser).start()
    app.run(debug=True)
