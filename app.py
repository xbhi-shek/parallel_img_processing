from flask import Flask, render_template, request, send_from_directory
import os
import subprocess

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
OUTPUT_FOLDER = "outputs"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["OUTPUT_FOLDER"] = OUTPUT_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files["file"]
        task = request.form["task"]
        
        if file:
            input_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
            output_path = os.path.join(app.config["OUTPUT_FOLDER"], f"processed_{file.filename}")
            file.save(input_path)

            subprocess.run(["mpirun", "-np", "4", "python3", "process_image.py", input_path, output_path, task])

            return render_template("index.html", uploaded_file=file.filename, processed_file=f"processed_{file.filename}")
    
    return render_template("index.html", uploaded_file=None)

@app.route("/uploads/<filename>")
def uploaded_file(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)

@app.route("/outputs/<filename>")
def processed_file(filename):
    return send_from_directory(app.config["OUTPUT_FOLDER"], filename)

if __name__ == "__main__":
    app.run(debug=True)
