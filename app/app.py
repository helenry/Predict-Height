from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from joblib import load

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"]) # jika url ditambahkan / maka akan menjalankan kode di bawah
def hello_world():
    request_type = request.method
    if request_type == "GET":
        return render_template("index.html", href="static/base.svg")
    else:
        input_data = request.form["input"]
        path = "static/prediction.svg"
        model = load("model.joblib")
        input_data = input_to_np(input_data)
        make_picture("AgesAndHeights.pkl", model, input_data, path)
        return render_template("index.html", href=path)

def make_picture(train_data_file, model, new_data_np, output_file):
    data = pd.read_pickle(train_data_file)
    data = data[data["Age"] > 0]

    ages = data["Age"]
    heights = data["Height"]

    x_new = np.array(list(range(19))).reshape(19, 1) # untuk membuat garis lurus
    preds = model.predict(x_new)

    fig = px.scatter(x=ages, y=heights, title="Age and Height Correlation", labels={"x": "Age (years)", "y": "Height (inches)"})
    fig.add_trace(go.Scatter(x=x_new.reshape(19), y=preds, mode="lines", name="Model")) # garis

    # new input data
    new_preds = model.predict(new_data_np)
    fig.add_trace(go.Scatter(x=new_data_np.reshape(len(new_data_np)), y=new_preds, name="New Predictions", mode="markers", marker=dict(color="orange", size=15)))
    fig.write_image(output_file, width=800, engine="kaleido")
    fig.show()

def input_to_np(input_data):
    def is_float(s):
        try:
            float(s)
            return True
        except:
            return False
    floats = np.array([float(x) for x in input_data.split(",") if is_float(x)])
    return floats.reshape(len(floats), 1)