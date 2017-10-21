# -*- coding: utf-8 -*-
"""
Created on Sun Aug 13 09:24:12 2017

@author: 10007434
"""

from flask import Flask, send_file, render_template

app = Flask(__name__)

@app.route("/")
def hello():
    return render_template('Step23_index.html', msg="Hello", name="Python")

@app.route("/python")
def helloPy():
    return "Hello, Flask in python, routing!"

### うまくいかない。
@app.route("/img")
def helloImg():
    return send_file("C:\work\GitHub\python\Step23_Python.png", minetype="image/png")
    
app.run()