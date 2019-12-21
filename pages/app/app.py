from flask import Flask, render_template, request, redirect, url_for, send_from_directory
from wtforms import Form, FloatField, SubmitField, validators, ValidationError
import numpy as np
import os
from werkzeug.utils import secure_filename
from sklearn.externals import joblib
import re
from create_img import analyze
import glob

# 画像のアップロード先のディレクトリ
path = './uploads'

app = Flask(__name__, static_folder="image")

# ファイルを受け取る方法の指定
@app.route('/', methods=['GET', 'POST'])
def uploads_file():
    # リクエストがポストかどうかの判別
    if request.method == 'POST':
        # データの取り出し
        imgfolder = request.files.getlist('pic')
        for img in imgfolder:
            filename = str(os.path.split(img.filename)[1])
            img.save(path + '/' + filename)
        
        analyze()
        analyzedImages = glob.glob('image/*.jpg')

        return render_template('result.html', images=analyzedImages)

    if request.method == 'GET':
        return render_template('index.html')

if __name__ == "__main__":
    app.run()