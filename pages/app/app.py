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
UPLOAD_FOLDER = './uploads'

app = Flask(__name__, static_folder="uploads")

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

path = './uploads'

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
        analyzedFolder = []
        analyzedImages = glob.glob('image/*.jpg')
        for p in analyzedImages:
            analyzedImagesName = p.filename
            analyzedFolder.append(analyzedImagesName)

        return render_template('result.html', images=analyzedFolder)

    if request.method == 'GET':
        return render_template('index.html')

if __name__ == "__main__":
    app.run()