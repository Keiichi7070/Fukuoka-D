from flask import Flask, render_template, request, redirect, url_for, send_from_directory
from wtforms import Form, FloatField, SubmitField, validators, ValidationError
import numpy as np
import os
from werkzeug.utils import secure_filename
from sklearn.externals import joblib

# 画像のアップロード先のディレクトリ
UPLOAD_FOLDER = '../uploads'
# アップロードされる拡張子の制限
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'gif'])

app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allwed_file(filename):
    # .があるかどうかのチェックと、拡張子の確認
    # OKなら１、だめなら0
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# ファイルを受け取る方法の指定
@app.route('/', methods=['GET', 'POST'])
def uploads_file():
    # リクエストがポストかどうかの判別
    if request.method == 'POST':
        # ファイルがなかった場合の処理
        if 'pic' not in request.files:
            print('ファイルがありません')
            return redirect(request.url)
        # データの取り出し
        imgfile = request.files['pic']
        print(imgfile)
        # ファイル名がなかった時の処理
        if imgfile.filename == '':
            print('ファイルがありません')
            return redirect(request.url)
        # ファイルのチェック
        if imgfile and allwed_file(imgfile.filename):
            # 危険な文字を削除（サニタイズ処理）
            filename = secure_filename(imgfile.filename)
            # ファイルの保存
            imgfile.save(os.path.join("..","uploads", filename))
            # アップロード後のページに転送
            return redirect(url_for('uploaded_file', filename=filename))
    if request.method == 'GET':
        return render_template('index.html')

@app.route('/uploads/<filename>')
# ファイルを表示する
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

# Flaskとwtformsを使い、index.html側で表示させるフォームを構築する
class Form(Form):

    # html側で表示するsubmitボタンの表示
    submit = SubmitField("判定")

if __name__ == "__main__":
    app.run()