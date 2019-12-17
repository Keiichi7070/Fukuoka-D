from flask import Flask, render_template, request, redirect, url_for, send_from_directory
from wtforms import Form, FloatField, SubmitField, validators, ValidationError
import numpy as np
import os
from werkzeug.utils import secure_filename
from sklearn.externals import joblib
import re
from PIL import Image

# 画像のアップロード先のディレクトリ
UPLOAD_FOLDER = '../uploads'
# アップロードされる拡張子の制限
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'gif'])

app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

path = '../uploads'

files = []

for filename in os.listdir(path):
         if os.path.isfile(os.path.join(path, filename)):  #ファイルのみ取得
                     files.append(filename)

def allwed_file(filename):
    # .があるかどうかのチェックと、拡張子の確認
    # OKなら１、だめなら0
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


filenames = os.listdir('./')
imgl=[]
ww=[]
hh=[]
for fname in sorted(filenames):
    path, ext = os.path.splitext( os.path.basename(fname) )
    if ext=='.JPG' and path[0:2]!='._':
        pic=path+ext
        im=Image.open(pic)
        w=im.size[0]
        h=im.size[1]
        print(pic, w, h)
        imgl=imgl+[pic]
        ww=ww+[w]
        hh=hh+[h]

f=open('maggie.html','w')
print('<html>',file=f)
print('<body>',file=f)
print('<table>',file=f)
n=len(imgl)
m=int(n/5)+1
k=-1
for i in range(0,m):
    print('<tr>',file=f)
    for j in range(0,5):
        k=k+1
        if k<=n-1:
            pic=imgl[k]
            w1=200
            h1=int(hh[k]/ww[k]*200)
            print('<td align="center"><img src="'+pic+'" alt="pic" width="'+str(w1)+'", height="'+str(h1)+'"><br><a href="'+pic+'">I'+pic+'<a></td>',file=f)
        else:
            print('<td></td>',file=f)
    print('</tr>',file=f)
print('</table>',file=f)
print('</body>',file=f)
print('</html>',file=f)
f.close()

# ファイルを受け取る方法の指定
@app.route('/', methods=['GET', 'POST'])
def uploads_file():
    # リクエストがポストかどうかの判別
    if request.method == 'POST':
        # データの取り出し
        # imgfolder = request.files['pic']
        for filename in os.listdir(path):
            if os.path.isfile(os.path.join(path, filename)): #ファイルのみ取得
                files.append(filename)
        
    if request.method == 'GET':
        return render_template('index.html')

@app.route('/uploads/<filename>')
# ファイルを表示する
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == "__main__":
    app.run()