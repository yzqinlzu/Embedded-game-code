from flask import Flask,render_template,request#导入flask扩展
import os
import go_to_start
#创建flask应用实例，传入__name__确定资源所在路径
app = Flask(__name__,static_folder='static/',)
basedir = os.path.abspath(os.path.dirname(__file__))

UPLOAD_FOLDER='upload'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
#定义路由和视图函数
#flask中定义路由是通过装饰器实现的
#默认请求方法只支持get


#返回一个模板网页
#给网页填充数据
@app.route('/')
def uploader ():
    return render_template("index2.html")

@app.route('/uploader', methods=['POST'])
def upload_file ():
    file_dir = os.path.join(basedir, app.config['UPLOAD_FOLDER'])
    fullfilename=""
    #保存文件
    if request.method == 'POST':
        f = request.files['file']
        new_filename = "flietopredict"
        fullfilename = os.path.join(file_dir,new_filename)+".wav"
        f.save(fullfilename)
    #模型识别
    ans=[]
    ans = go_to_start.strat(fullfilename)
    text="".join(ans)
    print(text)
    return text

#启动web服务器
if __name__ == '__main__':
    app.run(debug=True)
