from flask import Flask,request,render_template

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('home.html')

@app.route('/predict',methods=['GET','POST'])
def predict():
    if request.method == 'GET':
        return render_template('home.html')
    else:
        stockName = request.form.get('stockName')
        numDays = request.form.get('numOfDays')
        print(stockName)
        print(numDays)
        return render_template('home.html')
    
if __name__=="__main__":
    app.run(port=8000,debug=True)