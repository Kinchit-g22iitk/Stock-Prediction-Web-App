from flask import Flask,request,render_template

from src.stockdata import StockData
from src.data_processing import DataProcessing
from src.models.gru import GRU_Model

from PIL import Image
import base64
import io

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
        numDays = int(request.form.get('numOfDays'))

        stockData = StockData()
        df = stockData.getStockData(tickerSymbol=stockName,startDate='2015-6-1',endDate='2023-6-15')
        graph_loc = stockData.getCloseGraph()

        im = Image.open(graph_loc)
        data = io.BytesIO()
        im.save(data, "JPEG")
        encoded_img_data = base64.b64encode(data.getvalue())

        dataProcessing = DataProcessing()
        df = dataProcessing.feature_selection(df)
        data = dataProcessing.transform(df)
        input_sequences, targets = dataProcessing.get_sequential_data(numDays)
        train_x,train_y,test_x,test_y = dataProcessing.get_train_test_data(input_sequences,targets)
        prediction_data = dataProcessing.get_prediction_data()

        # print(train_x.shape, train_y.shape, test_x.shape, test_y.shape)
        # print(prediction_data.shape)
        gru_model = GRU_Model(numDays)
        gru_model.train(train_x,train_y)
        print(gru_model.test(test_x,test_y))

        result = gru_model.prediction(input_sequences, targets, prediction_data)

        return render_template('home.html', img_data=encoded_img_data.decode('utf-8'), result=result)
    
if __name__=="__main__":
    app.run(port=8000,debug=True)