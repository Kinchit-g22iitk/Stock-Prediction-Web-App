from flask import Flask,request,render_template


from src.stockdata import StockData
from src.data_processing import DataProcessing
from src.models.gru import GRU_Model
from src.models.lstm import LSTM_Model
from src.models.simple_rnn import RNN_Model
from src.models.bidirectional_rnn import BidirectionalRNN_Model
from src.models.encoder import Encoder_Model
from src.get_graphs import get_graphs

from PIL import Image
import base64
import io

import numpy as np
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
        print(stockName)
        numDays = int(request.form.get('numOfDays'))
        print(numDays)
        model_type = request.form.get('model_type')
        stockData = StockData()
        df = stockData.getStockData(tickerSymbol=stockName,time='10y')
        last_price=df["Close"][-1]
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

        # gru_model = GRU_Model(numDays)
        # gru_model.train(train_x,train_y)
        # print(gru_model.test(test_x,test_y))
        if model_type == 'GRU':
            model = GRU_Model(numDays)
        elif model_type == 'LSTM':
            model = LSTM_Model(numDays)
        elif model_type == 'RNN':
            model = RNN_Model(numDays)
        elif model_type == 'Bidirectional-RNN':
            model = BidirectionalRNN_Model(numDays)
        elif model_type == 'Encoder-model':
            model = Encoder_Model(numDays)

        model.train(train_x, train_y)
        pred = model.Predict(test_x)

        # for i in range(numDays):
        #     pred[:,i,:]=dataProcessing.inv_transform(pred[:,i,:])
        graphs= get_graphs(pred,test_y)
        print(graphs)

        l = []
        for i in graphs:
            im = Image.open(i)
            data = io.BytesIO()
            im.save(data, "JPEG")
            encoded_img_data2 = base64.b64encode(data.getvalue())
            l.append(encoded_img_data2.decode('utf-8'))

        result = model.prediction(input_sequences, targets, prediction_data)
        result = np.array([dataProcessing.inverse_transform(result,numDays)])
        
        
        new_graphs = []
        for i in result[0]:
            new_graphs.append(i[4])
        per_change = []
        for i in range(numDays):
            if(i==0):
                per_change.append(100*(new_graphs[0]-last_price)/last_price)
            else:
                per_change.append(100*(new_graphs[i]-new_graphs[i-1])/new_graphs[i-1])    
        
        return render_template('home.html', img_data=encoded_img_data.decode('utf-8'), graphs_location=l, results=new_graphs, iterations=numDays, change=per_change)
    
if __name__=="__main__":
    app.run(port=8000,debug=True)
