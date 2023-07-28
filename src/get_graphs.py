from matplotlib import pyplot as plt
import os
import sys
folder_path = os.getcwd()
sys.path.append(folder_path)
from src.logger import logging

def get_graphs(pred, test_y):
    l = []
    logging.info('pred has shape {}'.format(pred.shape))
    graphs_loc = os.path.join(folder_path,'graphs')
    for i in range(pred.shape[1]):
        x = pred[:,i,:]
        y = test_y[:,i,:]
        print(x.shape, y.shape)
        plt.plot(x, 'r', label = 'Prediction')
        plt.plot(y, 'b', label = 'Actual')
        s = 'stock' + str(i)+'.jpg'
        graph_loc = os.path.join(graphs_loc, s)
        plt.savefig(graph_loc)
        l.append(graph_loc)
    
    return l