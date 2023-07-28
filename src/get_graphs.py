from matplotlib import pyplot as plt
import os
import sys
folder_path = os.getcwd()
sys.path.append(folder_path)
plt.style.use('seaborn')

def get_graphs(pred, test_y):
    l = []
    graphs_loc = os.path.join(folder_path,'graphs')
    for i in range(pred.shape[1]):
        plt.plot(pred[:,i,:], 'r', label = 'Prediction')
        plt.plot(test_y[:,i,:], 'b', label = 'Actual')
        s = 'stock' + str(i)+'.jpg'
        graph_loc = os.path.join(graphs_loc, s)
        plt.savefig(graph_loc)
        l.append(graph_loc)
    
    return l