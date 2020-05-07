import torch

import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import seaborn as sns
from pylab import rcParams
import matplotlib.pyplot as plt
from matplotlib import rc
from sklearn.preprocessing import MinMaxScaler
from pandas.plotting import register_matplotlib_converters
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
import random
from scipy.interpolate import UnivariateSpline  
import pickle
from sklearn.decomposition import PCA
import time
import datetime
from sklearn.metrics import mean_squared_error
from math import sqrt


class MV_LSTM(torch.nn.Module):
    def __init__(self,n_features,seq_length):
        super(MV_LSTM, self).__init__()
        self.n_features = n_features
        self.seq_len = seq_length
        self.n_hidden = 30 # number of hidden states
        self.n_layers = 4 # number of LSTM layers (stacked)
        self.dropout = nn.Dropout(0.1) 

        self.l_lstm = torch.nn.LSTM(input_size = n_features, 
                                 hidden_size = self.n_hidden,
                                 num_layers = self.n_layers, 
                                 batch_first = True
                                 )
        # according to pytorch docs LSTM output is 
        # (batch_size,seq_len, num_directions * hidden_size)
        # when considering batch_first = True
        self.l_linear = torch.nn.Linear(self.n_hidden*self.seq_len, 100)
        self.sigmoid = nn.Sigmoid()
     
        


    def init_hidden(self, batch_size):
        # even with batch_first = True this remains same as docs
        hidden_state = torch.zeros(self.n_layers,batch_size,self.n_hidden).cuda()
        cell_state = torch.zeros(self.n_layers,batch_size,self.n_hidden).cuda()
        self.hidden = (hidden_state, cell_state)


    def forward(self, x):        
        batch_size, seq_len, _ = x.size()

        lstm_out, self.hidden = self.l_lstm(x,self.hidden)
        #lstm_out, self.hidden = self.l_lstm(x)
        
        # lstm_out(with batch_first = True) is 
        # (batch_size,seq_len,num_directions * hidden_size)
        # for following linear layer we want to keep batch_size dimension and merge rest       
        # .contiguous() -> solves tensor compatibility error
        x = lstm_out.contiguous().view(batch_size,-1)
        return self.sigmoid(self.l_linear(x))
    
from numpy import array
from numpy import hstack

# split a multivariate sequence into samples
def split_sequences(sequences, n_steps):
    X, y = list(), list()
    
    for i in range(0,len(sequences),100):
        # find the end of this pattern
        end_ix = i + n_steps
        # check if we are beyond the dataset
        if i!=0 and end_ix > len(sequences):
            break
        
        sequences[i:end_ix,0]=np.insert(np.diff(sequences[i:end_ix,0]),0,0)
        # gather input and output parts of the pattern
        seq_x, seq_y = sequences[i:end_ix-33], sequences[end_ix-33:end_ix]
        
        X.append(seq_x)
        y.append(seq_y)
    return array(X), array(y)

#read training data#############################################################################################
df = pd.read_csv('data/time-series-19-covid-combined-4.csv', skiprows=1)
df.head()
df.info()
df.columns = ['day','country', 'territory', 'lat','long','confirmed','recovered','deaths']

is_china =  (df['country']=='China')

#read testing data##############################################################################################
df2 = pd.read_csv('data/time-series-19-covid-combined-4.csv', skiprows=1)
df2.head()
df2.info()
df2.columns = ['day','country', 'territory', 'lat','long','confirmed','recovered','deaths']

is_indonesia =  (df2['country']=='Indonesia')

#training data filtering#########################################################################################
data=df[df.country.isin(['China','Germany','Australia','Brazil','US','Belgium','Spain','Italy','UK','France','Japan','Malaysia','Vietnam','Iran','UEA','Singapore','Thailand','Korea, South','Japan','Iran','Netherlands','Russia','Chile','India','Greece','Mexico','Mongolia','Philippines','New Zealand','South Africa','Botswana','Uruguay','Paraguay','Madagascar','Peru', 'Portugal', 'Denmark','Hungary','Kenya','Ireland','Israel','Norway','Mauritius','Rwanda','Iceland','Kazakhstan','Switzerland','Cyprus','Zimbabwe'])][['confirmed','lat','long','recovered','deaths']]


#testing data filtering#########################################################################################
data2=df2[(is_indonesia)][['confirmed','lat','long','recovered','deaths']]
date=df2[(is_indonesia)][['day','confirmed']]

date.day = pd.to_datetime(date.day,format='%Y%m%d', errors='ignore')
date.set_index('day', inplace=True)
################################################################################################################

n_features = 5 # this is number of parallel inputs
n_timesteps = 100 # this is number of timesteps


#input splitting################################################################################################
X, Y = split_sequences(data.values, n_timesteps)



print (X.shape,Y.shape)

#normalization##################################################################################################
alld=np.concatenate((X,Y),1)
alld=alld.reshape(alld.shape[0]*alld.shape[1],alld.shape[2])



scaler = MinMaxScaler()
scaler.fit(alld)
X=[scaler.transform(x) for x in X]
y=[scaler.transform(y) for y in Y]

X=np.array(X)
y=np.array(y)[:,:,0]


#training#########################################################################################

mv_net = MV_LSTM(n_features,67).cuda()
criterion = torch.nn.MSELoss() # reduction='sum' created huge loss value
optimizer = torch.optim.Adam(mv_net.parameters(), lr=1e-3)

train_episodes = 10000

batch_size = 16

mv_net.train()

for t in range(train_episodes):
    
    for b in range(0,len(X),batch_size):
       
        p = np.random.permutation(len(X))
        
        inpt = X[p][b:b+batch_size,:,:]
        target = y[p][b:b+batch_size,:]    
        
        x_batch = torch.tensor(inpt,dtype=torch.float32).cuda()    
        y_batch = torch.tensor(target,dtype=torch.float32).cuda()
       
        mv_net.init_hidden(x_batch.size(0))
        
        output = mv_net(x_batch) 
        
        
        all_batch=torch.cat((x_batch[:,:,0], y_batch), 1)
        
        
        loss = 1000*criterion(output.view(-1), all_batch.view(-1))  

        loss.backward()
        optimizer.step()        
        optimizer.zero_grad() 
    print('step : ' , t , 'loss : ' , loss.item())


#evaluation#########################################################################################################
#data2x=data2[~(data2.confirmed==0)]
data2x=data2
truth = data2

data2x.values[0:len(data2x),0]=np.insert(np.diff(data2x.values[0:len(data2x),0]),0,0)
data2x=scaler.transform(data2x) 


X_test = np.expand_dims(data2x, axis=0)
print (X_test.shape)
mv_net.init_hidden(1)


lstm_out = mv_net(torch.tensor(X_test[:,-67:,:],dtype=torch.float32).cuda())
lstm_out=lstm_out.reshape(1,100,1).cpu().data.numpy()

print (data2x[-67:,0],lstm_out)
actual_predictions = scaler.inverse_transform(np.tile(lstm_out, (1, 1,5))[0])[:,0]

print (data2.values[-67:,0],actual_predictions)

#actual_predictions=lstm_out


x = np.arange(0, 54, 1)
x2 = np.arange(0, 67, 1)
x3 = np.arange(0, 100, 10)
x4 = np.arange(0, 50, 1)


#save prediction
with open('./lstmdata/predict_indo8.pkl', 'wb') as f:  # Python 3: open(..., 'wb')
    pickle.dump(pd.Series(actual_predictions), f,protocol=2)

#visualization####################################################################################################    
fig, ax = plt.subplots() 
plt.title('Days vs Confirmed Cases Accumulation')
plt.ylabel('Confirmed')

left, width = .25, .5
bottom, height = .25, .5
right = left + width
top = bottom + height

print (date.index)
date_list=pd.date_range(start=date.index[0],end=date.index[-1])
print (date_list)

plt.axvline(x=np.array(date_list)[66], color='r', linestyle='--')

ax.text(0.2*(left+right), 0.8*(bottom+top), 'input sequence',
        horizontalalignment='left',
        verticalalignment='center',
        fontsize=10, color='red',
        transform=ax.transAxes)
ax.text(0.0125*(left+right), 0.77*(bottom+top), '______________________',
        horizontalalignment='left',
        verticalalignment='center',
        fontsize=20, color='red',
        transform=ax.transAxes)



sumpred=np.cumsum(np.absolute(actual_predictions))

print (date.values.shape) 
print (sqrt(mean_squared_error(date.confirmed,sumpred)))          
#plt.plot(date.values[-67:],np.cumsum(data2.confirmed.values[-67:]))
plt.plot(np.array(date_list),sumpred,label='Prediction')
plt.plot(np.array(date_list),date.confirmed,label='Actual')
plt.xticks(rotation=90)
fig.autofmt_xdate()
plt.legend(loc=2)
plt.show() 

