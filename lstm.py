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
#matplotlib inline
#config InlineBackend.figure_format='retina'

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
    
    #data2_scale=data2_scale[~(data2_scale[:,0:3]==0).all(1)]
    #df_china=scaler.fit_transform(df[(is_china)][['confirmed','recovered','deaths']])
    
    #scaler = MinMaxScaler()
    for i in range(0,len(sequences),100):
        # find the end of this pattern
        end_ix = i + n_steps
        # check if we are beyond the dataset
        if i!=0 and end_ix > len(sequences):
            break
        
        sequences[i:end_ix,0]=np.insert(np.diff(sequences[i:end_ix,0]),0,0)
        # gather input and output parts of the pattern
        seq_x, seq_y = sequences[i:end_ix-33], sequences[end_ix-33:end_ix]
        
        
        #scaler.fit(seq_x)
        
        #print (i,len(sequences),seq_x.shape,seq_y.shape)
        
        X.append(seq_x)
        y.append(seq_y)
    return array(X), array(y)


df = pd.read_csv('data/time-series-19-covid-combined-4.csv', skiprows=1)
df.head()
df.info()
df.columns = ['day','country', 'territory', 'lat','long','confirmed','recovered','deaths']

is_china =  (df['country']=='China')

df2 = pd.read_csv('data/time-series-19-covid-combined-4.csv', skiprows=1)
df2.head()
df2.info()
df2.columns = ['day','country', 'territory', 'lat','long','confirmed','recovered','deaths']

is_indonesia =  (df2['country']=='Indonesia')

#data=df[(is_china)][['confirmed','lat','long','recovered','deaths']]
data=df[df.country.isin(['China','Germany','Australia','Brazil','US','Belgium','Spain','Italy','UK','France','Japan','Malaysia','Vietnam','Iran','UEA','Singapore','Thailand','Korea, South','Japan','Iran','Netherlands','Russia','Chile','India','Greece','Mexico','Mongolia','Philippines','New Zealand','South Africa','Botswana','Uruguay','Paraguay','Madagascar','Peru', 'Portugal', 'Denmark','Hungary','Kenya','Ireland','Israel','Norway','Mauritius','Rwanda','Iceland','Kazakhstan','Switzerland','Cyprus','Zimbabwe'])][['confirmed','lat','long','recovered','deaths']]
#data=df[df.country.isin(['China'])][['confirmed','lat','long','recovered','deaths']]
data2=df2[(is_indonesia)][['confirmed','lat','long','recovered','deaths']]

date=df2[(is_indonesia)][['day','confirmed']]
print (date)
date.day = pd.to_datetime(date.day,format='%Y%m%d', errors='ignore')
date.set_index('day', inplace=True)
#data['confirmed']=data['confirmed'].diff().fillna(0)
#data2['confirmed']=data2['confirmed'].diff().fillna(0)


#print (scaler.fit_transform(df[(is_china)][['confirmed','recovered','deaths']].values.astype(float).transpose(1,0)).shape)
#data.values[:,0]=np.insert(np.diff(data.values[:,0]),0,0)






# define input sequence
#in_seq1 = array([x for x in range(0,100,10)])
#in_seq2 = array([x for x in range(5,105,10)])
#out_seq = array([in_seq1[i]+in_seq2[i] for i in range(len(in_seq1))])
# convert to [rows, columns] structure
#in_seq1 = in_seq1.reshape((len(in_seq1), 1))
#in_seq2 = in_seq2.reshape((len(in_seq2), 1))
#out_seq = out_seq.reshape((len(out_seq), 1))
# horizontally stack columns
#dataset = hstack((in_seq1, in_seq2, out_seq))
#print (dataset.shape)

n_features = 5 # this is number of parallel inputs
n_timesteps = 100 # this is number of timesteps


print (data.values.shape)
# convert dataset into input/output
X, Y = split_sequences(data.values, n_timesteps)
#X_test, y_test = split_sequences(data2, n_timesteps)


print (X.shape,Y.shape)

#y_tile=np.tile(np.expand_dims(Y, axis=2), (1, 1,5))
alld=np.concatenate((X,Y),1)
alld=alld.reshape(alld.shape[0]*alld.shape[1],alld.shape[2])



scaler = MinMaxScaler()
scaler.fit(alld)
X=[scaler.transform(x) for x in X]
y=[scaler.transform(y) for y in Y]

X=np.array(X)
y=np.array(y)[:,:,0]


#print (X,X_test)
#quit()

mv_net = MV_LSTM(n_features,67).cuda()
criterion = torch.nn.MSELoss() # reduction='sum' created huge loss value
optimizer = torch.optim.Adam(mv_net.parameters(), lr=1e-3)

train_episodes = 10000

batch_size = 16

mv_net.train()

for t in range(train_episodes):
    
    #np.random.shuffle(X)
    
    for b in range(0,len(X),batch_size):
        #idx = np.random.randint(32, size=32)
        p = np.random.permutation(len(X))
        
        #print (X.shape)
        #X=X[idx]
        #y=y[idx]
        
        #y=sorted(y.all(), key=lambda k: idx)
        
        inpt = X[p][b:b+batch_size,:,:]
        target = y[p][b:b+batch_size,:]    
        #print (inpt.shape)
        x_batch = torch.tensor(inpt,dtype=torch.float32).cuda()    
        y_batch = torch.tensor(target,dtype=torch.float32).cuda()
        #all_batch=torch.cat((x_batch, y_batch), 1).cuda()
        #print (x_batch.shape, y_batch.shape,all_batch.shape)
        
        mv_net.init_hidden(x_batch.size(0))
    #    lstm_out, _ = mv_net.l_lstm(x_batch,nnet.hidden)    
    #    lstm_out.contiguous().view(x_batch.size(0),-1)
        
        output = mv_net(x_batch) 
        
        
        all_batch=torch.cat((x_batch[:,:,0], y_batch), 1)
        
        
        loss = 1000*criterion(output.view(-1), all_batch.view(-1))  

        loss.backward()
        optimizer.step()        
        optimizer.zero_grad() 
    print('step : ' , t , 'loss : ' , loss.item())
    
#data2x=data2[~(data2.confirmed==0)]
data2x=data2
truth = data2





data2x.values[0:len(data2x),0]=np.insert(np.diff(data2x.values[0:len(data2x),0]),0,0)
#scaler = MinMaxScaler()
#scaler.fit(data2)
data2x=scaler.transform(data2x) 


X_test = np.expand_dims(data2x, axis=0)
print (X_test.shape)
mv_net.init_hidden(1)


#print (pd.np.tile(data2.T,(100,1)).shape,X_pca[0:5].shape)
#quit()

lstm_out = mv_net(torch.tensor(X_test[:,-67:,:],dtype=torch.float32).cuda())
lstm_out=lstm_out.reshape(1,100,1).cpu().data.numpy()

#print (X_test[:,-61:,0],lstm_out)

#lstm_out[0][0:58,0]=np.cumsum(lstm_out[0][0:58,0])
#lstm_out = (np.repeat(np.expand_dims(lstm_out,axis=2), 5, axis=2))
#actual_predictions = scaler.inverse_transform(np.array(lstm_out[0]))[:,0]
#longe=np.concatenate([X_test[:,-25:],np.tile(lstm_out, (1, 1,5))],axis=1)
#longe1=X_test[:,-25:]
#longe2=np.tile(lstm_out, (1, 1,5))
#longe[0][:,0]=np.cumsum(longe[0][:,0])
#actual_predictions = scaler.inverse_transform(longe[0])[:,0]
#actual_predictions1 = scaler.inverse_transform(longe1[0])[:,0]
#actual_predictions2 = scaler.inverse_transform(longe2[0])[:,0]
print (data2x[-67:,0],lstm_out)
actual_predictions = scaler.inverse_transform(np.tile(lstm_out, (1, 1,5))[0])[:,0]

#data2.values[0:len(data2),0]=np.insert(np.diff(data2.values[0:len(data2),0]),0,0)
print (data2.values[-67:,0],actual_predictions)

#actual_predictions=lstm_out


x = np.arange(0, 54, 1)
x2 = np.arange(0, 67, 1)
x3 = np.arange(0, 100, 10)
x4 = np.arange(0, 50, 1)

#yinterp=np.insert(actual_predictions[:,0], 0, data2[-1,0] , 0)
#print (yinterp.shape,x.shape) 
#yinterp = UnivariateSpline(x, yinterp, s = 5e8)(x) 
#print (data2[-2,0],yinterp[0]-(yinterp[0]-data2[-2,0]))
#yinterp= (yinterp-(yinterp[0]-data2[-2,0]))
#print (yinterp)
#print (data2.values[-54:-1,0])

##predcum=np.cumsum(np.append(np.diff(test[0:25,0]),np.diff(actual_predictions)))
#predcum=np.cumsum(actual_predictions)
#actualcum=np.cumsum(test[29:-11,0])
#print (actualcum.shape)
#test=np.insert(np.diff(test[:,0]),0,0)

with open('./lstmdata/predict_indo8.pkl', 'wb') as f:  # Python 3: open(..., 'wb')
    pickle.dump(pd.Series(actual_predictions), f,protocol=2)

fig, ax = plt.subplots() 
plt.title('Days vs Confirmed Cases Accumulation')
plt.ylabel('Confirmed')
#plt.grid(True)
#plt.autoscale(axis='x', tight=True)
#plt.axvline(x=24, color='b', linestyle='--', label='The day pm2.5 decreases')

#plt.axvline(x=50, color='g', linestyle='--', label='PSBB dimulai')
#plt.axvline(x=53, color='yellow', linestyle='--', label='prediksi dimulai (14 April 2020)')
#plt.plot(x4,test[33:,0])
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

