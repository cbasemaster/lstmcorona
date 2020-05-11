# lstmcorona

This is LSTM to predict the growth of COVID-19 pandemics

Before using please download the dataset here:

https://github.com/datasets/covid-19/tree/master/data 

Please modify the data pre-processing by yourself, for example if the duration of each sample is more than 100 days, you should modify the "split_sequences" function inside the code such that the proportion of input and label is around 75:25 or 70:30

