# lstmcorona

#### This is LSTM to predict the growth of COVID-19 pandemics

The link of paper : https://arxiv.org/pdf/2005.04809.pdf

#### Before using please download the dataset here:

https://github.com/datasets/covid-19/tree/master/data 

Please modify the data pre-processing by yourself, for example if the duration of each sample is more than 100 days, you should modify the "split_sequences" function inside the code such that the proportion of input and label is around 75:25 or 70:30

#### if it is helpful, please cite this paper:
```go
@misc{yudistira2020covid19,
title={COVID-19 growth prediction using multivariate long short term memory},
author={Novanto Yudistira},  
year={2020},  
eprint={2005.04809},  
archivePrefix={arXiv},  
primaryClass={cs.LG} 
}  
```
