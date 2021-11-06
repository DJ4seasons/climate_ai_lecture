"""
#Data Preprocessing
#Data downloaded from
#https://www.airkorea.or.kr/web/last_amb_hour_data?pMENU_NO=123
"""
import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler,QuantileTransformer

def main():
    ### Read file and get values as array
    var_name= 'PM25'
    pm25= get_array_from_file(var_name)

    ### Let's lookat the distribution of pm2.5 of junggu!
    plot_histogram(pm25,var_name,tit='Distribution of '+var_name)

    ### Normalization
    scaler= MinMaxScaler()
    pm25_norm= scaler.fit_transform(pm25)
    var_name1= var_name+'_MinMaxScaled'
    plot_histogram(pm25_norm,var_name1,tit='Distribution of '+var_name1)

    scaler2= QuantileTransformer()  ## Default n_quantiles=1000
    pm25_norm2= scaler2.fit_transform(pm25)
    var_name2= var_name+'_QuantileTransformed'
    plot_histogram(pm25_norm,var_name2,tit='Distribution of '+var_name2)


    return

def plot_histogram(data,var_name,tit=''):
    ### Draw histogram of the data
    n, bins, patches = plt.hist(data, 50,facecolor='g', alpha=0.75)
    plt.xlabel(var_name)
    plt.ylabel('Count')
    plt.title(tit)
    plt.grid()
    plt.show()
    return

def get_array_from_file(var_name):
    indir= '../../climate_ai_lecture_data/'
    infn= indir+'2018_2019_2020.csv'
    ## This CSV file has 1-line header
    ## let's collect pm2.5 (초미세먼지(㎍/㎥)) at 중구
    with open(infn,'r') as f:
        vals=[]
        miss_count=0
        for i,line in enumerate(f):
            if i==0:
                header= line.strip().split(',')
                print('Header: {}'.format(header))
                try:
                    h_idx= header.index(var_name)
                    print(h_idx, header[h_idx])
                except:
                    sys.exit(var_name+' is not available in header')
            else:
                ww=line.strip().split(',')
                if ww[-1] == '서울 중구 덕수궁길 15':
                    try:
                        val = float(ww[h_idx])
                        vals.append(val)
                    except:
                        # when there is no data copy from previous data point
                        vals.append(vals[-1])
                        miss_count+= 1
                        pass

    ## List to numpy array
    vals= np.asarray(vals)
    print('''
Array shape= {},
miss_count= {},
Min and Max= {}, {}'''.format(vals.shape, miss_count, vals.min(), vals.max()))
    return vals

if __name__=="__main__":
    main()
