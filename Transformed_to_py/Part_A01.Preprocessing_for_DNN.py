"""
#Data Preprocessing
#Data downloaded from
#https://www.airkorea.or.kr/web/last_amb_hour_data?pMENU_NO=123
"""
import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler,QuantileTransformer
from sklearn.model_selection import train_test_split
import joblib

def main():
    ### Read file and get values as array
    var_name= 'PM25'
    pm25= get_array_from_file(var_name)

    ### Let's lookat the distribution of pm2.5 of junggu!
    plot_histogram([pm25,],[var_name,],tit='Distribution of '+var_name)

    ### Re-arange as 24 consecutive input,and next one as output
    nt= pm25.shape[0]
    nh_per_day= 24
    pm25_x, pm25_y= [],[]
    for k in range(nh_per_day,nt,1):
        pm25_x.append(pm25[k-nh_per_day:k])
        pm25_y.append(pm25[k])
    pm25_x, pm25_y= np.asarray(pm25_x), np.asarray(pm25_y)
    print(pm25_x.shape, pm25_y.shape)

    ### Training set: 80% test set: 20% with shuffle
    X_train, X_test, y_train, y_test = train_test_split(
                                        pm25_x, pm25_y,
                                        test_size=0.2,
                                        random_state=1234, )  ## Shuffle=True in default
    print('After split: ',X_train.shape, X_test.shape, y_train.shape, y_test.shape)

    ### Normalization
    #scaler, scaler_nm = MinMaxScaler(), 'MinMaxScaled'
    scaler, scaler_nm = QuantileTransformer(), 'QuantileTransformed'

    scaler.fit(X_train.reshape([-1,1]))  ## It accepts [n_samples, n_features], but we know that each features are homogeneous
    xtr_shape= X_train.shape
    X_train= scaler.transform(X_train.reshape([-1,1])).reshape(xtr_shape)
    y_train= scaler.transform(y_train.reshape([-1,1]))
    xte_shape= X_test.shape
    X_test= scaler.transform(X_test.reshape([-1,1])).reshape(xte_shape)
    y_test= scaler.transform(y_test.reshape([-1,1]))

    var_name1= var_name+'_{}'.format(scaler_nm)
    plot_histogram([y_train,y_test],['y_train','y_test'],tit='Distribution of '+var_name1)

    ### Save data
    outdir= '../../climate_ai_lecture_data/'
    for fn,data in zip(['train_x','train_y','test_x','test_y'],
                        [X_train, y_train, X_test, y_test]):
        outfn= outdir+fn+'_{}.npy'.format(scaler_nm)
        np.save(outfn, data)
        print("Saved: ",outfn)

    ### Save scaler
    joblib.dump(scaler,outdir+'{}_params.joblib'.format(var_name1))

    return

def plot_histogram(data_list,var_name_list,tit=''):
    fig= plt.figure()
    fig.subplots_adjust(hspace=0.25)
    fig.suptitle(tit)

    nrow, ncol= 1,len(data_list)

    ### Draw histogram of the data
    n_bins=50
    for i in range(ncol):
        ax1 = fig.add_subplot(nrow,ncol,i+1)
        n, bins, patches = plt.hist(data_list[i], n_bins, alpha=0.75)
        ax1.set_xlabel(var_name_list[i])
        ax1.set_ylabel('Count')
        ax1.grid()
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
