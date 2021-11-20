"""
Test two cases
1. Load checkpoint(weight only) and do prediction
2. Load whole model and train further
"""

import sys
import os.path
import numpy as np
from subprocess import run
#import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import MinMaxScaler,QuantileTransformer
import joblib

def fc():
    model = keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=[24]),
    layers.Dense(128, activation='relu'),
    layers.Dense(256, activation='relu'),
    layers.Dense(516, activation='sigmoid'),
    layers.Dropout(0.5),
    layers.Dense(1)
    ])
    optimizer = keras.optimizers.Adam(learning_rate=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-07)
    return model,optimizer

def main():
    ### Now load checkpoint at epoch 125
    var_name= 'PM25'
    scaler_nm= 'MinMaxScaled'  #'QuantileTransformed' #
    indir= '../../climate_ai_lecture_data/'
    outdir= indir+'saved_model_dnn.{}/'.format(scaler_nm)
    tgt_epoch= 125
    fn_header= outdir+'dnn_4layers_weights.E{:04d}'.format(tgt_epoch)

    data=[]
    for fn in ['train_x','train_y','test_x','test_y']:
        infn= indir+fn+'_{}.npy'.format(scaler_nm)
        data.append(np.load(infn))
        print("Loaded: ",fn, data[-1].shape)
    x_train, y_train, x_test, y_test = data

    ## Define new model dnn2 and compile
    dnn2, optimizer = fc()
    dnn2.compile(loss='mse', optimizer=optimizer, metrics=['mse'])

    ## Load checkpoint and predict
    dnn2.load_weights(fn_header+".ckpt")
    loss, mse = dnn2.evaluate(x_test, y_test, verbose =2)
    y_pred= dnn2.predict(x_test).squeeze()
    print(type(y_pred), y_pred.shape, y_test.shape)

    ## De-normalize
    ## Load scaler
    var_name1= var_name+'_{}'.format(scaler_nm)
    indir= '../../climate_ai_lecture_data/'
    scaler= joblib.load(indir+'{}_params.joblib'.format(var_name1))

    print("MSE of testset: {:5.2f} ug/m^3".format(scaler.inverse_transform([[mse]])[0,0])) ## [[val]] represents shape of (1,1)

    sys.exit()
    ### Draw Errors' evolution by Epochs
    fig1= plot_error_by_epoch(x=(history.epoch,'Epoch'),
        y=[(history.history['mse'],'Train Error'),(history.history['val_mse'],'Val. Err')]
    )
    fig1.savefig(outdir+'Error_Evol.png')
    return

def plot_error_by_epoch(x,y):
    import matplotlib.pyplot as plt
    fig= plt.figure()
    #fig.subplots_adjust(hspace=0.25)
    fig.suptitle('Error by Epoch')
    #nrow, ncol= 1,1

    ### Plot the data
    ax1 = fig.add_subplot(1,1,1)
    xdata,xlabel= x
    ymax=[]
    for (ydata,ylabel) in y:
        ax1.plot(xdata,ydata,label=ylabel)
        ymax.append(max(ydata[-10:]))
    ymax= max(ymax)*3
    ax1.set_xlabel(xlabel)
    ax1.set_ylim([0,ymax])
    ax1.grid()
    ax1.legend(loc='best')

    plt.show()
    return fig

if __name__=="__main__":
    main()
