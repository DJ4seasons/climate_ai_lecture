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
    y_pred= dnn2.predict(x_test) #.squeeze()
    print(type(y_pred), y_pred.shape, y_test.shape)

    ## De-normalize
    ## Load scaler
    var_name1= var_name+'_{}'.format(scaler_nm)
    indir= '../../climate_ai_lecture_data/'
    scaler= joblib.load(indir+'{}_params.joblib'.format(var_name1))

    print("MSE of testset: {:5.2f} ug/m^3".format(scaler.inverse_transform([[mse]])[0,0])) ## [[val]] represents shape of (1,1)

    y_test_denorm= scaler.inverse_transform(y_test)
    y_pred= scaler.inverse_transform(y_pred)

    ### Check predictied results with plots
    fig1= plot_pred_vs_true(y_test_denorm, y_pred)
    fig1.savefig(outdir+'pred_vs_true.png')

    ### This time, loading saved model and train a little more
    tgt_epoch2= 100
    fn_header= outdir+'dnn_4layers_models.E{:04d}'.format(tgt_epoch2)
    ## Load checkpoint and predict
    #dnn3.load_model(fn_header+".ckpt")
    dnn3= keras.models.load_model(fn_header+".ckpt")

    ## Train further!
    history2 = dnn3.fit(x_train, y_train, epochs=tgt_epoch-tgt_epoch2,
                 validation_split = 0.1, verbose=1)

    loss,  mse = dnn3.evaluate(x_test, y_test, verbose=2)
    y_pred2= dnn3.predict(x_test) #.squeeze()
    print(type(y_pred2), y_pred2.shape, y_test.shape)

    ## De-normalize
    print("MSE of testset: {:5.2f} ug/m^3".format(scaler.inverse_transform([[mse]])[0,0])) ## [[val]] represents shape of (1,1)

    y_pred2= scaler.inverse_transform(y_pred2)

    ### Check predictied results with plots
    fig2= plot_pred_vs_true(y_test_denorm, y_pred2)
    fig2.savefig(outdir+'pred_vs_true.v2.png')

    return

def plot_pred_vs_true(y_test,y_pred):
    import matplotlib.pyplot as plt
    fig= plt.figure()
    fig.set_size_inches(8.5,5)    # Physical page size in inches, (lx,ly)
    #fig.subplots_adjust(hspace=0.25)
    fig.suptitle('Prediction vs. True',fontsize=16)
    nrow, ncol= 1,2

    ### Plot the data
    ax1 = fig.add_subplot(nrow,ncol,1)
    ax1.scatter(y_test,y_pred)
    xlim,ylim= ax1.get_xlim(), ax1.get_ylim()
    amax= max(xlim[1],ylim[1])
    xr= np.linspace(0,amax,100)
    ax1.plot(xr,xr,c='0.6',ls='--',lw=2.5)
    ax1.set_xlabel("True Values")
    ax1.set_ylabel("Predictions")
    ax1.grid()

    ax2 = fig.add_subplot(nrow,ncol,2)
    ax2.hist(y_pred-y_test,bins=25)
    ax2.yaxis.tick_right()
    ax2.set_xlabel('Prediction Error')
    ax2.set_ylabel('Count',rotation=-90,va='bottom')
    ax2.yaxis.set_label_position("right")

    ax2.grid()

    plt.show()
    return fig

if __name__=="__main__":
    main()
