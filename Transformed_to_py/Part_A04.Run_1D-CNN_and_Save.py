"""
#let's get into the real job!
#check whether tensorflow is correctly installed
import tensorflow as tf
print(tf.__version__)
"""

import sys
import os.path
import numpy as np
from subprocess import run
#import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def fc():
    #make 1d-CNN
    #filter: the dimensionality of the output space
    #.      (i.e. the number of output filters in the convolution)
    #kernel_size: specifying the length of the 1D convolution window.
    #strides: Specifying the stride length of the convolution.
    model = keras.Sequential([
        layers.Conv1D(filters=64, kernel_size=3, strides=1, activation='relu', input_shape=(24,1)),
        layers.Conv1D(filters=128, kernel_size=5, strides=1, activation='relu'),
        layers.Conv1D(filters=256, kernel_size=10, strides=1, activation='sigmoid'),
        layers.MaxPooling1D(pool_size=2),
        layers.Flatten(),
        layers.Dropout(0.5),
        layers.Dense(1)
        ])

    optimizer = keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07)
    model_name= '1d-cnn_3layers'
    return model,optimizer,model_name

def main():
    ### Load data
    scaler_nm= 'MinMaxScaled'  #'QuantileTransformed' #
    indir= '../../climate_ai_lecture_data/'

    data=[]
    for fn in ['train_x','train_y','test_x','test_y']:
        infn= indir+fn+'_{}.npy'.format(scaler_nm)
        data.append(np.load(infn))
        print("Loaded: ",fn, data[-1].shape)

    x_train, y_train, x_test, y_test = data
    ## Make x-data 3D
    x_train = np.expand_dims(x_train, axis=2)
    x_test = np.expand_dims(x_test, axis=2)
    print(np.shape(x_train), np.shape(y_train), np.shape(x_test), np.shape(y_test))

    ### Build a CNN model
    cnn, optimizer, model_name= fc()
    outdir= indir+'saved_model.{}.{}/'.format(model_name,scaler_nm)
    if not os.path.isdir(outdir):
        run('mkdir {}'.format(outdir),shell=True)
        print("Out-directory is made: "+outdir)

    ### Once the model is created, you can config the model with losses and metrics
    ### with model.compile(),
    ### train the model with model.fit(),
    ### or use the model to do prediction with model.predict().
    cnn.compile(loss='mse', optimizer=optimizer, metrics=['mse'])
    print(cnn.summary())

    ### Let's do whether model works okay
    test=True
    if test:
        example_batch = x_train[0:2]
        example_result = cnn.predict(example_batch)
        print(example_result)
        print(np.shape(x_train[0:2]), np.shape(example_result))

    ### Set check-point
    BATCH_SIZE = 36
    VAL_SPLIT= 0.1
    STEPS_PER_EPOCH = int(y_train.shape[0]*(1-VAL_SPLIT) / BATCH_SIZE)
    PERIOD_W, PERIOD_M = 20,100
    EPOCHS = 200
    print(STEPS_PER_EPOCH)

    save_weights = keras.callbacks.ModelCheckpoint(
        filepath= outdir+model_name+"_weights.E{epoch:04d}.ckpt",
        verbose=1,
        save_weights_only=True,  ## (model.save_weights(filepath)) or (model.save(filepath))
        save_freq= PERIOD_W * STEPS_PER_EPOCH)
    save_models = keras.callbacks.ModelCheckpoint(
        filepath= outdir+model_name+"_models.E{epoch:04d}.ckpt",
        verbose=1,
        save_weights_only=False,  ## (model.save_weights(filepath)) or (model.save(filepath))
        save_freq= PERIOD_M * STEPS_PER_EPOCH)

    ### Let's do training with model.fit
    history = cnn.fit(x_train, y_train, epochs=EPOCHS,
        validation_split = VAL_SPLIT, verbose=1,
        batch_size=BATCH_SIZE, steps_per_epoch=STEPS_PER_EPOCH,
        callbacks = [save_weights,save_models])

    #keras.models.save_model(cnn, indir+"saved_model_cnn/")
    #cnn.save(outdir)  ## Skip since "save_models" are already set

    ### history.history is a dictionry
    for key, value in history.history.items():
        print("{}: {}".format(key, value[-5:]))

    ### Draw Errors' evolution by Epochs
    fig1= plot_error_by_epoch(x=(history.epoch,'Epoch'),
        y=[(history.history['mse'],'Train Error'),(history.history['val_mse'],'Val. Err')]
    )
    fig1.savefig(outdir+'{}_Error_Evol.png'.formt(model_name))
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
