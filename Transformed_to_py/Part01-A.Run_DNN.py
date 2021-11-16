"""
#let's get into the real job!
#check whether tensorflow is correctly installed
import tensorflow as tf
print(tf.__version__)
"""

#import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import os.path
from subprocess import run

def main():
    ### Load data
    scaler_nm= 'QuantileTransformed' #'MinMaxScaled'  #
    indir= '../../climate_ai_lecture_data/'
    outdir= indir+'saved_model_dnn.{}/'.format(scaler_nm)
    if not os.path.isdir(outdir):
        run('mkdir {}'.format(outdir),shell=True)
        print("Out-directory is made: "+outdir)

    data=[]
    for fn in ['train_x','train_y','test_x','test_y']:
        infn= indir+fn+'_{}.npy'.format(scaler_nm)
        data.append(np.load(infn))
        print("Loaded: ",fn, data[-1].shape)

    x_train, y_train, x_test, y_test = data

    ### Build a DNN model
    dnn = keras.Sequential([
        layers.Dense(64, activation='relu', input_shape=[24]),
        layers.Dense(128, activation='relu'),
        layers.Dense(256, activation='relu'),
        layers.Dense(516, activation='sigmoid'),
        layers.Dropout(0.5),
        layers.Dense(1)
        ])
    optimizer = keras.optimizers.Adam(learning_rate=0.0005, beta_1=0.9, beta_2=0.999, epsilon=1e-07)

    ### Once the model is created, you can config the model with losses and metrics
    ### with model.compile(),
    ### train the model with model.fit(),
    ### or use the model to do prediction with model.predict().
    dnn.compile(loss='mse', optimizer=optimizer, metrics=['mse'])
    print(dnn.summary())

    ### Let's do whether model works okay
    test=True
    if test:
        example_batch = x_train[0:2]
        example_result = dnn.predict(example_batch)
        print(example_result)
        print(np.shape(x_train[0:2]), np.shape(example_result))

    ### Set check-point
    save_weights = keras.callbacks.ModelCheckpoint(
        filepath= outdir+"dnn_4layers_weights.E{epoch:04d}.ckpt",
        verbose=1,
        save_weights_only=True,  ## (model.save_weights(filepath)) or (model.save(filepath))
        period=25) #save_freq=5)
    save_models = keras.callbacks.ModelCheckpoint(
        filepath= outdir+"dnn_4layers_models.E{epoch:04d}.ckpt",
        verbose=1,
        save_weights_only=False,  ## (model.save_weights(filepath)) or (model.save(filepath))
        period=100) #save_freq=5)

    ### Let's do training with model.fit
    EPOCHS = 200
    history = dnn.fit(x_train, y_train, epochs=EPOCHS,
        validation_split = 0.1, verbose=1,
        callbacks = [save_weights,save_models])

    #keras.models.save_model(dnn, indir+"saved_model_dnn/")
    dnn.save(outdir)

    ### history.history is a dictionry
    for key, value in history.history.items():
        print("{}: {}".format(key, value[-5:]))

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
    for (ydata,ylabel) in y:
        ax1.plot(xdata,ydata,label=ylabel)

    ax1.set_xlabel(xlabel)
    ax1.set_ylim([0,0.002])
    ax1.grid()
    ax1.legend(loc='best')

    plt.show()
    return fig

if __name__=="__main__":
    main()
