import numpy as np
import tensorflow as tf
import random as rn
np.random.seed(123)
rn.seed(123)
#single thread
session_conf = tf.ConfigProto(
intra_op_parallelism_threads=1,
inter_op_parallelism_threads=1)

from keras import backend as K
tf.set_random_seed(123)
sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
K.set_session(sess)


import configuration.config as cfg
import matplotlib

if cfg.run_config['Xserver'] == False:
    print "No X-server"
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import time
from keras.utils import plot_model
import utilities.utils as util
import numpy as np
import logging
from keras.models import Sequential
from keras.layers.core import Dense
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pylab as pylab
params = {'legend.fontsize': 'x-large',
          'figure.figsize': (15, 5),
         'axes.labelsize': 'x-large',
         'axes.titlesize':'x-large',
         'xtick.labelsize':'x-large',
         'ytick.labelsize':'x-large'}
pylab.rcParams.update(params)
sns.set_style("whitegrid")


data_folder = 'resources/data/NN_ECG/'
look_back = 8
look_ahead = 5
batch_size = 1024
epochs = 100
patience = 1
activation = 'sigmoid'
learning_rate = .001
decay = 0.0
num_neurons_l1 = 64
num_neurons_l2 = 32


def prepare_seq2seq_data(dataset, look_back, look_ahead):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back - look_ahead):
        input_seq = dataset[i:(i + look_back)]
        output_seq = dataset[i + look_back:(i + look_back + look_ahead)]
        dataX.append(input_seq)
        dataY.append(output_seq)

    dataX = np.reshape(np.array(dataX),[-1,look_back])
    dataY = np.reshape(np.array(dataY),[-1,look_ahead])
    print dataY.shape
    return dataX,dataY

train = np.load(data_folder + "train.npy")
validation1 = np.load(data_folder + "validation1.npy")
validation2 = np.load(data_folder + "validation2.npy")
test = np.load(data_folder + "test.npy")

# standardize data. use the training set mean and variance to transform rewst of the sets
train, train_scaler = util.standardize(train[:, 0])
validation1 = train_scaler.transform(validation1[:, 0])
v2_labels = validation2[:, 1]
validation2 = train_scaler.transform(validation2[:, 0])
test_labels = test[:, 1]
test = train_scaler.transform(test[:, 0])

# prepare sequence data and labels
X_train, y_train = prepare_seq2seq_data(train, look_back, look_ahead)
X_validation1, y_validation1 = prepare_seq2seq_data(validation1, look_back, look_ahead)
X_validation2, y_validation2 = prepare_seq2seq_data(validation2, look_back, look_ahead)
X_test, y_test = prepare_seq2seq_data(test, look_back, look_ahead)
_, y_v2_labels = prepare_seq2seq_data(v2_labels, look_back, look_ahead)
_, y_test_labels = prepare_seq2seq_data(test_labels, look_back, look_ahead)



NN = Sequential()
NN.add(Dense(num_neurons_l1, input_shape=(look_back,),activation=activation))
NN.add(Dense(num_neurons_l2,activation=activation))
NN.add(Dense(look_ahead))
NN.compile(loss="mse",optimizer=Adam(lr=learning_rate,decay= decay))
early_stopping = EarlyStopping(monitor='val_loss', patience=patience)
NN.fit(X_train, y_train, epochs=epochs, batch_size=batch_size,validation_data=(X_validation1, y_validation1),callbacks=[early_stopping ],shuffle=True)

train_predictions = NN.predict(X_train, batch_size=batch_size)
train_predictions = train_scaler.inverse_transform(train_predictions)
train_true = train_scaler.inverse_transform(y_train)
print "Train Loss %f" % NN.evaluate(X_train,y_train,batch_size=batch_size,verbose=1)


v2_predictions = NN.predict(X_validation2, batch_size=batch_size)
v2_predictions = train_scaler.inverse_transform(v2_predictions)
v2_true = train_scaler.inverse_transform(y_validation2)
print "V2 Loss %f" % NN.evaluate(X_validation2,y_validation2,batch_size=batch_size,verbose=1)

test_predictions = NN.predict(X_test, batch_size=batch_size)
test_predictions = train_scaler.inverse_transform(test_predictions)
test_true = train_scaler.inverse_transform(y_test)
print "Test loss %f" % NN.evaluate(X_test,y_test,batch_size=batch_size,verbose=1)

plot_dimension = 0
plt.xlabel("Time step")
plt.ylabel("ECG")
plt.plot(v2_true[:,0], label="True value", linewidth=1,color=sns.xkcd_rgb["denim blue"])
plt.plot(v2_predictions[:,0], label="Predicted value", linewidth=1, linestyle="--",color=sns.xkcd_rgb["medium green"])
error = abs(v2_true[:,0] - v2_predictions[:,0])
plt.plot(error, label='Error',color=sns.xkcd_rgb["pale red"], linewidth=0.5)
plt.legend(bbox_to_anchor=(1, .99))
plt.show()

plot_dimension = 0
plt.xlabel("Time step")
plt.ylabel("ECG")
plt.plot(test_true[:,0], label="True value", linewidth=1,color=sns.xkcd_rgb["denim blue"])
plt.plot(test_predictions[:,0], label="Predicted value", linewidth=1, linestyle="--",color=sns.xkcd_rgb["medium green"])
error = abs(test_true[:,0] - test_predictions[:,0])
plt.plot(error, label='Error',color=sns.xkcd_rgb["pale red"], linewidth=0.5)
plt.legend(bbox_to_anchor=(1, .99))
plt.show()

np.save(data_folder + "train_predictions",train_predictions)
np.save(data_folder + "train_true",train_true)
np.save(data_folder + "v2_predictions",v2_predictions)
np.save(data_folder + "v2_true",v2_true)
np.save(data_folder + "v2_labels", y_v2_labels)
np.save(data_folder + "test_predictions",test_predictions)
np.save(data_folder + "test_true",test_true)
np.save(data_folder + "test_labels", y_test_labels)
