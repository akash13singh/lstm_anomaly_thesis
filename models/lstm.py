import time
from keras.layers.core import Dense, Activation, Dropout, RepeatVector
from keras.layers.recurrent import LSTM
from keras.models import Sequential
from keras.layers.wrappers import TimeDistributed
import logging
from keras.optimizers import Adam
logger = logging.getLogger(__name__).addHandler(logging.StreamHandler())
from keras.callbacks import EarlyStopping,Callback
import keras.backend as K


class ResetStatesCallback(Callback):
    def on_epoch_begin(self, epoch, logs={}):
        print "Resetting states before epoch %d"%(epoch)
        self.model.reset_states()

    # def on_batch_begin(self, batch, logs={}):
    #     print "batch number: %d"%(batch)
    #     lr = K.get_value(self.model.optimizer.lr)
    #     print lr

class VanillaLSTM(object):
    #layers = {input: 1, 2: 64, 3: 256, 4: 100, output: 1}
    def __init__(self,look_back,layers,dropout,loss,learning_rate):
        self.look_back = look_back
        self.n_hidden = len(layers) -2
        self.model = Sequential()
        self.layers = layers
        self.loss = loss
        #self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.dropout = dropout
        logging.info("Vanilla LSTM Model Info: %s"%(locals()))

    def build_model(self):
        #first add input to hidden1
        self.model.add(LSTM(
            input_length= self.look_back,
            input_dim = self.layers['input'],
            output_dim= self.layers['hidden1'],
            return_sequences = True,
            ))
        self.model.add(Dropout(self.dropout))

        #add hidden layers return sequence true
        for i in range (2,self.n_hidden):
            self.model.add(LSTM(
                self.layers["hidden"+str(i)],
                return_sequences=True))
            self.model.add(Dropout(self.dropout))

        # add hidden_last return Sequences False
        self.model.add(LSTM(
            self.layers['hidden'+str(self.n_hidden)],
            return_sequences=False))
        self.model.add(Dropout(self.dropout))

        #add output
        self.model.add(Dense(
            output_dim=self.layers['output']))
        self.model.add(Activation("linear"))

        #compile model and print summary
        start = time.time()
        self.model.compile(loss=self.loss, optimizer= Adam(lr=self.learning_rate))
        logging.info("Compilation Time : %s"%str(time.time() - start))
        self.model.summary()
        return self.model


class MultiStepLSTM(object):
    def __init__(self, look_back, look_ahead, layers, dropout, loss, learning_rate):
        self.look_back = look_back
        self.look_ahead = look_ahead
        self.n_hidden = len(layers) - 2
        self.model = Sequential()
        self.layers = layers
        self.loss = loss
        #self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.dropout = dropout
        logging.info("MultiStepLSTM LSTM Model Info: %s" % (locals()))

    def build_model(self):
        # first add input to hidden1
        self.model.add(LSTM(
            input_length=self.look_back,
            input_dim=self.layers['input'],
            output_dim=self.layers['hidden1'],
            return_sequences= True if self.n_hidden>1 else False))
        self.model.add(Dropout(self.dropout))

        #add hidden layers
        for i in range(2,self.n_hidden+1):
            return_sequences = True
            if i == self.n_hidden:
                return_sequences = False
            self.model.add(LSTM(self.layers["hidden"+str(i)],return_sequences=return_sequences))
            self.model.add(Dropout(self.dropout))

        #add dense layer with output dimension to get output for one time_step
        self.model.add(Dense(output_dim=self.layers['output']))

        #Repeat for look_ahead steps to get outputs for look_ahead timesteps.
        self.model.add(RepeatVector(self.look_ahead))

        #add activation
        self.model.add(Activation("linear"))

        # compile model and print summary
        start = time.time()
        self.model.compile(loss=self.loss, optimizer= Adam(lr=self.learning_rate))
        logging.info("Compilation Time : %s" % str(time.time() - start))
        self.model.summary()
        return self.model

class StatefulMultiStepLSTM(object):
    def __init__(self,batch_size, look_back, look_ahead, layers, dropout, loss, learning_rate):
        self.batch_size = batch_size
        self.look_back = look_back
        self.look_ahead = look_ahead
        self.n_hidden = len(layers) - 2
        self.model = Sequential()
        self.layers = layers
        self.loss = loss
        # self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.dropout = dropout
        logging.info("StatefulMultiStepLSTM LSTM Model Info: %s" % (locals()))

    def build_model(self):
        # first add input to hidden1
        self.model.add(LSTM(
            units=self.layers['hidden1'],
            batch_input_shape=(self.batch_size,self.look_back,self.layers['input']),
            #batch_size=self.batch_size,
            stateful=True,
            unroll=True,
            return_sequences=True if self.n_hidden > 1 else False))
        self.model.add(Dropout(self.dropout))

        # add hidden layers
        for i in range(2, self.n_hidden + 1):
            return_sequences = True
            if i == self.n_hidden:
                return_sequences = False
            self.model.add(LSTM(units = self.layers["hidden" + str(i)], stateful=True,return_sequences=return_sequences,unroll=True))
            self.model.add(Dropout(self.dropout))

        # add dense layer with output dimension to get output for one time_step
        self.model.add(Dense(units=self.layers['output']))

        # Repeat for look_ahead steps to get outputs for look_ahead timesteps.
        self.model.add(RepeatVector(self.look_ahead))

        # add activation
        self.model.add(Activation("linear"))

        # compile model and print summary
        start = time.time()
        self.model.compile(loss=self.loss, optimizer=Adam(lr=self.learning_rate,decay= .99))
        #self.model.compile(loss=self.loss, optimizer=Adam(lr=self.learning_rate))
        logging.info("Compilation Time : %s" % str(time.time() - start))
        self.model.summary()
        return self.model


def train_stateful_model(model, x_train, y_train, batch_size, epochs, shuffle, validation, validation_data, patience):
    logging.info("Training...")
    training_start_time = time.time()
    if validation:
        early_stopping = EarlyStopping(monitor='val_loss', patience=patience)

        history_callback = model.fit(x_train, y_train,batch_size=batch_size, epochs=epochs, validation_data = validation_data,
                                     shuffle=shuffle, verbose=2, callbacks=[ResetStatesCallback(),early_stopping])
    else:
        history_callback = model.fit(x_train, y_train, batch_size=batch_size, epochs= epochs,callbacks=[ResetStatesCallback()],
                                     shuffle=shuffle, verbose=2)
    logging.info('Training duration (s) : %s', str(time.time() - training_start_time))
    logging.info("Training Loss per epoch: %s" % str(history_callback.history["loss"]))
    if validation:
        logging.info("Validation  Loss per epoch: %s" % str(history_callback.history["val_loss"]))
    print(history_callback.history.keys())
    return history_callback
    # for epoch in range(epochs):
    #     model.fit(x_train, y_train, batch_size=batch_size, epochs=1,shuffle=shuffle, verbose=2)
    #     model.reset_states()


def train_model(model, x_train, y_train, batch_size, epochs, shuffle, validation, validation_data, patience):
    logging.info("Training...")
    training_start_time = time.time()
    if validation:
        early_stopping = EarlyStopping(monitor='val_loss', patience=patience)

        history_callback = model.fit(x_train, y_train,batch_size=batch_size, epochs=epochs, validation_data = validation_data,
                                     shuffle=shuffle, verbose=2, callbacks=[early_stopping])
    else:
        history_callback = model.fit(x_train, y_train, batch_size=batch_size, epochs= epochs,shuffle=shuffle, verbose=2)
    logging.info('Training duration (s) : %s', str(time.time() - training_start_time))
    logging.info("Training Loss per epoch: %s" % str(history_callback.history["loss"]))
    if validation:
        logging.info("Validation  Loss per epoch: %s" % str(history_callback.history["val_loss"]))
    print(history_callback.history.keys())
    return history_callback
    # for epoch in range(epochs):
    #     model.fit(x_train, y_train, batch_size=batch_size, epochs=1,shuffle=shuffle, verbose=2)
    #     model.reset_states()
