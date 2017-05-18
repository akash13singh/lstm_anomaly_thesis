import numpy as np
from sklearn.metrics import mean_squared_error
import time
import utilities as util
import logging
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from lstm import VanillaLSTM
from keras.utils.visualize_util import plot
np.random.seed(13)

logger = logging.getLogger(__name__).addHandler(logging.StreamHandler())
logging.basicConfig(filename='logs/anomaly.log',level=logging.INFO)

def get_peak_data(look_back,train_test_ratio = 0.7,):
    dataset = np.load("resources/data/peak_unique_users_Jul-Nov_onethird_complete_days.npy")
    train_size = int(len(dataset) * train_test_ratio)
    train, test = dataset[0:train_size, :], dataset[train_size:len(dataset), :]

    train,scaler = util.standardize(train)
    logging.info("Scaler Mean: %f"%(scaler.mean_))
    logging.info("Scaler Variance: %f"%(scaler.var_))
    logging.info("Scaled Training Data Mean: %f"%(np.mean(train)))
    logging.info("Scaled Training Data Variance: %f" % (np.var(train)))

    logging.info("Trasnfrom test data using scaler...")
    test = scaler.transform(test)
    logging.info("Scaled Test Data Mean: %f" % (np.mean(test)))
    logging.info("Scaled Test Data Variance: %f" % (np.var(test)))
    trainX, trainY = util.prepare_lookback_data(train,look_back)
    testX, testY = util.prepare_lookback_data(test,look_back)
    return trainX, trainY, testX, testY, scaler


def run_esr(experiment_id,model,batch_size,n_epochs,look_back,train_test_ratio):
    global_start_time = time.time()
    logging.info(" HYPERPRAMRAMS : %s"%(str(locals())))
    print('Loading data... ')
    X_train, y_train, X_test, y_test, scaler= get_peak_data(look_back, train_test_ratio)

    logging.info('\nData Loaded. Compiling...\n')

    try:
        logging.info("Training...")

        history_callback = model.fit(
                X_train, y_train,
                batch_size=batch_size, nb_epoch=n_epochs, validation_split=0.05)

        logging.info('Training duration (s) : %s', str(time.time() - global_start_time))
        logging.info("Training Loss per epoch: %s"%str(history_callback.history["loss"]))
        logging.info("Validation  Loss per epoch: %s"%str(history_callback.history["val_loss"]))
        print(history_callback.history.keys())

        logging.info("Predicting ...")
        trainPredictions = model.predict(X_train)
        testPredictions = model.predict(X_test)
        logging.info(X_test.shape)

        #rescale
        trainPredictions = scaler.inverse_transform(trainPredictions)
        trainY = scaler.inverse_transform(y_train)
        testPredictions = scaler.inverse_transform(testPredictions)
        testY = scaler.inverse_transform(y_test)

        logging.info("testY,testPrediction")
        for i in range(20):
            logging.info("%f,%f"%(testY[i],testPredictions[i]))
        logging.info("mean squared error on train %f"%(mean_squared_error(trainY,trainPredictions)))
        logging.info("mean squared error on test %f" % (mean_squared_error(testY, testPredictions)))

        print trainY.shape
        print trainPredictions.shape
        print testY.shape
        print testPredictions.shape

    except KeyboardInterrupt:
        print("prediction exception")
        print('Training duration (s) : ', time.time() - global_start_time)
        return model, y_test, 0


    try:
        logging.info("Plotting...")
        plt.figure(1)
        plt.subplot(311)
        plt.title("Performance on training data after %d epochs, look back %d & batch_size %d" % (n_epochs, look_back,batch_size))
        plt.plot(trainY, label="actual")
        plt.plot(trainPredictions, label="predicted")
        plt.legend()
        plt.subplot(312)
        plt.title("Error")
        error = ((trainY - trainPredictions))
        plt.plot(error, 'r')
        plt.subplot(313)
        plt.title("Squared Error")
        squared_error = ((trainY - trainPredictions) ** 2)
        plt.plot(squared_error, 'b')
        plt.tight_layout()
        plt.savefig("imgs/"+experiment_id+"_train.png",bbox_inches='tight',dpi = (200))

        plt.figure(2)
        plt.subplot(311)
        plt.title("Performance on test data after %d epochs, look back %d & batch_size %d"%(n_epochs,look_back,batch_size))
        plt.plot(testY, label = "actual")
        plt.plot(testPredictions,label = "predicted")
        plt.legend()
        plt.subplot(312)
        plt.title("Error")
        mse = ((testY - testPredictions))
        plt.plot(mse, 'r')
        plt.subplot(313)
        plt.title("Squared Error")
        mse = ((testY - testPredictions)**2)
        plt.plot(mse, 'b')
        plt.tight_layout()
        plt.savefig("imgs/"+experiment_id + "_test.png", bbox_inches='tight',dpi = (200))
        plt.show()
        logging.info("Fitting Gaussian on Training Errors:")
        util.fit_gaussian((trainY - trainPredictions),experiment_id, "Train_Error")
        util.fit_gaussian((testY - testPredictions),experiment_id, "Test_Error")



    except Exception as e:
        logging.error("plotting exception")
        logging.error( str(e))

    # sns.distplot(trainY - trainPredictions)
    # plt.savefig("imgs/"+experiment_id + "_plot_fitting.png")
    # sns.plt.show()
    return model, testPredictions, testY


if __name__ == "__main__":
    experiment_id = "test2"
    look_back = 10
    batch_size = 1024
    n_epochs = 3
    train_test_ratio = 0.7
    dropout = 0.4

    #lstm = lstm.VanillaLSTM(look_back, layers={'input': 1, 'hidden1': 64, 'hidden2': 256, 'hidden3': 100, 'output': 1})
    logging.info("----------------------------------------------------")
    logging.info('Run id %s' % (experiment_id))
    lstm = VanillaLSTM(look_back, layers={'input': 1, 'hidden1': 20, 'hidden2': 40, 'hidden3': 5, 'output': 1},dropout=dropout)
    model = lstm.build_model()
    plot(model,to_file="imgs/vanilla_lstm.png",show_shapes=True,show_layer_names=True)
    run_esr(experiment_id,model,batch_size,n_epochs,look_back,train_test_ratio)
    logging.info("-----------------------------------------------------------------------")
    logging.info("             ")

