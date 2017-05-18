import numpy as np
from sklearn.metrics import mean_squared_error
import time
import logging
import config as cfg
from lstm import MultiStepLSTM
import matplotlib
if cfg.run_config['Xserver']== False:
    print "No X-server"
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
if cfg.run_config['Xserver']:
    from keras.utils.visualize_util import plot
import utilities as util
import plotly
import plotly.plotly as py
import plotly.graph_objs as go
plotly.tools.set_credentials_file(username='aakashsingh', api_key='iMfR7hS1dbnmJ9XB17XO')

np.random.seed(13)
logger = logging.getLogger(__name__).addHandler(logging.StreamHandler())

def get_peak_data(look_back,look_ahead,train_test_ratio = 0.7,):
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

    trainX, trainY = util.prepare_seq2seq_data(train,look_back,look_ahead)
    testX, testY = util.prepare_seq2seq_data(test,look_back,look_ahead)
    return trainX, trainY, testX, testY, scaler


def optimize_objective_function(params):
    params =  params.flatten()
    logging.info("inside objective function. params received: %s"%params)
    optimizers = ['adadelta', 'adam', 'rmsprop']
    optimizer = optimizers[int(params[1])]
    dropout = float(params[0])
    batch_size = int(params[2])
    logging.info("Using HyperParams. optimizer:%s, dropout:%f, batch_size:%d"%(optimizer,dropout,batch_size))
    n_epochs = 5
    train_test_ratio = 0.7
    look_back = 24
    look_ahead = 12
    layers={'input': 1, 'hidden1': 5, 'hidden2': 30, 'hidden3': 5, 'output': 1}
    lstm = MultiStepLSTM(look_back=24, look_ahead=12, layers=layers, dropout=dropout, loss='mse',
                         optimizer=optimizer)

    model = lstm.build_model()

    X_train, y_train, X_test, y_test, scaler = get_peak_data(look_back, look_ahead, train_test_ratio)
    history_callback = model.fit(X_train, y_train,batch_size=batch_size, nb_epoch=n_epochs, shuffle=False, verbose=2)
    test_loss = model.evaluate(X_test,y_test,batch_size=batch_size,verbose=1)
    print "test loss %s"%(test_loss)
    logging.info("test loss %s"%(test_loss))
    testPredictions = model.predict(X_test)

    # rescale
    testPredictions = scaler.inverse_transform(testPredictions)
    testY = scaler.inverse_transform(y_test)

    # extract first timestep for true values
    testY_1 = testY[:, 0]
    testY_1 = testY_1.flatten()

    # mean contains the mean value of a reading calculated at different points in time
    # diagonals contains the reading's values calculated at different points in time
    testPredictions_mean, test_diagonals = util.diagonal_means(testPredictions)

    # the top left and bottom right predictions do not contain predictions for all timesteps
    # fill the missing prediction values in diagonals. curenttly using the first predicted value for all missing timesteps
    for idx, diag in enumerate(test_diagonals):
        diag = diag.flatten()
        # missing value filled with
        test_diagonals[idx] = np.hstack((diag, np.full(look_ahead - len(diag), diag[0])))
    test_diagonals = np.asarray(test_diagonals)

    #rmse on next next timestep only
    rmse = mean_squared_error(testY_1, test_diagonals[:, 0])**0.5
    logging.info("optimizing run RMSE: %f"%(rmse))
    return rmse


def run_esr(experiment_id, model,batch_size,n_epochs,look_back,look_ahead,train_test_ratio):
    global_start_time = time.time()
    logging.info(" HYPERPRAMRAMS : %s"%(str(locals())))
    print('Loading data... ')

    X_train, y_train, X_test, y_test, scaler= get_peak_data(look_back,look_ahead,train_test_ratio)
    logging.info('\nData Loaded. Compiling...\n')

    try:
        logging.info("Training...")

        history_callback = model.fit(
                X_train, y_train,
                batch_size=batch_size, nb_epoch=n_epochs, validation_split=0.05,shuffle=False,verbose=2)

        logging.info('Training duration (s) : %s', str(time.time() - global_start_time))
        logging.info("Training Loss per epoch: %s"%str(history_callback.history["loss"]))
        logging.info("Validation  Loss per epoch: %s"%str(history_callback.history["val_loss"]))
        print(history_callback.history.keys())

        logging.info("Evaluating....")
        test_loss = model.evaluate(X_test, y_test, batch_size=batch_size, verbose=1)
        print "Test Loss %s" % (test_loss)
        logging.info("Test Loss %s" % (test_loss))

        logging.info("Predicting ...")
        trainPredictions = model.predict(X_train)
        testPredictions = model.predict(X_test)

        #rescale
        trainPredictions = scaler.inverse_transform(trainPredictions)
        trainY = scaler.inverse_transform(y_train)
        testPredictions = scaler.inverse_transform(testPredictions)
        testY = scaler.inverse_transform(y_test)

        #extract first timestep for true values
        trainY_1 = trainY[:, 0]
        testY_1 = testY[:, 0]
        testY_1 = testY_1.flatten()

        # mean contains the mean value of a reading calculated at different points in time
        # diagonals contains the reading's values calculated at different points in time
        trainPredictions_mean,train_diagonals = util.diagonal_means( trainPredictions)
        testPredictions_mean, test_diagonals = util.diagonal_means(testPredictions)

        # the top left and bottom right predictions do not contain predictions for all timesteps
        # fill the missing prediction values in diagonals. curenttly using the first predicted value for all missing timesteps
        for idx,diag in enumerate(test_diagonals):
            diag = diag.flatten()
            #missing value filled with
            test_diagonals[idx] = np.hstack((diag , np.full(look_ahead -len(diag), diag[0])))
        test_diagonals = np.asarray(test_diagonals)

        for i in range(look_ahead):
            logging.info("Test RMSE on %d timestep prediction %f" % ((i+1),mean_squared_error(testY_1,test_diagonals[:,i]) **0.5 ))

        test_shifted_1 = util.shift_time_series(testY_1,1)
        logging.info("testY_1,test_shifted_1")
        for i in range(3):
            logging.info("%f,%f" % (testY_1[i], test_shifted_1[i]))
        logging.info("RMSE Naive One Timestep Shift %f",mean_squared_error(testY_1[1:],test_shifted_1[1:]) **0.5)

        # plt.plot(testY_1, label="actual", linewidth=1)
        # plt.plot(test_shifted_1, label="prediction", linewidth=1, linestyle="--")
        # error = ((testY_1-test_shifted_1 ))
        # plt.title("Prediction on test data for using shift of 1")
        # plt.plot(error, label="error", linewidth=0.5)
        # plt.legend()
        # plt.tight_layout()
        # plt.savefig(("imgs/" + experiment_id + "_test_shift_1.png"), bbox_inches='tight', dpi=(200))
        # if cfg.run_config['Xserver']:
        #     plt.show()

    except KeyboardInterrupt:
        print("prediction exception")
        print('Training duration (s) : ', time.time() - global_start_time)
        return model, y_test, 0

    try:
        print "Plotting test results"
        x_data = np.arange(0,len(testY_1))
        trace_true = go.Scatter(
            x= x_data,
            y= testY_1,
            name = "actual"
        )

        step = 1
        if look_ahead > 5:
            step = int(look_ahead / 5)
        for idx, i in enumerate(np.arange(0, look_ahead, step)):
            plt.figure(idx)
            plt.title("Prediction on test data for t+%d timestep.  %d epochs, look back %d, look_ahead %d & batch_size %d"%(i+1,n_epochs,look_back,look_ahead,batch_size))
            plt.plot(testY_1,label = "actual",linewidth=1)
            plt.plot(test_diagonals[:,i],label = "prediction",linewidth=1,linestyle="--")
            error = ((testY_1 - test_diagonals[:,i]))
            plt.plot(error, label = "error",linewidth=0.5)
            plt.legend()
            plt.tight_layout()
            plt.savefig(("imgs/" + experiment_id + "_test_timestep_%d.png"%(i+1)), bbox_inches='tight', dpi=(200))

            trace_predicted = go.Scatter(
                x=x_data,
                y=test_diagonals[:, i],
                name="predicted_t+%d" % (i + 1)
            )
            trace_error = go.Scatter(
                x=x_data,
                y=error,
                name="error_t+%d" % (i + 1)
            )
            plotly_data = [trace_true,trace_predicted,trace_error]
            #py.plot(plotly_data, filename='lstm_timestep_%d'%(i+1))

        if cfg.run_config['Xserver']:
            plt.show()

    except Exception as e:
        logging.error("plotting exception")
        logging.error(str(e))

    # sns.distplot(trainY - trainPredictions)
    # plt.savefig("imgs/"+experiment_id + "_plot_fitting.png")
    # sns.plt.show()
    return model, testPredictions, testY


if __name__ == "__main__":
    #load config params
    FORMAT = '%(asctime)-15s. %(message)s'
    logger = logging.basicConfig(filename=cfg.run_config['log_file'],level=logging.INFO,format=FORMAT)
    experiment_id = cfg.multi_step_lstm_config['experiment_id']
    look_back = cfg.multi_step_lstm_config['look_back']
    look_ahead = cfg.multi_step_lstm_config['look_ahead']
    batch_size = cfg.multi_step_lstm_config['batch_size']
    n_epochs = cfg.multi_step_lstm_config['n_epochs']
    train_test_ratio = cfg.multi_step_lstm_config['train_test_ratio']
    dropout = cfg.multi_step_lstm_config['dropout']
    layers = cfg.multi_step_lstm_config['layers']
    loss = cfg.multi_step_lstm_config['loss']
    optimizer = cfg.multi_step_lstm_config['optimizer']

    logging.info("----------------------------------------------------")
    logging.info('Run id %s' % (experiment_id))

    lstm = MultiStepLSTM(look_back = look_back, look_ahead = look_ahead, layers=layers,dropout=dropout,loss = loss, optimizer=optimizer)

    model = lstm.build_model()
    if cfg.run_config['Xserver']:
        plot(model, to_file="imgs/seq2seq_lstm.png", show_shapes=True, show_layer_names=True)
    run_esr(experiment_id,model,batch_size,n_epochs,look_back,look_ahead,train_test_ratio)

    logging.info("-----------------------------------------------------------------------")
    logging.info("             ")

