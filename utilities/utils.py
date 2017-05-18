import configuration.config as cfg
import matplotlib
if cfg.run_config['Xserver']== False:
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
from scipy.stats import norm
if cfg.run_config['Xserver']:
    import seaborn as sns
import logging
from keras.callbacks import EarlyStopping

logger = logging.getLogger(__name__).addHandler(logging.StreamHandler())

def prepare_seq2seq_data(dataset, look_back, look_ahead):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back - look_ahead):
        input_seq = dataset[i:(i + look_back)]
        output_seq = dataset[i + look_back:(i + look_back + look_ahead)]
        dataX.append(input_seq)
        dataY.append(output_seq)
    dataX = np.reshape(np.array(dataX),[-1,look_back,1])
    dataY = np.reshape(np.array(dataY),[-1,look_ahead,1])
    return dataX,dataY

#use only for training data
#use returned scaler object to scale test data
def standardize(data):
    scaler = StandardScaler()
    data = scaler.fit_transform(data)
#     print "Scaler Mean: %f" % (scaler.mean_)
#     print "Scaler Variance: %f" % (scaler.var_)
    print "data mean %f, data variance %f"%(np.mean(data),np.var(data))
    return data,scaler

#format data for lstm, by adding previous "look_back" timesteps
# def prepare_lookback_data(dataset,look_back):
# 	dataX, dataY = [], []
# 	for i in range(len(dataset)-look_back-1):
# 		a = dataset[i:(i+look_back), ]
# 		dataX.append(a)
# 		dataY.append(dataset[i + look_back, ])
# 	return np.array(dataX), np.array(dataY)
#
# def prepare_seq2seq_data(dataset, look_back, look_ahead):
#     dataX, dataY = [], []
#     for i in range(len(dataset) - look_back - look_ahead):
#         input_seq = dataset[i:(i + look_back), :]
#         output_seq = dataset[i + look_back:(i + look_back + look_ahead), :]
#         dataX.append(input_seq)
#         dataY.append(output_seq)
#     dataX = np.array(dataX)
#     dataY = np.array(dataY)
#     return dataX,dataY
#
# def get_seq2seq_data(data_file,look_back,look_ahead):
#     train = np.load(data_file+"train.npy")
#     validation = np.load(data_file+"validation.npy")
#     test = np.load(data_file + "test.npy")
#
#     train,scaler = standardize(train)
#     logging.info("Scaler Mean: %f"%(scaler.mean_))
#     logging.info("Scaler Variance: %f"%(scaler.var_))
#     logging.info("Scaled Training Data Mean: %f"%(np.mean(train)))
#     logging.info("Scaled Training Data Variance: %f" % (np.var(train)))
#
#     test = scaler.transform(test)
#     validation = scaler.transform(validation)
#     X_train, y_train = prepare_seq2seq_data(train,look_back,look_ahead)
#     X_validation, y_validation = prepare_seq2seq_data(validation,look_back,look_ahead)
#     X_test, y_test = prepare_seq2seq_data(test,look_back,look_ahead)
#     return X_train, y_train, X_validation, y_validation, X_test, y_test, scaler
#
# #use only for training data
# #use returned scaler object to scale test data
# def standardize(data):
#     scaler = StandardScaler()
#     data = scaler.fit_transform(data)
#     print "Scaler Mean: %f" % (scaler.mean_)
#     print "Scaler Variance: %f" % (scaler.var_)
#     return data,scaler

#compute diagonal mean of upper traingle of a 2d array.
# e.g
# [[1 2 3 4],
#  [2 3 4 5],
#  [3 4 5 6]
# ]
# returns [ 1, 2, 3]
# used to compute means of multi time step predictions
def diagonal_means(input):
    #fetch diagonals in a list
    diagonals = [input[::-1, :].diagonal(i) for i in range(-input.shape[0] + 1, 1)]
    means = np.array([np.mean(arr) for arr in diagonals])
    means = np.reshape(means, [means.shape[0], 1])
    return means,diagonals

#get all diagonals from a two d array
def get_diagonals(input):
    #fetch diagonals in a list
    diagonals = [input[::-1, :].diagonal(i) for i in range(-input.shape[0] + 1, 1)]
    return diagonals


# def get_yahoo_data(look_back,train_test_ratio=0.7):
#     df = pd.read_csv("../resources/data/yahoo/A1Benchmark/real_9.csv")
#     df_values = df.values
#     timeseries = df_values[:, 1]
#     is_anomaly = df_values[:, 1]
#     train_size = int(len(timeseries) * 0.7)
#     train, test = timeseries[0:train_size], timeseries[train_size:len(timeseries)]
#     train,scaler = standardize(train)
#     print "Scaler Mean: %f"%(scaler.mean_)
#     print "Scaler Variance: %f"%(scaler.var_)
#     print "Scaled Training Data Mean: %f"%(np.mean(train))
#     print "Scaled Training Data Variance: %f" % (np.var(train))
#
#     print "Trasnform test data using scaler..."
#     test = scaler.transform(test)
#     trainX, trainY = prepare_dataset(train,look_back)
#     testX, testY = prepare_dataset(test,look_back)
#     return trainX, trainY, testX, testY, scaler

def fit_gaussian(data,experiment_id,title_string):
    print "fitting gaussian"
    plt.figure()
    mu, std = norm.fit(data)
    plt.hist(data, bins=25, normed=True, alpha=0.6, color='g')
    xmin, xmax = plt.xlim()
    logging.info("max, min of data: %f %f"%(xmax,xmin))
    x = np.linspace(xmin, xmax, 100)
    pdf_fitted = norm.pdf(x, mu, std)
    plt.plot(x, pdf_fitted, 'k', linewidth=2)
    title = title_string+". Gaussian Fit results: mu = %.2f,  std = %.2f" % (mu, std)
    plt.title(title)
    plt.savefig("imgs/" + experiment_id + "_"+title_string+"_gaussian.png", bbox_inches='tight', dpi=(200))
    if cfg.run_config['Xserver']:
        plt.show()


def shift_time_series(X,shift_step):
    return np.roll(X,shift_step)


def mkdir_p(mypath):
    '''Creates a directory. equivalent to using mkdir -p on the command line'''

    from errno import EEXIST
    from os import makedirs,path

    try:
        makedirs(mypath)
    except OSError as exc: # Python >2.5
        if exc.errno == EEXIST and path.isdir(mypath):
            pass
        else: raise

def save_figure(path,filename,plt):
    mkdir_p(path)
    plt.savefig("%s/%s"%(path,filename),bbox_inches='tight', dpi=(200))



def load_data(data_folder,look_back,look_ahead):
    print('Loading data... ')
    train = np.load(data_folder + "train.npy")
    validation1 = np.load(data_folder + "validation1.npy")
    validation2 = np.load(data_folder + "validation2.npy")
    test = np.load(data_folder + "test.npy")

    # standardize data. use the training set mean and variance to transform rewst of the sets
    train, train_scaler = standardize(train[:, 0])
    validation1 = train_scaler.transform(validation1[:, 0])
    validation2_labels = validation2[:, 1]
    validation2 = train_scaler.transform(validation2[:, 0])
    test_labels = test[:, 1]
    test = train_scaler.transform(test[:, 0])

    # prepare sequence data and labels
    X_train, y_train = prepare_seq2seq_data(train, look_back, look_ahead)
    X_validation1, y_validation1 = prepare_seq2seq_data(validation1, look_back, look_ahead)
    X_validation2, y_validation2 = prepare_seq2seq_data(validation2, look_back, look_ahead)
    X_validation2_labels, y_validation2_labels = prepare_seq2seq_data(validation2_labels, look_back, look_ahead)
    X_test, y_test = prepare_seq2seq_data(test, look_back, look_ahead)
    X_test_labels, y_test_labels = prepare_seq2seq_data(test_labels, look_back, look_ahead)
    return train_scaler, X_train, y_train, X_validation1, y_validation1, X_validation2, y_validation2, y_validation2_labels, X_test, y_test, y_test_labels


if __name__ == "__main__":
    print 22
    arr = np.array([[1,2,3,4,5],[2,3,4,5,6],[3,4,5,6,7],[4,5,6,7,8],[5,6,7,8,9],[6,7,8,9,10],[7.1,8.1,9.1,10,11]])
    train_diagonals = get_diagonals(arr)
    # diagonals contains a reading's values calculated at different points in time


    # the top left and bottom right predictions do not contain predictions for all timesteps
    # fill the missing prediction values in diagonals. curenttly using the first predicted value for all missing timesteps
    for idx, diag in enumerate(train_diagonals):
        diag = diag.flatten()
        # missing value filled with the first value

        train_diagonals[idx] = np.hstack((diag, np.full(5 - len(diag), diag[0])))
    train_diagonals = np.asarray(train_diagonals)
    print train_diagonals