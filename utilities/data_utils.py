import numpy as np
from sklearn.datasets import load_svmlight_file
import arff
import pandas as pd


def iterate_mini_batches(X,y,batch_size):
    n_train = X.shape[0]
    shuffledIndex = np.random.permutation(n_train)
    shuffledX = X[shuffledIndex, :]
    shuffledy = y[shuffledIndex]

    for ndx in range(0, n_train, batch_size):
        yield shuffledX[ndx:min(ndx + batch_size, n_train)], shuffledy[ndx:min(ndx + batch_size, n_train)]


def split_training_test(X,y,train_test_ratio=0.8):
    n = X.shape[0]
    indices = np.random.permutation(n)
    training_idx, test_idx = indices[:int(n*train_test_ratio)], indices[int(n*train_test_ratio):]
    X_train,X_test = X[training_idx, :], X[test_idx, :]
    Y_train,Y_test = y[training_idx], y[test_idx]
    return X_train,Y_train,X_test,Y_test


def normalize_columns(X):
    return (X - np.min(X,axis=0))/ (np.max(X,axis=0) - np.min(X,axis=0))

def generate_uniform_anomalies(min,max,n,d):
    #return np.full((n,d),2)
    return np.random.uniform(min,max,size=(n,d))

def read_KDD_data():
    dataset = arff.load(open('resources/datasets/KDDCup99_withoutdupl_norm_1ofn.arff', 'rb'))
    data = np.array(dataset['data'])
    df = pd.DataFrame(data)
    df.iloc[:,:80].apply(lambda x: pd.to_numeric(x))
    df[[80]] = df[[80]].replace(['yes', 'no'], [1, 0])

    #divide data into normal and anomaly
    df_anomaly = df[df[80] ==1]
    df_normal = df[df[80] ==0]
    n_normal = df_normal.shape[0]
    n_anomaly = df_anomaly.shape[0]

    # normal_indices = np.random.permutation(n_normal)
    # df_normal_train, df_normal_test = df_normal_train, df_normal_test = df_normal.iloc[normal_indices[ :int(n_normal*.8)],] ,df_normal.iloc[normal_indices[ int(n_normal*.8) :],]

    #shuffle data
    df_normal = df_normal.sample(frac=1,random_state=29)
    df_anomaly = df_anomaly.sample(frac=1, random_state=29)

    #create normal training, test datasest
    df_normal_train, df_normal_test = df_normal.iloc[ :int(n_normal*.8),] ,df_normal.iloc[int(n_normal*.8) :,]

    #create anomalous training,test sets
    if n_anomaly >= .08* n_normal:
        df_anomaly_train, df_anomaly_test =  df_anomaly.iloc[ : int(.04*n_normal),], df_anomaly.iloc[ int(.04*n_normal):(int(.04*n_normal)),]
    else:
        df_anomaly_train, df_anomaly_test = df_anomaly.iloc[: int(.5 * n_anomaly),], df_anomaly.iloc[int(.5 * n_anomaly):,]

    #combine the frames to form final training and test sets
    #df_train = pd.concat([df_normal_train,df_anomaly_train])
    df_train = df_normal_train
    df_test = pd.concat([df_normal_test, df_anomaly_test])
    df_train = df_train.sample(frac=1,random_state=29)
    df_test = df_test.sample(frac=1, random_state=29)

    return df_train.iloc[:,:80],df_train.iloc[:,80:],df_test.iloc[:,:80],df_test.iloc[:,80:]


def read_GAS_data():
    X, y = load_svmlight_file("resources/datasets/gas/gas.dat")
    X = X.toarray()
    X = normalize_columns(X)
    X_train, y_train, X_test,y_test = split_training_test(X,y)
    return X_train, X_test

if __name__ == "__main__":
    print read_GAS_data()