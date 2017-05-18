from data_utils import read_GAS_data,iterate_mini_batches
from data_utils import generate_uniform_anomalies
import numpy as np
from autoencoder import SimpleAutoencoder
import tensorflow as tf
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import confusion_matrix
from sklearn import svm
from sklearn import metrics

hidden_dimensions = [56,10,56]
learning_rate = 0.001
n_epochs = 40
batch_size = 512
display_step = 1
tf.set_random_seed(29)
np.random.seed(29)


def create_anomaly_dataset():
    X_train, X_test = read_GAS_data()
    Y_train = np.ones(shape= (X_train.shape[0],1))
    Y_test = np.ones(shape=(X_test.shape[0], 1))

    num_anomalies = int(.08 * (X_train.shape[0] + X_test.shape[0]))
    anomalies = generate_uniform_anomalies(0,1,num_anomalies,d = X_train.shape[1])

    X_train_anomalies = anomalies[:int(num_anomalies)/2,:]
    X_test_anomalies = anomalies[int(num_anomalies)/2:,:]
    Y_train_anomalies =  np.full((int(num_anomalies)/2,1),-1)
    Y_test_anomalies = np.full((int(num_anomalies)/2,1),-1)

    X_train = np.concatenate((X_train, X_train_anomalies), axis=0)
    X_test = np.concatenate((X_test, X_test_anomalies), axis=0)
    Y_train = np.concatenate((Y_train,Y_train_anomalies),axis =0)
    Y_test = np.concatenate((Y_test, Y_test_anomalies), axis=0)

    shuffle_train,shuffle_test = np.random.permutation(X_train.shape[0]),np.random.permutation(X_test.shape[0])
    X_train, Y_train = X_train[shuffle_train, :], Y_train[shuffle_train]
    X_test, Y_test = X_test[shuffle_test, :], Y_test[shuffle_test]

    return X_train,Y_train,X_test,Y_test


def train_autoencoder(X_train,Y_train):

    ae = SimpleAutoencoder(X_train.shape[1],hidden_dimensions,learning_rate)
    ae.construct_model()

    init = tf.global_variables_initializer()
    # Launch the graph
    with tf.Session() as sess:
        sess.run(init)
        print ("number of examples: %d" % (len(X_train)))
        print (" batch size: %d" % (batch_size))
        n_batches = int(len(X_train) / batch_size)
        print ("num batches: %d" % (n_batches))

        for epoch in range(n_epochs):
            # Loop over all batches
            for Xi, Yi in iterate_mini_batches(X_train, Y_train, batch_size):
                # Run optimization op (backprop) and cost op (to get loss value)
                _, c = sess.run([ae.optimizer,ae.loss], feed_dict={ae.input_placeholder: Xi})
            # Display logs per epoch step
            if epoch % display_step == 0:
                print("Epoch:", '%02d' % (epoch + 1),
                      "cost=", "{:.9f}".format(c))

        print("Optimization Finished!")

        encoded_X_train, reconstructed_X_train = sess.run([ae.fc2, ae.reconstuction], feed_dict={ae.input_placeholder: X_train})
        encoded_X_test = sess.run(ae.fc2, feed_dict={ae.input_placeholder: X_test})
        np.save("resources/files/encoded_X_train", encoded_X_train)
        np.save("resources/files/reconstructed_X_train", reconstructed_X_train)
        np.save("resources/files/encoded_X_test", encoded_X_test)

def detect_anomalies():

    encoded_X_train = np.load("resources/files/encoded_X_train.npy")
    encoded_X_test = np.load("resources/files/encoded_X_test.npy")
    print(encoded_X_train.shape)
    print(encoded_X_test.shape)

    clf = svm.OneClassSVM(nu=0.1, kernel="linear")
    clf.fit(encoded_X_train)
    y_pred_train = clf.predict(encoded_X_train)
    y_pred_test = clf.predict(encoded_X_test)
    y_pred_outliers = clf.predict(np.full((100,hidden_dimensions[1]),4))

    # print y_pred_train[y_pred_train == -1].size
    # print y_pred_test[y_pred_test == -1].size
    # print y_pred_outliers[y_pred_outliers == -1].size

    # n_normal_points_test = X_test[y_pred_test == 1]
    # n_anomalies_test = X_test[y_pred_test == -1]
    # print(n_normal_points_test.shape)
    # print(n_anomalies_test.shape)

    print("Train Accuracy: %f"%(accuracy_score(Y_train, y_pred_train)))
    print("Test Accuracy: %f"%( accuracy_score(Y_test, y_pred_test)))
    print("Precision: %f" % (precision_score(Y_test, y_pred_test,pos_label=1)))
    #print("Recall: %f" % (precision_score(Y_test, y_pred_test, pos_label=-1)))
    print "Confusion Matrix: (Anomalies, Normal)"
    print confusion_matrix(Y_test,y_pred_test,labels=[-1,1])
    fpr, tpr, thresholds = metrics.roc_curve(Y_test, y_pred_test, pos_label=1)
    print "AUC: %f"%metrics.auc(fpr, tpr)

if __name__ == "__main__":
    X_train, Y_train, X_test, Y_test = create_anomaly_dataset()
    #train_autoencoder(X_train,Y_train)
    detect_anomalies()