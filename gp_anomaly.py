from data_utils import generate_uniform_anomalies, read_GAS_data, iterate_mini_batches
import numpy as np
import GPy
from IPython.display import display
from autoencoder import SimpleAutoencoder
import matplotlib
import tensorflow as tf
import GPy

np.random.seed(29)
hidden_dimensions = [56,10,56]
learning_rate = 0.001
n_epochs = 40
batch_size = 512
display_step = 1
tf.set_random_seed(29)

def create_anomaly_dataset():
    X_train, X_test = read_GAS_data()
    X_test = X_test[:20,]
    Y_train = np.ones(shape= (X_train.shape[0],1))
    Y_test = np.ones(shape=(X_test.shape[0], 1))

    num_anomalies = 20
    anomalies = generate_uniform_anomalies(0,1,num_anomalies,d = X_train.shape[1])

    X_train_anomalies = anomalies[:int(num_anomalies) / 2, :]
    X_test_anomalies = anomalies[int(num_anomalies) / 2:, :]
    Y_train_anomalies = np.full((int(num_anomalies) / 2, 1), 0)
    Y_test_anomalies = np.full((int(num_anomalies) / 2, 1), 0)

    X_test = np.concatenate((X_test, X_test_anomalies), axis=0)
    Y_test = np.concatenate((Y_test, Y_test_anomalies), axis=0)

    shuffle_train,shuffle_test = np.random.permutation(X_train.shape[0]),np.random.permutation(X_test.shape[0])
    X_train, Y_train = X_train[shuffle_train, :], Y_train[shuffle_train]

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


def do_gp_regression():
    encoded_X_train = np.load("resources/files/encoded_X_train.npy")
    encoded_X_test = np.load("resources/files/encoded_X_test.npy")
    X = encoded_X_train[:1000,]
    Y = Y_train[:1000]

    # define kernel, notice the number of input dimensions is now 2
    ker = GPy.kern.RBF(input_dim=X.shape[1], ARD=True)

    # create simple GP model
    m = GPy.models.GPRegression(X, Y, ker)
    m.rbf.lengthscale.constrain_bounded(0.01, 0.1)
    m.optimize()
    X_test = encoded_X_test

    mean, variance = m.predict(X_test, include_likelihood=True)
    print m
    for idx,val in enumerate(mean):
        print "Prediction: %f    True:%d    Variance:%f"%(float(val),Y_test[idx],float(variance[idx]))
    #print variance

if __name__ == "__main__":
    X_train, Y_train, X_test, Y_test = create_anomaly_dataset()
    train_autoencoder(X_train,Y_train)
    do_gp_regression()
