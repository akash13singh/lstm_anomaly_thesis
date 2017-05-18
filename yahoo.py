import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error
import time

np.random.seed(13)

def run_yahoo():
    global_start_time = time.time()

    print 'Loading data... '
    #X_train, y_train, X_test, y_test,scaler = util.get_peak_data(look_back,train_test_ratio)
    X_train, y_train, X_test, y_test, scaler= util.get_yahoo_data(look_back, train_test_ratio)
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    X_test = np.reshape(X_test, (X_test .shape[0], X_test .shape[1], 1))

    print '\nData Loaded. Compiling...\n'
    model = build_model()

    try:
        print("Training...")
        model.fit(
                X_train, y_train,
                batch_size=batch_size, nb_epoch=n_epochs, validation_split=0.05)

        print 'Training duration (s) : ', time.time() - global_start_time

        print("Predicting ...")
        trainPredictions = model.predict(X_train)
        testPredictions = model.predict(X_test)
        print X_test.shape
        #rescale
        trainPredictions = scaler.inverse_transform(trainPredictions)
        trainY = scaler.inverse_transform(y_train)
        testPredictions = scaler.inverse_transform(testPredictions)
        testY = scaler.inverse_transform(y_test)
        print testPredictions
        print testY
        print "mean squared error on train %f"%(mean_squared_error(trainY,trainPredictions))
        print "mean squared error on test %f" % (mean_squared_error(testY, testPredictions))

    except KeyboardInterrupt:
        print("prediction exception")
        print 'Training duration (s) : ', time.time() - global_start_time
        return model, y_test, 0

    try:
        print "Plotting..."
        plt.figure(1)
        plt.subplot(311)
        plt.title("Performance on training data after %d epochs, look back %d" % (n_epochs, look_back))
        plt.plot(trainY, label="actual")
        plt.plot(trainPredictions, label="predicted")
        plt.legend()
        plt.subplot(313)
        plt.title("Error")
        error = ((trainY - trainPredictions))
        plt.plot(error, 'r')
        plt.subplot(312)
        plt.title("Squared Error")
        squared_error = ((trainY - trainPredictions) ** 2)
        plt.plot(squared_error, 'b')
        plt.show()

        plt.figure(1)
        plt.subplot(311)
        plt.title("Performance on test data after %d epochs, look back %d"%(n_epochs,look_back))
        plt.plot(testY, label = "actual")
        plt.plot(testPredictions,label = "predicted")
        plt.legend()
        plt.subplot(313)
        plt.title("Error")
        mse = ((testY - testPredictions))
        plt.plot(mse, 'r')
        plt.subplot(312)
        plt.title("Squared Error")
        mse = ((testY - testPredictions)**2)
        plt.plot(mse, 'b')
        plt.show()
    except Exception as e:
        print("plotting exception")
        print str(e)


    return model, testPredictions, testY