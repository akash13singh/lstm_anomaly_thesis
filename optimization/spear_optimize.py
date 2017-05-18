from models.lstm import MultiStepLSTM
from configuration.config import *
import logging

from utilities.utils import *


FORMAT = '%(asctime)-15s. %(message)s'
logger = logging.basicConfig(filename=opt_config['log_file'], level=logging.INFO, format=FORMAT)

def optimize(dropout,optimizer,batch_size):
    # result = np.square(y - (5.1/(4*np.square(math.pi)))*np.square(x) +
    #      (5/math.pi)*x - 6) + 10*(1-(1./(8*math.pi)))*np.cos(x) + 10
    result = objective_function(dropout,optimizer,batch_size)
    result = float(result)
    print 'Result = %f' % result
    #time.sleep(np.random.randint(60))
    return result

def objective_function(dropout,optimizer,batch_size):
    dropout = dropout[0]
    optimizer = optimizer[0]
    batch_size = batch_size[0]
    opt_id = opt_config['opt_run_id']
    data_file = multi_step_lstm_config['data_file']
    look_back = multi_step_lstm_config['look_back']
    look_ahead = multi_step_lstm_config['look_ahead']
    n_epochs = multi_step_lstm_config['n_epochs']
    train_test_ratio = multi_step_lstm_config['train_test_ratio']
    layers = multi_step_lstm_config['layers']
    loss = multi_step_lstm_config['loss']
    logging.info("----------------------------------------------------")
    logging.info('Optimizing id %s' % (opt_id))
    lstm = MultiStepLSTM(look_back=look_back, look_ahead=look_ahead, layers=layers, dropout=dropout, loss=loss,
                         optimizer=optimizer)
    model = lstm.build_model()
    logging.info(" HYPERPRAMRAMS : %s" % (str(locals())))
    print('Loading data... ')
    X_train, y_train, X_test, y_test, scaler = get_seq2seq_data(data_file, look_back, look_ahead, train_test_ratio)
    logging.info('\nData Loaded. Compiling...\n')
    logging.info("Training...")
    history_callback = model.fit(
        X_train, y_train,
        batch_size=batch_size, nb_epoch=n_epochs, validation_split=0.05, shuffle=True, verbose=2)
    test_loss = model.evaluate(X_test, y_test, batch_size=batch_size, verbose=1)
    return test_loss


# Write a function like this called 'main'
def main(job_id, params):
    print 'Anything printed here will end up in the output directory for job #%d' % job_id
    print params
    FORMAT = '%(asctime)-15s. %(message)s'
    logger = logging.basicConfig(filename=opt_config['log_file'], level=logging.INFO, format=FORMAT)
    return optimize(params['dropout'], params['optimizer'], params['batch_size'])

