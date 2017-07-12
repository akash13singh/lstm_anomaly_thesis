
run_config = { 'Xserver' : True,
               'log_file' : 'logs/run.log',
               'experiment_id' : "nab_jun22_overfitting",
               'data_folder': 'resources/data/nab/nab_machine_temperature/',
               #'data_folder': 'resources/data/discords/space_shuttle/',
               #'data_folder': 'resources/data/discords/dutch_power/',
               #'data_folder': 'resources/data/discords/ECG/',
               'save_figure': True
               }

opt_config = { 'Xserver' : True,
               'log_file' : '../logs/opt.log',
               'opt_run_id': "machine_temp_may24",
               'data_folder': '../resources/data/discords/dutch_power/',
               'save_figure': True,
               'model': 'stateful',
               'max_iter': 3,
                'initial_evals': 1
               }

multi_step_lstm_config = {  'batch_size': 256,
                            'n_epochs': 40,
                            'dropout': 0.1 ,
                            'look_back': 12,
                            'look_ahead':1,
                            #'layers':{'input': 1, 'hidden1':20, 'hidden2':5,  'output': 1},
                            #'layers':{'input': 1, 'hidden1': 200, 'hidden2': 80, 'hidden3': 40, 'hidden4': 10,'output': 1},
                            'layers': {'input': 1, 'hidden1': 120,    'output': 1},
                            'loss': 'mse',
                            #'optimizer': 'adam',
                            'train_test_ratio' : 0.7,
                            'shuffle': False,
                            'validation': True,
                            'learning_rate': .02,
                            'patience':20,
                           }
#
# multi_step_lstm_config = {  'experiment_id' : "multistep_multikpi",
#                             'batch_size' : 1024,
#                             'n_epochs' : 5,
#                             'train_test_ratio' : 0.7,
#                             'dropout' : 0.4,
#                             'look_back' : 24,
#                             'look_ahead' : 12,
#                             'layers':{'input': 1, 'hidden1': 5, 'hidden2': 30, 'hidden3': 5, 'output': 1},
#                             'loss':'mse',
#                             'optimizer':'rmsprop'
#                            }
