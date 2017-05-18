
run_config = { 'Xserver' : False,
               'log_file' : 'logs/run.log',
               'experiment_id' : "discord_power_may18",
               #'data_folder': 'resources/data/nab/nab_machine_temperature/',
               'data_folder': 'resources/data/discords/dutch_power/',
               'save_figure': True
               }

opt_config = { 'Xserver' : False,
               'log_file' : '../logs/opt.log',
               'opt_run_id': "stateful_power_may18",
               'data_folder': '../resources/data/discords/dutch_power/',
               'save_figure': False,
               'model': 'stateful',
               'max_iter': 10,
               'initial_evals': 3
               }

multi_step_lstm_config = {  'batch_size': 672,
                            'n_epochs': 50,
                            'dropout': 0.3,
                            'look_back': 1,
                            'look_ahead' : 1,
                            'layers':{'input': 1, 'hidden1':300, 'hidden2':50,  'output': 1},
                            #'layers':{'input': 1, 'hidden1': 200, 'hidden2': 80, 'hidden3': 40, 'hidden4': 10,'output': 1},
                            #'layers': {'input': 1, 'hidden1': 200,  'output': 1},
                            'loss': 'mse',
                            #'optimizer': 'adam',
                            'train_test_ratio' : 0.7,
                            'shuffle': False,
                            'validation': True,
                            'learning_rate': .001,
                            'patience': 2,
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
