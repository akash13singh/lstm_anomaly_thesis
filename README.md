# Anomaly Detection for Temporal Data using LSTM.
This repository contains the code used in my master thesis on LSTM based anomaly detection for time series data. The thesis report can be downloaded from [here](http://www.diva-portal.org/smash/record.jsf?pid=diva2:1149130).

**Abstract**

We explore the use of Long short-term memory (LSTM) for anomaly detection in temporal data. Due to the challenges in obtaining labeled anomaly datasets, an unsupervised approach is employed. We train recurrent neural networks (RNNs) with LSTM units to learn the normal time series patterns and predict future values. The resulting prediction errors are modeled to give anomaly scores. We investigate different ways of maintaining LSTM state, and the effect of using a fixed number of time steps on LSTM prediction and detection performance. LSTMs are also compared to feed-forward neural networks with fixed size time windows over inputs. Our experiments, with three real-world datasets, show that while LSTM RNNs are suitable for general purpose time series modeling and anomaly detection, maintaining LSTM state is crucial for getting desired results. Moreover, LSTMs may not be required at all for simple time series.

## Requirements
1. Keras 2.0.3
2. TensorFlow 1.0.0
3. sickit-learn 0.18.2
4. GPyOpt 1.0.3. (only required for hyper-parameter tuning)



## Configuration: 
First set the configuration settings in configuration/config.py. This file has different configuration settings.

        1. Use `run_config` to set parameters for the program execution
         like data folder, log_file etc.
            * Xserver: denotes if the machine has a display environment. Set it to false when
                       running on remote machines with no display. Else the plotting comamnds will result in an error.
            * experiment_id: a id used to identify different runs for example in the logs. The result plots are saved in folder: imgs/<experiment_id>.

        2. Use `opt_congfig` to set parameters for optimization runs. Refer:
           [1](https://github.com/SheffieldML/GPyOpt) ,[2](http://pythonhosted.org/GPyOpt/)

        3. `multi_step_lstm_config`: contains parameters specific to LSTM network
        
For hyper-parameters used refer to table 4.1 in thesis reports.

## How to Execute:
1. **Data Pre-processing**: The LSTM network needs data formatted such that each input
 sample has *look_back* number of data points and each output sample has *look_ahead* number
 of time-steps. To convert the data into appropriate format and create train, test,  and validation datasets python notebooks have been used.
 We provide notebooks for the three datasets used in the thesis which can be used as examples for new datasets.
 The three notebooks along with the processed dataset files are:
    1. ECG: notebooks/discords_ECG.ipynb, resources/data/discords/ECG/
    2. power_consumption: notebooks/discords_power_consumption.ipynb, resources/data/discords/dutch_power/
    3. machine_temperature: notebooks/NAB_machine_temp.ipynb, resources/data/nab/nab_machine_temperature/
    
 This step is done in "Part 1" of corresponding notebook.

2. **Prediction Modeling**: The main LSTM models used are in the file models/lstm.py. For training the model and generating predictions two main files
    are provided:
     1. *lstm_predictor.py*: This file uses the default LSTM implementation by keras.
     2. *stateful_lstm_predictor.py*: uses the stateful LSTM implementation

    Once the configuration setting `data_folder` has been set correctly, the code will look for train, test, and validation sets in those files.


3. **Anomaly Detection**:
Running the LSTM models which generate the predictions for train, test, and validation sets. For anomaly detection we need to calculate prediction errors or residuals,
model them using Gaussian distribution and then set thresholds. This is done in "Part 3" of the corresponding notebook files.

