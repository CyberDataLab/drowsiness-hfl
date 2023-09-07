from itertools import combinations
import random

import os
import math

# Make TensorFlow logs less verbose
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import flwr as fl
import tensorflow as tf

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

from keras.models import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization
from tensorflow.keras.optimizers import Adadelta

from typing import Dict

from flwr.common.logger import log
from logging import INFO
from csv import writer


from numpy.random import seed
from tensorflow.keras.utils import set_random_seed

from tensorflow.keras.metrics import Precision, Recall, TrueNegatives, TruePositives, FalsePositives, FalseNegatives


''' UTILS '''
ALL_IDS = list(range(1,21+1))

DRIVERS_IDS
    = {
    1: [3,4,5,12,  2,8,  1],
    2: [13,14,    6,7,10,11,16],
    3: [15,18,  9,17,20,21,  19]
}


# Parameters:
# size: number of subjects that must be present in the generated combinations
# Returns:
# List of lists, where each list corresponds to a different combination of n_unknown size
def generate_combinations(size):
    return list(combinations(ALL_IDS, size))


# Parameters:
# comb: list of drivers from the combination
# Returns:
# True: if there is at least one driver from each company
# False: other case
def one_per_company(comb):
    conds = {
        1: False,
        2: False,
        3: False
    }

    for cid in comb:
        for empid in DRIVERS_IDS:
            if (cid in DRIVERS_IDS[empid]):
                conds[empid] = True

    return conds[1] and conds[2] and conds[3]


def subset_valid_combinations(combs, n):
    selected = []
    idx_checked = []
    n_selected = 0

    seed(123)

    while (n_selected < n):
        idx = int(np.random.randint(0, len(combs), 1))

        if (idx not in idx_checked):
            comb = combs[idx]

            if (one_per_company(comb)):
                selected.append(comb)
                n_selected += 1

        idx_checked.append(idx)

    return selected


''' FL '''
class FlowerClient(fl.client.NumPyClient):
    def __init__(self, cid, model, x_train, y_train, x_val, y_val) -> None:
        self.cid = cid
        self.model = model
        self.x_train, self.y_train = x_train, y_train
        self.x_val, self.y_val = x_val, y_val

    def get_parameters(self):
        return self.model.get_weights()

    def fit(self, parameters, config):
        self.model.set_weights(parameters)
        
        seed(1)
        set_random_seed(2)
        
        self.model.fit(self.x_train, self.y_train,
                       epochs=int(config["epochs"]),
                       batch_size=32,
                       verbose=0)
        
        return self.model.get_weights(), len(self.x_train), {}

    def evaluate(self, parameters, config):
        self.model.set_weights(parameters)
        
        loss, acc = self.model.evaluate(self.x_val, self.y_val, verbose=0)
        
        return loss, len(self.x_val), {"accuracy": acc, "client": self.cid}
    
''' FUNCTION TO LOAD DATA IN A PARTICULAR CLIENT'''
def prepare_model_data(client_file):
    df = pd.read_csv(client_file)
    
    train, test = train_test_split(df, test_size=0.30, random_state=42)
    
    X_train = train[['psd_delta', 'psd_theta', 'psd_alpha', 'psd_beta', 'psd_gamma','eog_blinks', 'eog_var']]
    X_test = test[['psd_delta', 'psd_theta', 'psd_alpha', 'psd_beta', 'psd_gamma','eog_blinks', 'eog_var']]
    y_train = train['y_class']
    y_test = test['y_class']
    
    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    return X_train, X_test, y_train, y_test

''' FUNCTION TO LOAD DATA FROM A COMPANY '''
def get_data_company(company):
    base_path = "./data/horizontal_v3"
    # Load and process data from all its clients
    clients = os.listdir(f"{base_path}/{company}/")
    
    for client_id in UNSEEN_CLIENTS:
        try:
            clients.remove(f'client_{client_id}.csv')
        except:
            pass
    
    X_train, X_test, y_train, y_test = prepare_model_data(f'{base_path}/{company}/{clients[0]}')
    
    # Must work even when only exists clients[0]
    for file in clients[1:]:
        path = f'{base_path}/{company}/{file}'
        X_train_act, X_test_act, y_train_act, y_test_act = prepare_model_data(path)

        X_train = np.vstack((X_train, X_train_act))
        X_test = np.vstack((X_test, X_test_act))
        y_train = np.concatenate((y_train, y_train_act))
        y_test = np.concatenate((y_test, y_test_act))
        
    return X_train, X_test, y_train, y_test

def client_fn(cid: str) -> fl.client.Client:
    # Model best hyperparameters (See notebook Milestone0-Optimization-Baseline)
    neurons = 36
    activation = "relu"
    learning_rate = 0.180165
    optimizer = Adadelta(learning_rate=learning_rate)
    
    input_shape = (7,)
    
    # Create model
    model = Sequential()
    
    model.add(Dense(neurons, input_shape=input_shape, activation=activation))
    
    model.add(BatchNormalization())
        
    model.add(Dense(neurons, activation=activation))
    model.add(Dense(neurons, activation=activation))
    model.add(Dense(neurons, activation=activation))
    
    model.add(Dense(1, activation='sigmoid'))
    
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    # Load data partition    
    x_train_cid, x_val_cid, y_train_cid, y_val_cid = get_data_company(cid)

    # Create and return client
    return FlowerClient(cid, model, x_train_cid, y_train_cid, x_val_cid, y_val_cid)

class SaveModelStrategy(fl.server.strategy.FedAvg):
    def aggregate_fit(self, rnd, results, failures):
        aggregated_weights = super().aggregate_fit(rnd, results, failures)
        
        if aggregated_weights is not None:
            # Save aggregated_weights
            np.savez(f"./tmp/milestone2-round-{rnd}-weights.npz", aggregated_weights)
            
        return aggregated_weights

    def aggregate_evaluate(self, rnd, results, failures):
        super_result = super().aggregate_evaluate(rnd, results, failures)
        
        data = {}
        for r in results:
            acc = r[1].metrics["accuracy"]
            client = r[1].metrics["client"]
            data[client] = acc
        
        df = pd.DataFrame(data, index=[0], columns=sorted(data.keys()))
        df.to_csv(f"./results/milestone2.csv", mode='a', index=False, header=False)
        
        return super_result
    
seed(1)
set_random_seed(2)

neurons = 36
activation = "relu"
learning_rate = 0.180165
optimizer = Adadelta(learning_rate=learning_rate)

input_shape = (7,)

# Create model
model = Sequential()

model.add(Dense(neurons, input_shape=input_shape, activation=activation))

model.add(BatchNormalization())

model.add(Dense(neurons, activation=activation))
model.add(Dense(neurons, activation=activation))
model.add(Dense(neurons, activation=activation))

model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

# Get model weights as a list of NumPy ndarray's
weights = model.get_weights()
# Serialize ndarrays to `Parameters`
parameters = fl.common.weights_to_parameters(weights)

# Parameter configuration for training from the server
def fit_config(rnd: int) -> Dict[str, str]:
    config = {
        "round": str(rnd),
        "epochs": str(1) 
    }
    return config

''' Experimentation '''
def load_dataset_several_clients(clients):
    base_path = "./data/centralized"
    
    X_train, X_val, y_train, y_val = prepare_model_data(f'{base_path}/client_{clients[0]}.csv')
    
    for cid in clients[1:]:
        path = f'{base_path}/client_{cid}.csv'
        X_train_act, X_val_act, y_train_act, y_val_act = prepare_model_data(path)
    
        X_train = np.vstack((X_train, X_train_act))
        X_val = np.vstack((X_val, X_val_act))
        y_train = np.concatenate((y_train, y_train_act))
        y_val = np.concatenate((y_val, y_val_act))
        
    return X_train, X_val, y_train, y_val

def setup_model(weights_path):
    neurons = 36
    activation = "relu"
    learning_rate = 0.180165
    optimizer = Adadelta(learning_rate=learning_rate)

    input_shape = (7,)

    # Create model
    model = Sequential()

    model.add(Dense(neurons, input_shape=input_shape, activation=activation))

    model.add(BatchNormalization())

    model.add(Dense(neurons, activation=activation))
    model.add(Dense(neurons, activation=activation))
    model.add(Dense(neurons, activation=activation))

    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy', TruePositives(), TrueNegatives(), FalsePositives(), FalseNegatives()])
    
    a = np.load(weights_path, allow_pickle=True)

    n_elems = [252, 36,
               36, 36, 36, 36,
               1296, 36,
               1296, 36,
               1296, 36,
               36, 1]

    weights = []

    # https://numpy.org/devdocs/reference/generated/numpy.lib.format.html#version-numbering
    # Based on the doc, we know that the relevant data is at the end, and as there should be float32 => 4 bytes
    # we take the n_elementsz*4bytes from the end

    for i, t in enumerate(a["arr_0"][0].tensors):
        act = np.frombuffer(t[-n_elems[i]*4:], dtype=np.float32)
        weights.append(act)

    # Change the shape to adapt to the shape of the model
    weights[0] = weights[0].reshape(7,36)
    weights[6] = weights[6].reshape(36,36)
    weights[8] = weights[8].reshape(36,36)
    weights[10] = weights[10].reshape(36,36)
    weights[12] = weights[12].reshape(36,1)

    model.set_weights(weights)
    
    return model

def evaluate_FLmodel(KCs, UCs, rnd):
    path = f"./tmp/milestone2-round-{rnd}-weights.npz"
    model = setup_model(path)
    
    # Evaluate all known clients
    X_train, X_val, y_train, y_val = load_dataset_several_clients(KCs)
    res = model.evaluate(X_val, y_val, verbose=0)
    
    tp_k = res[2]
    tn_k = res[3]
    fp_k = res[4]
    fn_k = res[5]
    
    k_acc = (tp_k+tn_k)/(tp_k+tn_k+fp_k+fn_k)
    k_sens = (tp_k)/(tp_k+fn_k)
    k_spec = (tn_k)/(tn_k+fp_k)
    k_f1 = (tp_k)/( tp_k + (fp_k+fn_k)/2 )
    
    # Evaluate all knew clients
    X_train, X_val, y_train, y_val = load_dataset_several_clients(UCs)
    res_c = model.evaluate(X_val, y_val, verbose=0)
    
    tp_u = res_c[2]
    tn_u = res_c[3]
    fp_u = res_c[4]
    fn_u = res_c[5]
    
    u_acc = (tp_u+tn_u)/(tp_u+tn_u+fp_u+fn_u)
    u_sens = (tp_u)/(tp_u+fn_u)
    u_spec = (tn_u)/(tn_u+fp_u)
    u_f1 = (tp_u)/( tp_u + (fp_u+fn_u)/2 )
    
    return k_acc, k_sens, k_spec, k_f1, u_acc, u_sens, u_spec, u_f1

CLIENTS_IDS = ["company_1", "company_2", "company_3"]

# From 1 to 14
for n_unknown in range(4,14+1):
    combs = generate_combinations(21-n_unknown)
    if (n_unknown > 1):
        combs = subset_known_combinations(combs, 30)
    
    # Create the file if it didn't exist
    if not os.path.exists(f"./results/milestone2_{n_unknown}UCs.csv"):
        results_fed = pd.DataFrame(columns=["UCs", "k_acc", "k_sens", "k_spec", "k_f1", "u_acc", "u_sens", "u_spec", "u_f1"])
        results_fed.to_csv(f"./results/milestone2_{n_unknown}UCs.csv", mode='w', index=False, header=True)
    
    # Check if there are registries
    with open(f"./results/milestone2_{n_unknown}UCs.csv", 'r') as f:
        start_comb = len(f.readlines())-1 # substract header
    
    # Start/continue the n_unknown experiment
    for comb in combs[start_comb:]:
        if (one_per_company(comb)):
            global UNSEEN_CLIENTS
            UNSEEN_CLIENTS = list(set(ALL_IDS)-set(comb))
            
            # File to store the results of this federated training
            with open(f"./results/milestone2.csv", 'w') as f:
                csv_writer = writer(f)
                csv_writer.writerow(["company_1", "company_2", "company_3"])
            
            # FL
            fl.simulation.start_simulation(
                client_fn=client_fn,
                clients_ids=CLIENTS_IDS,
                client_resources={"num_cpus": 6},
                num_rounds=5,
                strategy=SaveModelStrategy(
                    min_available_clients = len(CLIENTS_IDS),
                    min_fit_clients = len(CLIENTS_IDS),
                    min_eval_clients = len(CLIENTS_IDS),
                    on_fit_config_fn = fit_config,
                    on_evaluate_config_fn = fit_config,
                    accept_failures=False,
                    initial_parameters=parameters
                ),
            )
            
            # Read training results
            df = pd.read_csv('./results/milestone2.csv')
            df["mean"] = df.mean(numeric_only=True, axis=1)

            best_rnd = df["mean"].idxmax()+1
            
            # Evaluate if with the weights of the best round for both known and unknown clients
            k_acc, k_sens, k_spec, k_f1, u_acc, u_sens, u_spec, u_f1 = evaluate_FLmodel(list(comb), UNSEEN_CLIENTS, best_rnd)
            
            # Store results for this training
            results_fed = pd.DataFrame(columns=["UCs", "k_acc", "k_sens", "k_spec", "k_f1", "u_acc", "u_sens", "u_spec", "u_f1"])
            fed_res = {
                "UCs": UNSEEN_CLIENTS,
                "k_acc": k_acc,
                "k_sens": k_sens,
                "k_spec": k_spec,
                "k_f1": k_f1,
                "u_acc": u_acc,
                "u_sens": u_sens,
                "u_spec": u_spec,
                "u_f1": u_f1
            }
            
            results_fed = results_fed.append(fed_res, ignore_index=True)
            results_fed.to_csv(f"./results/milestone2_{n_unknown}UCs.csv", mode='a', index=False, header=False)

# Remove all temp files created
base_dir = './tmp/'
for f in os.listdir(dir):
    os.remove(os.path.join(base_dir, f))