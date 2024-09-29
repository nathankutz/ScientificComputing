import argparse
import torch
import numpy as np
import models
from processdata import TimeSeriesDataset
from processdata import qr_place
from processdata import load_data
from sklearn.preprocessing import MinMaxScaler
import os

parser = argparse.ArgumentParser(description='In sample reconstructing with SHRED')

parser.add_argument('--dataset', type=str, default='SST', help='Dataset for reconstruction/forecasting')

parser.add_argument('--num_sensors', type=int, default=10, help='Number of sensors to use')

parser.add_argument('--placement', type=str, default='QR', help='Placement of sensors (random or QR)')

parser.add_argument('--epochs', type=int, default=1000, help='Maximum number of epochs')

parser.add_argument('--lags', type=int, default=52, help='Length of sensor trajectories used')

parser.add_argument('--dest', type=str, default='', help='Destination folder')

args = parser.parse_args()
lags = args.lags
num_sensors = args.num_sensors

load_X = load_data(args.dataset)
n = load_X.shape[0]
m = load_X.shape[1]

### Select indices for training, validation, and testing
train_indices = np.random.choice(n - lags, size=1000, replace=False)
mask = np.ones(n - lags)
mask[train_indices] = 0
valid_test_indices = np.arange(0, n - lags)[np.where(mask!=0)[0]]
valid_indices = valid_test_indices[::2]
test_indices = valid_test_indices[1::2]

### Set sensors randomly or according to QR
if args.placement == 'QR':
    sensor_locations, U_r = qr_place(load_X[train_indices].T, num_sensors)
else:
    _, U_r = qr_place(load_X[train_indices].T, num_sensors)
    sensor_locations = np.random.choice(m, size=num_sensors, replace=False)

### Fit min max scaler to training data, and then scale all data
sc = MinMaxScaler()
sc = sc.fit(load_X[train_indices])
transformed_X = sc.transform(load_X)

### Generate input sequences to a SHRED model
all_data_in = np.zeros((n - lags, lags, num_sensors))
for i in range(len(all_data_in)):
    all_data_in[i] = transformed_X[i:i+lags, sensor_locations]

### Generate training validation and test datasets both for reconstruction of states and forecasting sensors
device = 'cuda' if torch.cuda.is_available() else 'cpu'

train_data_in = torch.tensor(all_data_in[train_indices], dtype=torch.float32).to(device)
valid_data_in = torch.tensor(all_data_in[valid_indices], dtype=torch.float32).to(device)
test_data_in = torch.tensor(all_data_in[test_indices], dtype=torch.float32).to(device)

### -1 to have output be at the same time as final sensor measurements
train_data_out = torch.tensor(transformed_X[train_indices + lags - 1], dtype=torch.float32).to(device)
valid_data_out = torch.tensor(transformed_X[valid_indices + lags - 1], dtype=torch.float32).to(device)
test_data_out = torch.tensor(transformed_X[test_indices + lags - 1], dtype=torch.float32).to(device)

train_dataset = TimeSeriesDataset(train_data_in, train_data_out)
valid_dataset = TimeSeriesDataset(valid_data_in, valid_data_out)
test_dataset = TimeSeriesDataset(test_data_in, test_data_out)

train_dataset_sdn = TimeSeriesDataset(train_data_in[:,-1,:], train_data_out)
valid_dataset_sdn = TimeSeriesDataset(valid_data_in[:,-1,:], valid_data_out)
test_dataset_sdn = TimeSeriesDataset(test_data_in[:,-1,:], test_data_out)


### Train SHRED network for reconstruction
shred = models.SHRED(num_sensors, m, hidden_size=64, hidden_layers=2, l1=350, l2=400, dropout=0.0).to(device)
validation_errors = models.fit(shred, train_dataset, valid_dataset, batch_size=64, num_epochs=args.epochs, lr=1e-3, verbose=True, patience=5)

### Train SDN network for reconstruction
sdn = models.SDN(num_sensors, m, l1=350, l2=400, dropout=0.0).to(device)
validation_errors_sdn = models.fit(sdn, train_dataset_sdn, valid_dataset_sdn, batch_size=64, num_epochs=args.epochs, lr=1e-3, verbose=True, patience=5)

### Generate reconstructions from SHRED and SDN
test_recons = sc.inverse_transform(shred(test_dataset.X).detach().cpu().numpy())
test_recons_sdn = sc.inverse_transform(sdn(test_dataset_sdn.X).detach().cpu().numpy())

test_ground_truth = sc.inverse_transform(test_dataset.Y.detach().cpu().numpy())

### Generate reconstructions from QR/POD
qrpod_sensors = load_X[test_indices][:, sensor_locations]
C = np.zeros((num_sensors, m))
for i in range(num_sensors):
    C[i, sensor_locations[i]] = 1

qrpod_recons = (U_r @ np.linalg.inv(C @ U_r) @ qrpod_sensors.T).T

### Plot and save error
if not os.path.exists('ReconstructingResults/' + args.dest):
    os.makedirs('ReconstructingResults/' + args.dest)
np.save('ReconstructingResults/' + args.dest + '/reconstructions.npy', test_recons)
np.save('ReconstructingResults/' + args.dest + '/sdnreconstructions.npy', test_recons_sdn)
np.save('ReconstructingResults/' + args.dest + '/qrpodreconstructions.npy', qrpod_recons)
np.save('ReconstructingResults/' + args.dest + '/truth.npy', test_ground_truth)
