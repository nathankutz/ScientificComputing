import argparse
import torch
import numpy as np
import models
from processdata import TimeSeriesDataset
from processdata import qr_place
from processdata import load_data
from sklearn.preprocessing import MinMaxScaler
import os

parser = argparse.ArgumentParser(description='Out of sample forecasting with SHRED')

parser.add_argument('--dataset', type=str, default='SST', help='Dataset for reconstruction/forecasting')

parser.add_argument('--num_sensors', type=int, default=10, help='Number of sensors to use')

parser.add_argument('--placement', type=str, default='QR', help='Placement of sensors (random or QR)')

parser.add_argument('--epochs', type=int, default=1000, help='Maximum number of epochs')

parser.add_argument('--val_length', type=int, default=20, help='Length of validation set (Training set of 0.85*N, test set remainder)')

parser.add_argument('--lags', type=int, default=52, help='Length of sensor trajectories used')

parser.add_argument('--dest', type=str, default='', help='Destination folder')

args = parser.parse_args()
lags = args.lags
num_sensors = args.num_sensors

load_X = load_data(args.dataset)
n = load_X.shape[0]
m = load_X.shape[1]

### Select indices for training, validation, and testing
train_indices = np.arange(0, int(n*0.85))
valid_indices = np.arange(int(n*0.85), int(n*0.85) + args.val_length)
test_indices = np.arange(int(n*0.85) + args.val_length, n - lags)

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

### no -1 so output is one $\Delta t$ ahead of the final sensor measurements
sensor_train_data_out = torch.tensor(transformed_X[train_indices + lags][:, sensor_locations], dtype=torch.float32).to(device)
sensor_valid_data_out = torch.tensor(transformed_X[valid_indices + lags][:, sensor_locations], dtype=torch.float32).to(device)
sensor_test_data_out = torch.tensor(transformed_X[test_indices + lags][:, sensor_locations], dtype=torch.float32).to(device)

sensor_train_dataset = TimeSeriesDataset(train_data_in, sensor_train_data_out)
sensor_valid_dataset = TimeSeriesDataset(valid_data_in, sensor_valid_data_out)
sensor_test_dataset = TimeSeriesDataset(test_data_in, sensor_test_data_out)

sensor_forecaster = models.SHRED(num_sensors, num_sensors, hidden_size=32, hidden_layers=2, l1=100, l2=150, dropout=0.1).to(device)

sensor_val_errors = models.fit(sensor_forecaster, sensor_train_dataset, sensor_valid_dataset, batch_size=64, num_epochs=args.epochs, 
                                         verbose=True, lr=1e-3, patience=5)


### Train network for reconstruction
shred = models.SHRED(num_sensors, m, hidden_size=64, hidden_layers=2, l1=350, l2=400, dropout=0.1).to(device)
validation_errors = models.fit(shred, train_dataset, valid_dataset, batch_size=64, num_epochs=args.epochs, lr=1e-3, verbose=True, patience=5)

### Generate forecast from the trained LSTM forecaster and reconstructor
initialization = sensor_test_dataset.X[0:1].clone()
forecasted_sensors, forecasted_reconstructions = models.forecast(sensor_forecaster, shred, sensor_test_dataset)

### Generate forecasted high-dimensional fields from SHRED model
forecasted_long_sensors = np.zeros((forecasted_sensors.shape[0], m))
for i in range(len(forecasted_long_sensors)):
    forecasted_long_sensors[i, sensor_locations] = forecasted_sensors[i].detach().cpu().numpy()

scaled_forecast = sc.inverse_transform(forecasted_reconstructions.reshape(-1,m))
truths = np.zeros_like(scaled_forecast)
for i in range(len(forecasted_reconstructions)):
    truth = sc.inverse_transform(test_dataset.Y[i:i+1].detach().cpu().numpy())
    truths[i] = truth.reshape(scaled_forecast.shape[1])

### Generate forecasted high-dimensional fields from QR/POD
scaled_forecasted_sensors = sc.inverse_transform(forecasted_long_sensors)[:, sensor_locations]
C = np.zeros((num_sensors, m))
for i in range(num_sensors):
    C[i, sensor_locations[i]] = 1

qrpod_recons = (U_r @ np.linalg.inv(C @ U_r) @ scaled_forecasted_sensors[-len(scaled_forecast):].T).T

### Save reconstructions
if not os.path.exists('ForecastingResults/' + args.dest):
    os.makedirs('ForecastingResults/' + args.dest)
np.save('ForecastingResults/' + args.dest + '/reconstructions.npy', scaled_forecast)
np.save('ForecastingResults/' + args.dest + '/qrpodreconstructions.npy', qrpod_recons)
np.save('ForecastingResults/' + args.dest + '/truth.npy', truths)
