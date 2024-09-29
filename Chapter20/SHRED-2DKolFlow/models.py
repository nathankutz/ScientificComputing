import torch
from torch.utils.data import DataLoader
import numpy as np

class SHRED(torch.nn.Module):
    '''SHRED model accepts input size (number of sensors), output size (dimension of high-dimensional spatio-temporal state, hidden_size, number of LSTM layers,
    size of fully-connected layers, and dropout parameter'''
    def __init__(self, input_size, output_size, hidden_size=64, hidden_layers=2, l1=350, l2=400, dropout=0.0):
        super(SHRED,self).__init__()

        self.lstm = torch.nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                                 num_layers=hidden_layers, batch_first=True)
        
        self.linear1 = torch.nn.Linear(hidden_size, l1)
        self.linear2 = torch.nn.Linear(l1, l2)
        self.linear3 = torch.nn.Linear(l2, output_size)

        self.dropout = torch.nn.Dropout(dropout)

        self.hidden_layers = hidden_layers
        self.hidden_size = hidden_size

        

    def forward(self, x):
        
        h_0 = torch.zeros((self.hidden_layers, x.size(0), self.hidden_size), dtype=torch.float)
        c_0 = torch.zeros((self.hidden_layers, x.size(0), self.hidden_size), dtype=torch.float)
        if next(self.parameters()).is_cuda:
            h_0 = h_0.cuda()
            c_0 = c_0.cuda()

        _, (h_out, _) = self.lstm(x, (h_0, c_0))
        h_out = h_out[-1].view(-1, self.hidden_size)

        output = self.linear1(h_out)
        output = self.dropout(output)
        output = torch.nn.functional.relu(output)

        output = self.linear2(output)
        output = self.dropout(output)
        output = torch.nn.functional.relu(output)
    
        output = self.linear3(output)

        return output


class SDN(torch.nn.Module):
    '''SDN model accepts input size (number of sensors), output size (dimension of high-dimensional spatio-temporal state,
    size of fully-connected layers, and dropout parameter'''
    def __init__(self, input_size, output_size, l1=350, l2=400, dropout=0.0):
        super(SDN,self).__init__()
        
        self.linear1 = torch.nn.Linear(input_size, l1)
        self.linear2 = torch.nn.Linear(l1, l2)
        self.linear3 = torch.nn.Linear(l2, output_size)

        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x):

        output = self.linear1(x)
        output = self.dropout(output)
        output = torch.nn.functional.relu(output)

        output = self.linear2(output)
        output = self.dropout(output)
        output = torch.nn.functional.relu(output)
        
        output = self.linear3(output)

        return output

def fit(model, train_dataset, valid_dataset, batch_size=64, num_epochs=4000, lr=1e-3, verbose=False, patience=5):
    '''Function for training SHRED and SDN models'''
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    val_error_list = []
    patience_counter = 0
    best_params = model.state_dict()
    for epoch in range(1, num_epochs + 1):
        
        for k, data in enumerate(train_loader):
            model.train()
            outputs = model(data[0])
            optimizer.zero_grad()
            loss = criterion(outputs, data[1])
            loss.backward()
            optimizer.step()

        if epoch % 20 == 0 or epoch == 1:
            model.eval()
            with torch.no_grad():
                val_outputs = model(valid_dataset.X)
                val_error = torch.linalg.norm(val_outputs - valid_dataset.Y)
                val_error = val_error / torch.linalg.norm(valid_dataset.Y)
                val_error_list.append(val_error)

            if verbose == True:
                print('Training epoch ' + str(epoch))
                print('Error ' + str(val_error_list[-1]))

            if val_error == torch.min(torch.tensor(val_error_list)):
                patience_counter = 0
                best_params = model.state_dict()
            else:
                patience_counter += 1


            if patience_counter == patience:
                model.load_state_dict(best_params)
                return torch.tensor(val_error_list).cpu()

    model.load_state_dict(best_params)
    return torch.tensor(val_error_list).detach().cpu().numpy()


def forecast(forecaster, reconstructor, test_dataset):
    '''Takes model and corresponding test dataset, returns tensor containing the
    inputs to generate the first forecast and then all subsequent forecasts 
    throughout the test dataset.'''
    initial_in = test_dataset.X[0:1].clone()
    vals = []
    for i in range(0, test_dataset.X.shape[1]):
        vals.append(initial_in[0, i, :].detach().cpu().clone().numpy())

    for i in range(len(test_dataset.X)):
        scaled_output = forecaster(initial_in).detach().cpu().numpy()

        vals.append(scaled_output.reshape(test_dataset.X.shape[2]))
        temp = initial_in.clone()
        initial_in[0,:-1] = temp[0,1:]
        initial_in[0,-1] = torch.tensor(scaled_output)

    device = 'cuda' if next(reconstructor.parameters()).is_cuda else 'cpu'
    forecasted_vals = torch.tensor(np.array(vals), dtype=torch.float32).to(device)
    reconstructions = []
    for i in range(len(forecasted_vals) - test_dataset.X.shape[1]):
        recon = reconstructor(forecasted_vals[i:i+test_dataset.X.shape[1]].reshape(1, test_dataset.X.shape[1], 
                                    test_dataset.X.shape[2])).detach().cpu().numpy()
        reconstructions.append(recon)
    reconstructions = np.array(reconstructions)
    return forecasted_vals, reconstructions