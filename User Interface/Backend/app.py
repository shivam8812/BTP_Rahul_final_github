from flask import Flask,Response,request
from flask_cors import CORS
from datetime import datetime, timedelta
import urllib.request
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import os
import time
app = Flask(__name__)
CORS(app)
maxind='7'
def fun(data):
    counter='8'
    today = (datetime.today() - timedelta(days=3)).strftime('%Y%m%d')
    date15 = (datetime.today() - timedelta(days=17)).strftime('%Y%m%d')

    # print(date15)
    # print(today)
    url = "https://power.larc.nasa.gov/api/temporal/daily/point?parameters=T2M_MAX,T2M_MIN,PRECTOTCORR,RH2M,PS,WS50M_MAX,WS50M_MIN&community=RE&longitude="+data['lon']+"&latitude="+data['lat']+"&start="+date15+"&end="+today+"&format=CSV"
    # print(url)
    urllib.request.urlretrieve(url,'climate'+counter+'.csv')
    filepath= r'C:/Users/Shiva/Desktop/BTP backend/climate'+counter+'.csv'
    os.chmod(filepath, 0o666)
    df = pd.read_csv(filepath,skiprows = 15)
    df['YEAR'] = df.YEAR.astype(str)
    df['MO'] = df.MO.astype(str)
    df['DY'] = df.DY.astype(str)

    df['date'] = df['YEAR'].str.cat(df['MO'],sep = '/')
    df['DATE'] = df['date'].str.cat(df['DY'],sep = '/')
    df.drop(columns = ['YEAR','MO','DY','date'],axis=1,inplace=True)
    df.set_index(['DATE'],inplace=True)
    df.to_csv('climate_final.csv')
    data1 = torch.tensor(df.values, dtype=torch.float32)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    class WeatherDataset(Dataset):
        def __init__(self, csv_file, seq_length=15, target_length=7):
            self.seq_length = seq_length
            self.target_length = target_length

            # Read the CSV file into a pandas dataframe
            self.data = pd.read_csv(csv_file)

            # Extract the weather parameters into a numpy array
            self.data = self.data.iloc[:, 1:].values

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            # Get the input sequence and target sequence
            input_seq = self.data[idx:idx+self.seq_length]
            # target_seq = self.data[idx+self.seq_length:idx+self.seq_length+self.target_length]

            # Convert the sequences into PyTorch tensors
            input_seq = torch.tensor(input_seq, dtype=torch.float32)
            # target_seq = torch.tensor(target_seq, dtype=torch.float32)

            return input_seq
        
    class HybridModel(nn.Module):
        def __init__(self, input_size, hidden_size, output_size):
            super(HybridModel, self).__init__()
            self.hidden_size = hidden_size
            self.rnn1 = nn.RNN(hidden_size, hidden_size, batch_first=True)
            self.lstm1 = nn.LSTM(input_size, hidden_size, batch_first=True)
            self.gru1 = nn.GRU(hidden_size, hidden_size, batch_first=True)
            self.bn1 = nn.BatchNorm1d(hidden_size)
            self.dropout1 = nn.Dropout(0.3)
            
            self.rnn2 = nn.RNN(hidden_size, hidden_size, batch_first=True)
            self.lstm2 = nn.LSTM(hidden_size, hidden_size, batch_first=True)
            self.gru2 = nn.GRU(hidden_size, hidden_size, batch_first=True)
            self.bn2 = nn.BatchNorm1d(hidden_size)
            self.dropout2 = nn.Dropout(0.3)
            
            self.rnn3 = nn.RNN(hidden_size, hidden_size, batch_first=True)
            self.lstm3 = nn.LSTM(hidden_size, hidden_size, batch_first=True)
            self.gru3 = nn.GRU(hidden_size, hidden_size, batch_first=True)
            self.bn3 = nn.BatchNorm1d(hidden_size)
            self.dropout3 = nn.Dropout(0.3)
            
            self.rnn4 = nn.RNN(hidden_size, hidden_size, batch_first=True)
            self.lstm4 = nn.LSTM(hidden_size, hidden_size, batch_first=True)
            self.gru4 = nn.GRU(hidden_size, hidden_size, batch_first=True)
            self.bn4 = nn.BatchNorm1d(hidden_size)
            self.dropout4 = nn.Dropout(0.3)
            
            self.fc = nn.Linear(hidden_size, output_size)
            self.relu = nn.ReLU()

        def forward(self, x):
            batch_size = x.size(0)
            hidden1_rnn = self.init_hidden(batch_size)
            hidden2_rnn = self.init_hidden(batch_size)
            hidden3_rnn = self.init_hidden(batch_size)
            hidden4_rnn = self.init_hidden(batch_size)
            
        

            

            out_lstm1,_ = self.lstm1(x)
            out_gru1,_= self.gru1(out_lstm1)

            out_lstm2, _ = self.lstm2(out_gru1)
            out_gru2, _ = self.gru2(out_lstm2)
            
            

            out_rnn1, hidden1_rnn = self.rnn1(out_gru2, hidden1_rnn)
            out_rnn2, hidden2_rnn = self.rnn2(out_rnn1, hidden2_rnn)
            out_rnn3, hidden3_rnn = self.rnn3(out_rnn2, hidden3_rnn)
            out_rnn4, hidden4_rnn = self.rnn4(out_rnn3, hidden4_rnn)

            out = self.fc(out_rnn4[:, -1:, :])
            out = self.relu(out)
            return out

        def init_hidden(self, batch_size):
            return torch.zeros(1, batch_size, self.hidden_size)




   
    model = HybridModel(input_size=7, hidden_size=64, output_size=7)
    model.load_state_dict(torch.load('best_model_'+ data['area'] +'_s.pth'))
    model.eval()
    # Load your input data as a pandas DataFrame
    input_data = WeatherDataset('climate_final.csv')
    train_loader = torch.utils.data.DataLoader(input_data)
    
   
    inputs = next(iter(train_loader))
    inputs = inputs.to(device)
    predicted_outputs = []
   
    for i in range():
        # Get predicted output for the next day
        with torch.no_grad():
            predicted_output = model(inputs)[-1]  # select the last element of the output tensor for the next day
            predicted_outputs.append(predicted_output[0].numpy())
            
                    # slice tensor1 to remove the first day
            inputs = inputs[:, 1:, :]

            
            inputs = torch.cat([inputs, predicted_output.unsqueeze(1)], dim=1)

    print(type(predicted_outputs[0]))   
    # print(len(predicted_outputs[0]))
    ret = []
    for i in predicted_outputs:
        j = i.tolist()
        ret.append(j)
    x = ret
    # x = predicted_outputs[0]
    # x.insert(0,data['area'])
    # Print the output array
    print(x)
    return x 

@app.route('/')
def hello():
    return 'Hello, World!'
    
# counter = 0
@app.route('/weather', methods=['POST'])
def predictWeather():
    counter='8'
    data = request.get_json(silent=True)
    return [fun(data), data['area']]

# Store all the predicted outputs for the 7 days
    

@app.route('/cyclone', methods=['POST'])
def predictCyclone():
    counter='8'
    data = request.get_json(silent=True)
    x=fun(data)

    
    y = []
    y.append(x[0][5])
    y.append(x[0][4])
    y.append(x[0][0])
    y.append(x[0][1])
    y.append(x[0][3])
    y.append(x[0][2])
    print(y)

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(6, 64)
            self.fc2 = nn.Linear(64, 32)
            self.fc3 = nn.Linear(32,32)
            self.fc4 = nn.Linear(32,32)
            self.fc5 = nn.Linear(32, 8)
            
        def forward(self, x):
            x = nn.functional.relu(self.fc1(x))
            x = nn.functional.relu(self.fc2(x))
            x = nn.functional.relu(self.fc3(x))
            x = nn.functional.relu(self.fc4(x))
            x = self.fc5(x)
            return x
    
    y = torch.tensor(y)
    net = Net()
    net.load_state_dict(torch.load('best_weights.pt'))
    net.eval()
    with torch.no_grad():
        out_cyc = net(y)
    print(out_cyc)
    out_cyc = out_cyc.tolist()
    max_index = out_cyc.index(max(out_cyc))
    print(max_index)
    return [str(max_index),data['area'],maxind]

def __init__():
    counter=0