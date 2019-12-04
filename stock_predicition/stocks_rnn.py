import torch
from torch import nn
from torch.nn import Linear, LSTM, Module, Tanh
from torch.utils.data import DataLoader, TensorDataset, Dataset


class StockPredictor(Module):
    def __init__(self, input_size, hidden_size, num_layers, batch_size):
        super().__init__()
        self.batch_size = batch_size
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = LSTM(input_size, hidden_size, num_layers, bias=False)
        self.linear256 = Linear(hidden_size, 128, bias=False)
        self.linear128 = Linear(128, 1, bias=False)
        self.tanh = Tanh()

    def init_hidden(self):
        return (torch.zeros(self.num_layers, self.batch_size, self.hidden_size),
                torch.zeros(self.num_layers, self.batch_size, self.hidden_size))

    def forward(self, x):
        lstm_out, self.hidden = self.lstm(x.view(len(x), self.batch_size, -1))
        out = self.linear256(lstm_out)
        out = self.tanh(out)
        out = self.linear128(out)
        out = self.tanh(out)
        return out


class StocksLoader(DataLoader):
    def __init__(self, dataset, **kwargs):
        super().__init__(dataset, **kwargs)
        self.data = dataset

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]


class StockDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __getitem__(self, index):
        label = self.labels[index]
        data = self.data[index]
        return data, label

    def __len__(self):
        return len(self.labels)


if __name__ == '__main__':
    import pandas as pd
    from sklearn.preprocessing import MinMaxScaler
    from torch.optim import Adam
    from stock_predicition import preprocessing

    import matplotlib.pyplot as plt
    import numpy as np
    # read stock data file
    df = pd.read_csv("data_stocks.csv")

    scl = MinMaxScaler()

    sp500 = preprocessing.pandas_column_to_numpy_array(df.SP500)
    sp500 = preprocessing.scale_np_array(sp500, scl)
    X, y = preprocessing.prepare_data_from_np_array(sp500, 15)
    X = torch.from_numpy(X)
    y = torch.from_numpy(y)
    data_set = StockDataset(X, y)
    train_loader = StocksLoader(data_set, shuffle=False, batch_size=128)
    model = StockPredictor(15, 256, 1, 1)
    optimizer = Adam(model.parameters())
    criterion = nn.MSELoss()
    losses = []
    NUM_EPOCHS = 100
    for epoch in range(NUM_EPOCHS):
        running_loss = 0.0
        print("Epoch {}/{}".format(epoch + 1, NUM_EPOCHS))
        for i, data in enumerate(train_loader, 0):
            optimizer.zero_grad()
            input_, label = data
            model.double()
            outputs = model(input_)
            loss = criterion(outputs, label)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss))
        losses.append(running_loss)
    losses = np.array(losses)
    torch.save(model, "model.pickle")
    plt.plot(losses)
    plt.show()
    plt.savefig("loss_trend.png")