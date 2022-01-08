import gzip, numpy, torch
import torch.nn as nn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

class MLP(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, hidden_size3, hidden_size4,  output_size):
        super(MLP, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size1)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(hidden_size1, hidden_size2)
        self.l3 = nn.Linear(hidden_size2, output_size)
        # self.l4 = nn.Linear(hidden_size3, output_size)
        # self.l5 = nn.Linear(hidden_size4, output_size)
        # torch.nn.init.uniform_(self.l1.weight, -0.001, 0.001)
        # torch.nn.init.uniform_(self.l2.weight, -0.001, 0.001)
        # torch.nn.init.uniform_(self.l3.weight, -0.001, 0.001)

    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        out = self.relu(out)
        out = self.l3(out)
        # out = self.relu(out)
        # out = self.l4(out)
        # out = self.relu(out)
        # out = self.l5(out)
        return out

if __name__ == '__main__':
    hlayer_size1 = 2933
    hlayer_size2 = 1466
    hlayer_size3 = 400
    hidden_size4 = 20
    if str(device) == 'cuda':
        batch_size = 256  # nombre de données lues à chaque fois
    else:
        batch_size = 5  # nombre de données lues à chaque fois
    nb_epochs = 10  # nombre de fois que la base de données sera lue
    eta = 0.001  # taux d'apprentissage

    # on lit les données
    ((data_train, label_train), (data_test, label_test)) = torch.load(gzip.open('mnist.pkl.gz'))

    # on crée les lecteurs de données
    train_dataset = torch.utils.data.TensorDataset(data_train, label_train)
    test_dataset = torch.utils.data.TensorDataset(data_test, label_test)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)

    model = MLP(data_train.shape[1], hlayer_size1, hlayer_size2, hlayer_size3, hidden_size4, label_train.shape[1])
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=eta)

    #training loop
    for n in range(nb_epochs):
        for x, t in train_loader:
            x, t = x.to(device), t.to(device)
            y = model(x)
            loss = criterion(y, t)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        print(f"epoch : {n}")

        # testing loop
        acc = 0
        for x, t in test_loader:
            x, t = x.to(device), t.to(device)
            y = model(x)
            acc += torch.argmax(y, 1) == torch.argmax(t, 1)
        print(acc/data_test.shape[0])