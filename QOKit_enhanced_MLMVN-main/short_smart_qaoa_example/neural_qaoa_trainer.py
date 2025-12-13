# Smart QAOA short example by ⚛️ Sigma PublishinQ Team ⚛️  
# https://www.linkedin.com/company/sigma-publishinq/about/


import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
import argparse
import os
import sys
sys.path.append('/content/drive/MyDrive/QOKit_enhanced_MLMVN/short_smart_qaoa_example/')
sys.path.append('/content/drive/MyDrive/QOKit_enhanced_MLMVN/QOKit/')

parser = argparse.ArgumentParser(
    description='Train neural network for QAOA MaxCut parameter prediction')

parser.add_argument("-n", "--number_of_nodes", type=int)
parser.add_argument("-p", "--layers", type=int)
parser.add_argument("--csv_path", help="path to CSV file", type=str)
args = parser.parse_known_args()[0]


class NeuralQAOANet(nn.Module):
    """Neural network for QAOA parameter prediction."""
    
    def __init__(self, bit_len, p):
        super(NeuralQAOANet, self).__init__()
        self.fc1 = nn.Linear(bit_len, 100)
        self.fc2 = nn.Linear(100, 100)
        self.fc3 = nn.Linear(100, 2 * p)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def df_to_dataloader(df, p, n, batch_size=32, train=True):
    """Converts DataFrame to DataLoader."""
    edges = ['x(%s,%s)' % (i, j) for i in range(n) for j in range(i + 1, n)]
    X = torch.tensor(df[edges].values, dtype=torch.float32)

    if train:
        # Parameters in QOKit: first gamma, then beta
        gamma_params = ['gamma%s' % i for i in range(p)]
        beta_params = ['beta%s' % i for i in range(p)]
        y = torch.tensor(df[gamma_params + beta_params].values, dtype=torch.float32)
    else:
        y = torch.zeros(X.shape[0], dtype=torch.float32)

    dataset = TensorDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=batch_size, pin_memory=True)
    return dataloader


def main():
    epochs = 100
    bit_len = int(args.number_of_nodes * (args.number_of_nodes - 1) / 2)
    p = args.layers

    # Create directories if they don't exist
    os.makedirs("train_datasets", exist_ok=True)
    os.makedirs("test_datasets", exist_ok=True)
    os.makedirs("preds", exist_ok=True)

    # Load data
    data_df = pd.read_csv(f"train_datasets/{args.csv_path}")
    test_df = pd.read_csv(f"test_datasets/{args.csv_path.replace('.csv', '_test_data.csv')}")

    # Split into train and validation
    msk = np.random.rand(len(data_df)) < 0.8
    train_df = data_df[msk]
    validation_df = data_df[~msk]

    # Create DataLoaders
    train_loader = df_to_dataloader(train_df, p, args.number_of_nodes)
    validation_loader = df_to_dataloader(validation_df, p, args.number_of_nodes)
    test_loader = df_to_dataloader(test_df, p, args.number_of_nodes, train=False)

    print(f'Training set size: {len(train_df)}. Validation set size: {len(validation_df)}. Test set size: {len(test_df)}')

    # Initialize model
    net = NeuralQAOANet(bit_len, p)
    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    # Training
    for epoch in range(epochs):
        net.train()
        running_loss = 0.0
        n_samples = 0
        
        for X, y in train_loader:
            optimizer.zero_grad()
            output = net(X)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
            
            n_samples += X.shape[0]
            running_loss += loss.item() * X.shape[0]

        # Validation
        net.eval()
        val_loss = 0.0
        n_val_samples = 0
        
        with torch.no_grad():
            for X, y in validation_loader:
                output = net(X)
                loss = criterion(output, y)
                val_loss += loss.item() * X.shape[0]
                n_val_samples += X.shape[0]

        print(f"Epoch: {epoch} loss: {running_loss/n_samples:.4f}, validation loss: {val_loss/n_val_samples:.4f}")

    # Prediction on test data
    net.eval()
    graphs, preds = [], []
    
    with torch.no_grad():
        for X, y in test_loader:
            output = net(X)
            graphs.append(X.cpu().numpy())
            preds.append(output.cpu().numpy())

    # Save results
    graphs = np.concatenate(graphs)
    edges = ['x(%s,%s)' % (i, j) for i in range(args.number_of_nodes) for j in range(i + 1, args.number_of_nodes)]
    graphs_df = pd.DataFrame(graphs, columns=edges)

    preds = np.concatenate(preds)
    gamma_params = ['gamma%s' % i for i in range(p)]
    beta_params = ['beta%s' % i for i in range(p)]
    preds_df = pd.DataFrame(preds, columns=gamma_params + beta_params)

    result = pd.concat([graphs_df, preds_df], axis=1)
    result.to_csv(f"preds/{args.csv_path.replace('.csv', '_preds.csv')}", index=False)
    print(f"Results saved to preds/{args.csv_path.replace('.csv', '_preds.csv')}")
    
    
    # Save trained model
    os.makedirs("models", exist_ok=True)
    torch.save(net.state_dict(), f"models/model_p_{p}_n_{args.number_of_nodes}.pth")
    print(f"Model saved to models/model_p_{p}_n_{args.number_of_nodes}.pth")


if __name__ == "__main__":
    main()