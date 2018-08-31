from sklearn.metrics import f1_score

import yaml
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import network


def main(data_loader):
    accum_loss = 0.0
    preds, trues = [], []

    for x, t in data_loader:

        x = x.to(device)
        t = t.to(device)
        y = model(x)

        t_arr = t.cpu().tolist()
        y_arr = y.argmax(1).cpu().tolist()

        preds.extend(y_arr)
        trues.extend(t_arr)

        loss = criterion(y, t)  # log-softmax
        accum_loss += loss.data.cpu().numpy()

        if model.training:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    f1_value = f1_score(trues, preds, average='macro')
    return accum_loss, f1_value


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./config.yaml')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    params = yaml.load(open(args.config))

    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    train_data = torchvision.datasets.FashionMNIST(
        './data/fashion-mnist',
        transform=transform,
        train=True,
        download=True)

    test_data = torchvision.datasets.FashionMNIST(
        './data/fashion-mnist',
        transform=transform,
        train=False,
        download=True)

    train_data_loader = torch.utils.data.DataLoader(
        dataset=train_data,
        batch_size=params['batch_size'],
        shuffle=True)

    test_data_loader = torch.utils.data.DataLoader(
        dataset=test_data,
        batch_size=params['batch_size'],
        shuffle=False)

    # model = network.MLP(**params)
    model = network.ConvNet(**params)
    model.to(device)

    optimizer = optim.Adam(model.parameters())
    criterion = nn.NLLLoss()  # Negative Log Liklihood Loss

    for epoch in range(1, 1+params['n_epoch']):
        model.train()
        accum_loss, f1_value = main(train_data_loader)
        msg = f'[Train] Epoch #{epoch}, Loss: {accum_loss}, F1: {f1_value}'

        model.eval()
        accum_loss, f1_value = main(test_data_loader)
        msg += f' [Valid] Epoch #{epoch}, Loss: {accum_loss}, F1: {f1_value}'
        print(msg)

        state_dict = model.state_dict()
        torch.save(state_dict, f'./model/epoch_{epoch:03d}.pth')
