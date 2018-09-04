import csv
import yaml
import argparse
import torch
import torchvision
import network


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./config.yaml')
    parser.add_argument('--model', default='./model/epoch_001.pth')  # NOQA
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    params = yaml.load(open(args.config))
    model = network.ConvNet(**params)
    state_dict = torch.load(args.model)
    model.load_state_dict(state_dict)
    print(f'loaded: {args.model}')
    model.to(device)

    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    test_data = torchvision.datasets.FashionMNIST(
        './data/fashion-mnist',
        transform=transform,
        train=False,
        download=True)

    test_data_loader = torch.utils.data.DataLoader(
        dataset=test_data,
        batch_size=params['batch_size'],
        shuffle=False)

    preds = []
    for x, _ in test_data_loader:
        x = x.to(device)
        y = model.forward(x)
        pred = y.argmax(1).tolist()
        preds += pred

    SUBMISSION_PATH = 'submission.csv'
    with open(SUBMISSION_PATH, 'w') as f:
        writer = csv.writer(f, lineterminator='\n')
        writer.writerow(preds)
