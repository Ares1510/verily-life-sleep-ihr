import argparse

def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=100, help='number of epochs to train')
    parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
    parser.add_argument('--patience', type=int, default=10, help='patience for early stopping')
    parser.add_argument('--weight_decay', type=float, default=0.25, help='weight decay')
    return parser.parse_args()