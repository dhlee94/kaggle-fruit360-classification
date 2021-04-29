from train import train
from utils import AddParserManager
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file_path', type=str, default='./config.yml')
    parser.add_argument('--config_name', type=str, default='base')

    args = parser.parse_args()
    cfg = AddParserManager(args.config_file_path, args.config_name)
    if train:
        train(cfg)
