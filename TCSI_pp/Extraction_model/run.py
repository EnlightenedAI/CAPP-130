import time
import torch
import numpy as np
from train_eval import train,test
from importlib import import_module
import argparse
from utils import build_dataset, get_time_dif
from torch.utils.data import TensorDataset, DataLoader
from torchsampler import ImbalancedDatasetSampler
import os

parser = argparse.ArgumentParser(description='Chinese Text Classification')
parser.add_argument('--model', type=str, required=True, help='choose a model: Bert, ERNIE')
parser.add_argument('--data', type=str, required=True, help='choose a Data')
args = parser.parse_args()

if __name__ == '__main__':
    dataset=args.data
    model_name = args.model
    x = import_module('models.' + model_name)
    config = x.Config()
    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    start_time = time.time()
    print("Loading data...")
    data_paths=f"../../CAPP_130/Subdataset/Extraction/{dataset}_dataset"
    train_data, dev_data, test_data = build_dataset(config,data_paths)
    train_sencens=[]
    train_lable=[]
    train_lens=[]
    train_mask=[]
    for train_s in train_data:
        train_sencens.append(train_s[0])
        train_lable.append(train_s[1])
        train_lens.append(train_s[2])
        train_mask.append(train_s[3])
    train_dataset = TensorDataset(torch.tensor(train_sencens),torch.tensor(train_lable),torch.tensor(train_lens),torch.tensor(train_mask))
    test_sencens=[]
    test_lable=[]
    test_lens=[]
    test_mask=[]
    for test_s in test_data:
        test_sencens.append(test_s[0])
        test_lable.append(test_s[1])
        test_lens.append(test_s[2])
        test_mask.append(test_s[3])
    test_dataset = TensorDataset(torch.tensor(test_sencens), torch.tensor(test_lable), torch.tensor(test_lens),
                                  torch.tensor(test_mask))
    dev_sencens = []
    dev_lable = []
    dev_lens = []
    dev_mask = []
    for dev_s in dev_data:
        dev_sencens.append(dev_s[0])
        dev_lable.append(dev_s[1])
        dev_lens.append(dev_s[2])
        dev_mask.append(dev_s[3])
    dev_dataset = TensorDataset(torch.tensor(dev_sencens), torch.tensor(dev_lable), torch.tensor(dev_lens),
                                  torch.tensor(dev_mask))

    train_iter = DataLoader(train_dataset,sampler=ImbalancedDatasetSampler(train_dataset), batch_size=config.batch_size)
    dev_iter = DataLoader(dev_dataset, batch_size=config.batch_size)
    test_iter = DataLoader(test_dataset, batch_size=config.batch_size)
    time_dif = get_time_dif(start_time)
    model = x.Model(config).to(config.device)
    if not os.path.exists(f'save_model'):
        os.mkdir(f'/save_model')
    config.save_path= f'/save_model/saved_{args.model}_dict_{dataset} .ckpt'
    config.save_path_acc_loss=f'/save_model/saved_{args.model}_dict_{dataset}.json'
    config.learning_rate= 5e-5
    train(config, model, train_iter, test_iter, dev_iter)
    model.load_state_dict(torch.load(config.save_path))
    model.eval()
    test(config, model, test_iter)

