import sys
import os
import torch
import argparse
import pyhocon
import random
import datetime
from gowalla_data import *
from utils import *
from model import *
import numpy as np

torch.set_default_tensor_type(torch.FloatTensor)

parser = argparse.ArgumentParser(description='pytorch version of GraphSAGE')

parser.add_argument('--agg_func', type=str, default='MEAN')
parser.add_argument('--epochs', type=int, default=40)
parser.add_argument('--b_sz', type=int, default=16)
parser.add_argument('--seed', type=int, default=6)
parser.add_argument('--cuda', action='store_true', help='use CUDA')
parser.add_argument('--gcn', action='store_true')
parser.add_argument('--learn_method', type=str, default='sup')
parser.add_argument('--unsup_loss', type=str, default='normal')
parser.add_argument('--max_vali_f1', type=float, default=0)
parser.add_argument('--name', type=str, default='debug')
args = parser.parse_args()

if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    else:
        device_id = torch.cuda.current_device()
        print('using device', device_id, torch.cuda.get_device_name(device_id))

# device = torch.device("cuda" if args.cuda else "cpu")
device = torch.device("cuda")
print('DEVICE:', device)
print('agg_func:', args.agg_func)

if __name__ == '__main__':
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.set_default_tensor_type(torch.FloatTensor)

    # load data
    dataCenter = DataCenter()
    dataCenter.load_dataSet('./dataset/hou_node_map.pickle')
    # features
    features = torch.FloatTensor(getattr(dataCenter, 'feats')).to(device)
    train_loader, test_loader = tra_data(args.b_sz)

    hidden_emb_size = 128
    graphSage = GraphSage(2, features.size(1), hidden_emb_size, features,
                          getattr(dataCenter, 'adj_lists'), device, gcn=args.gcn, agg_func=args.agg_func)
    graphSage.to(device)

    num_labels = len(set(getattr(dataCenter, 'labels')))

    input_dim = 138
    hidden_dim = 200
    num_layers = 1
    output_dim = getattr(dataCenter, 'target_num')
    max_len = 9
    model = BiRNN(input_dim, hidden_dim, num_layers, output_dim, graphSage, max_len).to(device)

    if args.learn_method == 'sup':
        print('GraphSage with Supervised Learning')
    elif args.learn_method == 'plus_unsup':
        print('GraphSage with Supervised Learning plus Net Unsupervised Learning')
    else:
        print('GraphSage with Net Unsupervised Learning')

    starttime = datetime.datetime.now()

    for epoch in range(args.epochs):
        print('----------------------EPOCH %d-----------------------' % epoch)
        print('----------------------Train-----------------------')

        apply_model2(train_loader, model)

    # features = get_gnn_embeddings(graphSage, dataCenter, ds)
        print('----------------------Test-----------------------')
        test_model2(model, test_loader)

    endtime = datetime.datetime.now()
    print(starttime)
    print(endtime)
    print((endtime - starttime).seconds)
