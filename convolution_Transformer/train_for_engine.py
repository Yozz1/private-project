import torch.nn as nn
import torch
import torch.optim as optim

import os
import argparse
import datetime
import numpy as np
import pandas as pd

from tqdm import tqdm
from utils import utils
from base.AdaRNN import AdaRNN

from data_preprocess import NASA_data

# import pretty_errors
import dataset.data_process as data_process
import matplotlib.pyplot as plt
from tst import Transformer
from create_data import read_create_data
from matplotlib import pyplot as plt
from com import huatu
from simple_data_prepocess import read_simple
from MLP_TST.transformer import MLP_Transformer


def pprint(*text):
    # print with UTC+8 time
    time = '[' + str(datetime.datetime.utcnow() +
                     datetime.timedelta(hours=8))[:19] + '] -'
    print(time, *text, flush=True)
    if args.log_file is None:
        return
    with open(args.log_file, 'a') as f:
        print(time, *text, flush=True, file=f)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train_epoch(args, model, optimizer, src_train_loader):
    model.train()
    criterion = nn.MSELoss()
    criterion_1 = nn.L1Loss()
    loss_all = []
    loss_1_all = []
    # dist_mat = torch.zeros(args.num_layer, args.len_seq).cuda()
    len_loader = len(src_train_loader)
    list1 = np.array([['pred1', 'label1']])
    for feature_all, label, label_reg in tqdm(src_train_loader, total=len_loader):
        optimizer.zero_grad()
        feature_all = feature_all.cuda()
        label_reg = label_reg.cuda()

        pred_all, list_encoding = model(feature_all)
        pred_all = torch.mean(pred_all, dim=1).view(pred_all.shape[0])
        loss_s = criterion(pred_all, label_reg)


        total_loss = loss_s
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        pred = pred_all.data.cpu().numpy()
        label = label_reg.data.cpu().numpy()
        label1 = np.column_stack((pred, label))
        label1 = label1.reshape(-1, 2)
        list1 = np.vstack((list1, label1))

    loss = np.array(loss_all).mean(axis=0)
    loss_l1 = np.array(loss_1_all).mean()

    return loss, loss_l1,  list1



def test_epoch(model, test_loader, prefix='Test'):
    model.eval()
    total_loss = 0
    total_loss_1 = 0
    total_loss_r = 0
    correct = 0
    criterion = nn.MSELoss()
    criterion_1 = nn.L1Loss()

    list1 = np.array([['pred1', 'label1']])
    for feature, label, label_reg in tqdm(test_loader, desc=prefix, total=len(test_loader)):
        feature, label_reg = feature.cuda().float(), label_reg.cuda().float()
        with torch.no_grad():
            pred, _ = model(feature)
            pred = torch.mean(pred, dim=1).view(pred.shape[0])
        loss = criterion(pred, label_reg)
        loss_r = torch.sqrt(loss)
        loss_1 = criterion_1(pred, label_reg)
        total_loss += loss.item()
        total_loss_1 += loss_1.item()
        total_loss_r += loss_r.item()

        pred = pred.data.cpu().numpy()
        label = label_reg.data.cpu().numpy()
        label1 = np.column_stack((pred, label))
        label1 = label1.reshape(-1, 2)
        list1 = np.vstack((list1, label1))

    loss = total_loss / len(test_loader)
    loss_1 = total_loss_1 / len(test_loader)
    loss_r = loss_r / len(test_loader)
    return loss, loss_1, loss_r, list1


def test_epoch_inference(model, test_loader, prefix='Test'):
    model.eval()
    total_loss = 0
    total_loss_1 = 0
    total_loss_r = 0
    correct = 0
    criterion = nn.MSELoss()
    criterion_1 = nn.L1Loss()
    i = 0
    list1 = np.array([['pred1', 'label1']])

    for feature, label, label_reg in tqdm(test_loader, desc=prefix, total=len(test_loader)):
        feature, label_reg = feature.cuda().float(), label_reg.cuda().float()
        with torch.no_grad():
            pred, _ = model(feature)
            pred = torch.mean(pred, dim=1).view(pred.shape[0])
        loss = criterion(pred, label_reg)
        loss_r = torch.sqrt(loss)
        loss_1 = criterion_1(pred, label_reg)
        total_loss += loss.item()
        total_loss_1 += loss_1.item()
        total_loss_r += loss_r.item()
        if i == 0:
            label_list = label_reg.cpu().numpy()
            predict_list = pred.cpu().numpy()
        else:
            label_list = np.hstack((label_list, label_reg.cpu().numpy()))
            predict_list = np.hstack((predict_list, pred.cpu().numpy()))

        pred = pred.data.cpu().numpy()
        label = label_reg.data.cpu().numpy()
        label1 = np.column_stack((pred, label))
        label1 = label1.reshape(-1, 2)
        list1 = np.vstack((list1, label1))

        i = i + 1
    loss = total_loss / len(test_loader)
    loss_1 = total_loss_1 / len(test_loader)
    loss_r = total_loss_r / len(test_loader)
    return loss, loss_1, loss_r, label_list, predict_list, list1


def inference(model, data_loader):
    loss, loss_1, loss_r, label_list, predict_list, list1 = test_epoch_inference(
        model, data_loader, prefix='Inference')
    return loss, loss_1, loss_r, label_list, predict_list, list1


def inference_all(output_path, model, model_path, loaders):
    pprint('评价中...')
    loss_list = []
    loss_l1_list = []
    loss_r_list = []
    model.load_state_dict(torch.load(model_path))
    i = 0
    list_name = ['train', 'valid', 'test']
    for loader in loaders:
        loss, loss_1, loss_r, label_list, predict_list, list = inference(
            model, loader)
        loss_list.append(loss)
        loss_l1_list.append(loss_1)
        loss_r_list.append(loss_r)

        if i == 0:
            list1 = np.array(list)
            # list1 = sorted(list1, key=lambda x: x[1], reverse=True)
            list1 = pd.DataFrame(list1)
            list1.to_csv('label_train_all.csv', index=None, header=None)

        if i == 1:
            list2 = np.array(list)
            list2 = sorted(list2, key=lambda x: x[1], reverse=True)
            list2 = pd.DataFrame(list2)
            list2.to_csv('label_test1.csv', index=None, header=None)

        if i == 2:
            list3 = np.array(list)
            # list3 = sorted(list3, key=lambda x: x[1], reverse=True)
            list3 = pd.DataFrame(list3)
            list3.to_csv('label_test2.csv', index=None, header=None)

        i = i + 1

    return loss_list, loss_l1_list, loss_r_list, list3


def main_transfer(args):
    print(args)

    output_path = args.outdir + '_'
    # "-hidden" + str(args.hidden_dim) + "-head" + str(args.num_head)
    save_model_name = args.model_name  +'_' + str(args.lr) + '.pkl'
    utils.dir_exist(output_path)
    pprint('参数计算...')

    source_loader,  test_loader, valid_loader = read_simple(batch_size=args.batch_size, featrue=args.d_input,shuffle=True, )


    args.log_file = os.path.join(output_path, 'run.log')
    pprint('创建模型...')
    ######
    # Model parameters
    d_model = args.hidden_dim  # 32  Lattent dim
    q = args.qv  # Query size
    v = args.qv  # Value size
    h = args.num_head  # 4   Number of heads
    N = args.num_layer  # Number of encoder and decoder to stack
    attention_size = 12  # Attention window size
    pe = "regular"  # Positional encoding
    chunk_mode = None
    d_input = args.d_input  # From dataset
    d_output = 1  # From dataset


    #模型选择
    if d_input == 24:
        model = MLP_Transformer(d_input, d_model, d_output, q, v, h, N,
                                attention_size=attention_size,
                                chunk_mode=chunk_mode,
                                pe=pe, pe_period=24).cuda()
    else:
        model = Transformer(d_input, d_model, d_output, q, v, h, N,
                                attention_size=attention_size,
                                chunk_mode=chunk_mode,
                                pe=pe, pe_period=24).cuda()

    #####
    num_model = count_parameters(model)
    print('#模型参数量:', num_model)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    best_score = np.inf
    best_epoch, stop_round = 0, 0

    loss_all = []
    for epoch in range(args.n_epochs):
        pprint('Epoch:', epoch)
        pprint('训练中...')

        train_epoch(args, model, optimizer, source_loader )

        pprint('每批次评估...')
        train_loss, train_loss_l1, train_loss_r, list1 = test_epoch(
            model, source_loader, prefix='Train')
        val_loss, val_loss_l1, val_loss_r, list2 = test_epoch(
            model, valid_loader, prefix='Valid')
        test_loss, test_loss_l1, test_loss_r, list3 = test_epoch(
            model, test_loader, prefix='Test')

        pprint('train %.6f, valid %.6f, test %.6f' %
               (train_loss_r, val_loss_r, test_loss_r))
        loss_all.append(train_loss)

        #测试集最优
        # if val_loss_r < best_score:
        #     best_score = val_loss_r
        #     stop_round = 0
        #     best_epoch = epoch
        #     torch.save(model.state_dict(), os.path.join(
        #         output_path, save_model_name))
        # else:
        #     stop_round += 1
        #     if stop_round >= args.early_stop:
        #         pprint('提前停止')
        #         break

        # 验证集最优
        if test_loss_r < best_score:
            best_score = test_loss_r.item()
            stop_round = 0
            best_epoch = epoch
            torch.save(model.state_dict(), os.path.join(
                output_path, save_model_name))
        else:
            stop_round += 1
            if stop_round >= args.early_stop:
                pprint('提前停止')
                break





    pprint('val最低loss:', best_score, '@', best_epoch)
    loaders = source_loader, valid_loader, test_loader
    loss_list, loss_l1_list, loss_r_list, list3 = inference_all(output_path, model, os.path.join(
        output_path, save_model_name), loaders)
    pprint('MSE: train %.6f, valid %.6f, test %.6f' %
           (loss_list[0], loss_list[1], loss_list[2]))
    pprint('L1:  train %.6f, valid %.6f, test %.6f' %
           (loss_l1_list[0], loss_l1_list[1], loss_l1_list[2]))
    pprint('RMSE: train %.6f, valid %.6f, test %.6f' %
           (loss_r_list[0], loss_r_list[1], loss_r_list[2]))
    pprint('结束.')

    loss_all = np.array(loss_all)
    loss_all = loss_all.reshape(-1, )
    loss_all = pd.DataFrame(loss_all)
    loss_all.to_csv('训练loss.csv', index=None, header=None)
    plt.plot(loss_all, label='loss值', linewidth=0.8, color='purple')
    plt.title('测试集')
    plt.legend()
    plt.show()

    huatu()




def get_args():
    parser = argparse.ArgumentParser()

    # model
    parser.add_argument('--model_name', default='NASA_engine')
    parser.add_argument('--d_feat', type=int, default=12)

    parser.add_argument('--hidden_size', type=int, default=64)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--pre_epoch', type=int, default=40)  # 25
    parser.add_argument('--num_layer', type=int, default=1)  # 25

    # training
    parser.add_argument('--n_epochs', type=int, default=100000)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--early_stop', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=2048)
    parser.add_argument('--train_type', type=str, default='all')

    parser.add_argument('--data_mode', type=str,
                        default='pre_process')
    parser.add_argument('--num_domain', type=int, default=2)
    parser.add_argument('--len_seq', type=int, default=1)
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--num_head', type=int, default=32)
    parser.add_argument('--d_input', type=int, default=24)
    parser.add_argument('--qv', type=int, default=64)

    # other
    parser.add_argument('--seed', type=int, default=10)
    parser.add_argument('--data_path', default="./data/")
    parser.add_argument('--outdir', default='./outputs')
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--log_file', type=str, default='run.log')
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--len_win', type=int, default=0)
    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = get_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    main_transfer(args)
