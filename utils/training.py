import datetime
import os

import torch
from argparse import Namespace
from models.utils.federated_model import FederatedModel
from datasets.utils.federated_dataset import FederatedDataset
from typing import Tuple
from torch.utils.data import DataLoader
from utils.logger import CsvWriter


def global_evaluate(model: FederatedModel, test_dl: DataLoader, setting: str, name: str) -> Tuple[list, list]:
    accs = 0
    dl = test_dl
    net = model.global_net
    status = net.training
    net.eval()
    correct, total, top1, top5 = 0.0, 0.0, 0.0, 0.0
    for batch_idx, (images, labels) in enumerate(dl):
        with torch.no_grad():
            images, labels = images.to(model.device), labels.to(model.device)
            outputs_dict = net(images)
            logits = outputs_dict['logits']
            _, max5 = torch.topk(logits, 5, dim=-1)
            labels = labels.view(-1, 1)
            top1 += (labels == max5[:, 0:1]).sum().item()
            top5 += (labels == max5).sum().item()
            total += labels.size(0)
    top1acc = round(100 * top1 / total, 2)
    top5acc = round(100 * top5 / total, 2)
    accs = top1acc
    net.train(status)
    return accs


def train(model: FederatedModel, private_dataset: FederatedDataset,
          args: Namespace) -> None:
    if args.csv_log:
        csv_writer = CsvWriter(args, private_dataset)
        model.weight_path = os.path.join(csv_writer.para_foloder_path, 'weight.txt')
        # model.eh_path = os.path.join(csv_writer.para_foloder_path, 'eh.txt')
        # 删除过去的 看情况是否用
        if os.path.exists(model.weight_path):
            os.remove(model.weight_path)

    pri_train_loaders, test_loaders, net_cls_counts, client_type = private_dataset.get_data_loaders()
    model.trainloaders = pri_train_loaders
    model.testlodaers = test_loaders
    model.net_cls_counts = net_cls_counts
    model.client_type = client_type

    if hasattr(model, 'ini'):
        model.ini()

    accs_list = []

    Epoch = args.communication_epoch
    option_learning_decay = args.learning_decay

    for epoch_index in range(Epoch):
        model.epoch_index = epoch_index
        if hasattr(model, 'loc_update'):
            '''
            测速Module
            '''
            # if epoch_index==1:
            #     start_time = datetime.datetime.now()
            #     epoch_loc_ss_dict = model.loc_update(pri_train_loaders)
            #     end_time = datetime.datetime.now()
            #     use_time=end_time - start_time
            #     print(end_time - start_time)
            #     with open(args.dataset+'_time.csv','a') as f:
            #         f.write(args.model+','+str(use_time)+'\n')
            # else:
            model.loc_update(pri_train_loaders)
            if option_learning_decay == True:
                model.local_lr = args.local_lr * (1 - epoch_index / Epoch * 0.9)
        accs = global_evaluate(model, test_loaders, private_dataset.SETTING, private_dataset.NAME)
        accs = round(accs, 3)
        accs_list.append(accs)
        print('The ' + str(epoch_index) + ' Communcation Accuracy:' + str(accs) + ' ' + 'Method:' + model.args.model)
    if args.csv_log:
        csv_writer.write_acc(accs_list)
