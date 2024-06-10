import copy
from functorch import make_functional, make_functional_with_buffers

import torch.nn.functional as F
import numpy as np
import torch.nn as nn
import torch
import torchvision
from argparse import Namespace

from torch import optim

from models.utils.util import krum, trimmed_mean, bulyan, row_into_parameters, fools_gold, geometric_median_update, multi_krum
from utils.conf import get_device, base_path
from utils.conf import checkpoint_path
from utils.util import create_if_not_exists
import os
from tqdm import tqdm
from utils.util import HE, EH
import numpy as np
from utils.finch import FINCH


class FederatedModel(nn.Module):
    """
    Federated learning model.
    """
    NAME = None

    def __init__(self, nets_list: list,
                 args: Namespace, transform: torchvision.transforms) -> None:
        super(FederatedModel, self).__init__()
        self.nets_list = nets_list

        self.args = args
        self.transform = transform

        # For Online
        self.random_state = np.random.RandomState()
        self.online_num = np.ceil(self.args.parti_num * self.args.online_ratio).item()
        self.online_num = int(self.online_num)

        self.global_net = None
        self.device = get_device(device_id=self.args.device_id)

        self.local_epoch = args.local_epoch
        self.local_batch_size = args.local_batch_size
        self.local_lr = args.local_lr
        self.trainloaders = None
        self.testlodaers = None
        self.publicloader = None
        self.net_cls_counts = None
        self.client_type = None

        self.epoch_index = 0  # Save the Communication Index

        self.checkpoint_path = checkpoint_path() + self.args.dataset + '/' + self.args.structure + '/'
        create_if_not_exists(self.checkpoint_path)
        self.net_to_device()
        self.random_net = copy.deepcopy(self.nets_list[0]).to(self.device)

        self.agg_num = 0

        if self.args.averaging in ['krum', 'bulyan', 'trimmed_mean', 'multi_krum']:
            self.momentum = 0.9
            self.learning_rate = self.local_lr
            self.current_weights = np.concatenate([i.data.numpy().flatten() for i in copy.deepcopy(self.nets_list[0]).cpu().parameters()])
            self.velocity = np.zeros(self.current_weights.shape, self.current_weights.dtype)
            self.n = 5

        if self.args.averaging == 'rsa':
            self.alpha = 0.001
            self.l1_lambda = 0.5
            self.weight_lambda = 0.01

        if self.args.averaging == 'rfa':
            self.max_iter = 3

        if self.args.averaging == 'dnc':
            self.sub_dim = 10000
            self.num_iters = 1
            self.filter_frac = 1.0

        if self.args.averaging == 'sage_flow':
            self.eth = 2.2
            self.delta = 5

    def net_to_device(self):
        for net in self.nets_list:
            net.to(self.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def get_scheduler(self):
        return

    def ini(self):
        pass

    def col_update(self, communication_idx, publoader):
        pass

    def loc_update(self, priloader_list):
        pass

    def load_pretrained_nets(self):
        if self.load:
            for j in range(self.args.parti_num):
                pretrain_path = os.path.join(self.checkpoint_path, 'pretrain')
                save_path = os.path.join(pretrain_path, str(j) + '.ckpt')
                self.nets_list[j].load_state_dict(torch.load(save_path, self.device))
        else:
            pass

    def copy_nets2_prevnets(self):
        nets_list = self.nets_list
        prev_nets_list = self.prev_nets_list
        for net_id, net in enumerate(nets_list):
            net_para = net.state_dict()
            prev_net = prev_nets_list[net_id]
            prev_net.load_state_dict(net_para)

    def aggregate_nets(self, freq=None):
        global_net = self.global_net
        temp_net = copy.deepcopy(global_net)
        nets_list = self.nets_list

        evils = self.args.evils
        if evils == 'RandomNoise':
            for i in self.online_clients:
                if self.client_type[i] == False:
                    random_net = copy.deepcopy(self.random_net)
                    self.nets_list[i] = random_net

        elif evils == 'AddNoise':
            for i in self.online_clients:
                if self.client_type[i] == False:
                    sele_net = self.nets_list[i]
                    random_net = copy.deepcopy(self.random_net)
                    noise_weight = 0.5
                    for name, param in sele_net.state_dict().items():
                        param += torch.tensor(copy.deepcopy(noise_weight * (random_net.state_dict()[name] - param)), dtype=param.dtype)

        online_clients = self.online_clients

        if self.args.averaging == 'weight':
            online_clients_dl = [self.trainloaders[online_clients_index] for online_clients_index in online_clients]
            online_clients_len = [len(dl.sampler.indices) for dl in online_clients_dl]
            online_clients_all = np.sum(online_clients_len)
            freq = online_clients_len / online_clients_all
            global_w = self._agg(online_clients, nets_list, freq)
            print(freq)
            global_net.load_state_dict(global_w)

        elif self.args.averaging == 'equal':
            parti_num = len(online_clients)
            freq = [1 / parti_num for _ in range(parti_num)]
            global_w = self._agg(online_clients, nets_list, freq)
            print(freq)
            global_net.load_state_dict(global_w)

        elif self.args.averaging in ['krum', 'bulyan', 'trimmed_mean', 'multi_krum']:

            with torch.no_grad():
                all_grads = []
                for i in online_clients:
                    grads = {}
                    net_all_grads = []
                    for name, param0 in temp_net.state_dict().items():
                        param1 = nets_list[i].state_dict()[name]
                        grads[name] = (param0.detach() - param1.detach()) / self.local_lr
                        net_all_grads.append(copy.deepcopy(grads[name].view(-1)))

                    net_all_grads = torch.cat(net_all_grads, dim=0).cpu().numpy()
                    all_grads.append(net_all_grads)
                all_grads = np.array(all_grads)

            # bad_client_num = int(self.args.bad_client_rate * len(self.online_clients))
            f = len(self.online_clients) // 2  # worse case 50% malicious points
            k = len(self.online_clients) - f - 1

            if self.args.averaging == 'krum':
                current_grads = krum(all_grads, len(online_clients), f - k)
            elif self.args.averaging == 'multi_krum':
                current_grads = multi_krum(all_grads, len(online_clients), f - k, n=self.n)
            elif self.args.averaging == 'bulyan':
                current_grads = bulyan(all_grads, len(online_clients), f - k)
            elif self.args.averaging == 'trimmed_mean':
                current_grads = trimmed_mean(all_grads, len(online_clients), k)
            self.velocity = self.momentum * self.velocity - self.learning_rate * current_grads
            self.current_weights += self.velocity

            row_into_parameters(self.current_weights, self.global_net.parameters())

        elif self.args.averaging in ['SDEA']:
            self.weights_ori = nn.Parameter(torch.FloatTensor(self.args.parti_num).to(self.device))
            nn.init.constant_(self.weights_ori, 1. / self.online_num)
            self.weight_optimizer = torch.optim.Adam([self.weights_ori],
                                                     lr=self.args.public_lr, weight_decay=1e-5)

            parti_num = len(online_clients)
            iterator = tqdm(range(self.args.public_epoch))
            for _ in iterator:
                for batch_idx, (images, _) in enumerate(self.publicloader):
                    weights_ori = self.weights_ori[torch.tensor(online_clients)]
                    weights = torch.exp(weights_ori) / torch.sum(torch.exp(weights_ori))
                    img_0, img_1, _, _ = images[0], images[1], images[2], images[3]
                    img_0 = img_0.to(self.device)
                    img_1 = img_1.to(self.device)
                    buffers = list(self.global_net.buffers())
                    used_buffers = len(buffers) > 0
                    if used_buffers:
                        func_g, _, _ = make_functional_with_buffers(self.global_net)
                    else:
                        func_g, _ = make_functional(self.global_net)
                    para_list = []
                    if used_buffers:
                        buffer_list = []
                    for index, net_id in enumerate(online_clients):
                        net = nets_list[net_id]
                        if used_buffers:
                            _, params, buffers = make_functional_with_buffers(net)
                            buffer_list.append(buffers)
                        else:
                            _, params = make_functional(net)
                        para_list.append(params)

                    new_g_para = []
                    new_g_buffer = []
                    for j in range(len(para_list[0])):
                        new_g_para.append(
                            torch.sum(torch.stack([weights[i] * para_list[i][j] for i in range(parti_num)]), dim=0))
                    if used_buffers:
                        for j in range(len(buffer_list[0])):
                            new_g_buffer.append(
                                torch.sum(torch.stack([weights[i] * buffer_list[i][j] for i in range(parti_num)]), dim=0)
                            )
                    if used_buffers:
                        q_dict = func_g(new_g_para, new_g_buffer, img_0)
                        k_dict = func_g(new_g_para, new_g_buffer, img_1)

                    else:
                        q_dict = func_g(new_g_para, img_0)
                        k_dict = func_g(new_g_para, img_1)

                    q_logits = q_dict['logits']

                    k_logits = k_dict['logits']

                    sharpened_probs_q = torch.nn.functional.softmax(q_logits, dim=-1)
                    sharpened_probs_k = torch.nn.functional.softmax(k_logits, dim=-1)

                    loss_eh = 0.5 * (EH(sharpened_probs_q) + EH(sharpened_probs_k))
                    loss_he = 0.5 * (HE(sharpened_probs_q) + HE(sharpened_probs_k))

                    loss = loss_eh - loss_he
                    self.weight_optimizer.zero_grad()
                    loss.backward()
                    iterator.desc = "Col loss = sharp: %0.3f, diver: %0.3f" % (loss_eh, -loss_he)
                    self.weight_optimizer.step()
                print(weights)

            weights = torch.exp(weights_ori) / torch.sum(torch.exp(weights_ori))

            '''
            Finch
            '''

            try:
                fin = FINCH()

                weights = weights.cpu().detach().view(-1, 1)
                fin.fit(weights)
                select_partitions = (fin.partitions)['parition_0']
                evils_center = min(select_partitions['cluster_centers'])
                evils_center_idx = np.where(select_partitions['cluster_centers'] == evils_center)[0]
                evils_idx = select_partitions['cluster_core_indices'][int(evils_center_idx)]
                benign_idx = [i for i in range(len(weights)) if i not in evils_idx]

                weights_envis_sum = torch.sum(weights[evils_idx])
                weights_benign_sum = 1 - weights_envis_sum
                weights[benign_idx] = weights_benign_sum / len(benign_idx)
                weights = weights.view(-1)
                # print(weights)
                freq = weights
            except Exception as e:
                print(e)
                print('use mean')
                parti_num = len(online_clients)
                freq = [1 / parti_num for _ in range(parti_num)]
                freq = torch.tensor(freq)

            global_w = self._agg(online_clients, nets_list, freq)
            print(freq)
            if self.args.csv_log:
                with open(self.weight_path, 'a') as f:
                    f.write(str(self.agg_num) + ': ' + str(list(freq.cpu().numpy()))[1:-1] + '\n')
            global_net.load_state_dict(global_w)

        elif self.args.averaging == 'eval':
            prob_dict = {}
            for index, net_id in enumerate(online_clients):
                prob_dict[(index)] = []
            for batch_idx, (images, _) in enumerate(self.publicloader):
                img_0, img_1, _, _ = images[0], images[1], images[2], images[3]
                img_0 = img_0.to(self.device)
                for index, net_id in enumerate(online_clients):
                    net = nets_list[net_id]
                    outputs_dict = net(img_0)
                    logits = outputs_dict['logits']
                    probs = torch.nn.functional.softmax(logits, dim=-1)
                    eh = EH(probs)
                    print(eh)
                    prob_dict[(index)].append(eh.cpu().detach().numpy().astype(float))

        for _, net in enumerate(nets_list):
            net.load_state_dict(global_net.state_dict())
        self.agg_num += 1

    def _agg(self, online_clients, nets_list, freq):
        global_w = (self.global_net.state_dict())
        first = True
        for index, net_id in enumerate(online_clients):
            net = nets_list[net_id]
            net_para = net.state_dict()
            # if net_id == 0:
            if first:
                first = False
                for key in net_para:
                    global_w[key] = net_para[key] * freq[index]
            else:
                for key in net_para:
                    global_w[key] += net_para[key] * freq[index]
        return global_w
