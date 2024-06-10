import torch.multiprocessing
import setproctitle
import datetime
import socket
import torch
import uuid
import sys
import os

torch.multiprocessing.set_sharing_strategy('file_system')
conf_path = os.getcwd()
sys.path.append(conf_path)
sys.path.append(conf_path + '/datasets')
sys.path.append(conf_path + '/backbone')
sys.path.append(conf_path + '/models')
from datasets import Priv_NAMES as DATASET_NAMES
from utils.conf import set_random_seed
from datasets import get_prive_dataset, get_public_dataset
from utils.best_args import best_args
from argparse import ArgumentParser
from models import get_all_models
from utils.training import train
from models import get_model

def parse_args():
    parser = ArgumentParser(description='Federated Learning', allow_abbrev=False)
    parser.add_argument('--device_id', type=int, default=1, help='The Device Id for Experiment')
    # parser.add_argument('--communication_epoch', type=int, default=100, help='The Communication Epoch in Federated Learning')

    parser.add_argument('--local_epoch', type=int, default=10, help='The Local Epoch for each Participant')

    parser.add_argument('--local_batch_size', type=int, default=64)
    # parser.add_argument('--parti_num', type=int, default=10, help='The Number for Participants')
    parser.add_argument('--seed', type=int, default=0, help='The random seed.')
    parser.add_argument('--model', type=str, default='fedprox',  # moon fedavg fedreg fedavgnorm fedalign fedours
                        help='Model name.', choices=get_all_models())
    parser.add_argument('--structure', type=str, default='homogeneity')
    parser.add_argument('--dataset', type=str, default='fl_cifar10', choices=DATASET_NAMES)  # fl_mnist, fl_cifar10, fl_cifar100 fl_fmnist
    parser.add_argument('--pri_aug', type=str, default='weak', help='Private data augmentation')
    parser.add_argument('--beta', type=float, default=0.5, help='The beta for label skew')  # 0.3 0.5
    parser.add_argument('--online_ratio', type=float, default=1, help='The ratio for online clients')
    parser.add_argument('--optimizer', type=str, default='sgd', help='adam or sgd')
    # parser.add_argument('--local_lr', type=float, default=1e-2, help='The learning rate for local updating')
    parser.add_argument('--reg', type=float, default=1e-5, help="L2 regularization strength")
    parser.add_argument('--learning_decay', type=bool, default=False, help='Learning rate decay')
    parser.add_argument('--csv_log', action='store_true',
                        # default=True,
                        help='Enable csv logging')

    # 聚合方式
    parser.add_argument('--averaging', type=str, default='ehhe_novel', help='Averaging strategy')

    # sage_flow ehhe
    # ins inswofinch ehhe ehhewofinch
    parser.add_argument('--public_dataset', type=str, default='pub_tyimagenet')  # pub_tyimagenet pub_fmnist pub_market1501,pub_usps, pub_svhn, pub_syn
    parser.add_argument('--pub_aug', type=str, default='weak', help='Public data augmentation')
    parser.add_argument('--public_len', type=int, default=2000)
    parser.add_argument('--public_batch_size', type=int, default=64)
    parser.add_argument('--public_lr', type=float, default=0.005)
    parser.add_argument('--public_epoch', type=int, default=20)
    # Malicious Clients
    parser.add_argument('--evils', type=str, default='PairFlip')  # PairFlip,SymFlip  AddNoise RandomNoise None lie_attack min_max min_max
    parser.add_argument('--bad_client_rate', type=float, default=0.2)  # 0.2 0.4 0.6
    parser.add_argument('--noise_data_rate', type=float, default=0.5)
    torch.set_num_threads(4)
    # add_management_args(parser)
    args = parser.parse_args()

    best = best_args[args.dataset][args.model]
    if args.beta in best:
        best = best[args.beta]
    else:
        best = best[0.5]
    for key, value in best.items():
        setattr(args, key, value)
    if args.seed is not None:
        set_random_seed(args.seed)
    # Trick Setting!
    local_lr_dict = {
        'fl_cifar10': 0.01,
        'fl_cifar100': 0.01,
        'other': 1e-2

    }
    communication_epoch_dict = {
        'fl_cifar10': 100,
        'fl_cifar100': 100,
        'other': 50
    }
    parti_num_dict = {
        'fl_cifar10': 10,
        'fl_cifar100': 10,
        'other': 20
    }

    if args.dataset in local_lr_dict:
        args.local_lr = local_lr_dict[args.dataset]
    else:
        args.local_lr = local_lr_dict['other']
    if args.dataset in communication_epoch_dict:
        args.communication_epoch = communication_epoch_dict[args.dataset]
    else:
        args.communication_epoch = communication_epoch_dict['other']
    if args.dataset in parti_num_dict:
        args.parti_num = parti_num_dict[args.dataset]
    else:
        args.parti_num = parti_num_dict['other']
    if args.seed is not None:
        set_random_seed(args.seed)
    if args.evils == 'None':
        del args.bad_client_rate
        del args.noise_data_rate
    return args


def main(args=None):
    if args is None:
        args = parse_args()

    args.conf_jobnum = str(uuid.uuid4())
    args.conf_timestamp = str(datetime.datetime.now())
    args.conf_host = socket.gethostname()

    priv_dataset = get_prive_dataset(args)
    backbones_list = priv_dataset.get_backbone(args.parti_num, None, model_name=args.model)

    model = get_model(backbones_list, args, priv_dataset.get_transform())
    # if args.averaging in ['ins', 'rel', 'insrel', 'twist', 'fl_trust', 'dnc', 'sage_flow']:
    publ_dataset = get_public_dataset(args)
    model.publicloader = publ_dataset.get_data_loaders()

    args.arch = model.nets_list[0].name
    print('{}_{}_{}_{}_{}_{}_{}_{}'.format(args.model, args.parti_num, args.dataset, args.beta, args.evils, args.averaging, args.communication_epoch, args.local_epoch))
    setproctitle.setproctitle('{}_{}_{}_{}_{}_{}_{}'.format(args.model, args.dataset, args.beta, args.evils, args.averaging, args.communication_epoch, args.local_epoch))
    train(model, priv_dataset, args)


if __name__ == '__main__':
    main()
