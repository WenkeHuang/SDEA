import pandas as pd
import  os
import numpy as np
import csv

path = './data/'
scenario = 'fl_cifar10' # fl_cifar10,fl_mnist fl_cifar10
beta = 0.5 # 1.0 0.5
if scenario in ['fl_cifar10','fl_cifar100']:
    comm_epoch = 100
else:
    comm_epoch = 50
if scenario in ['fl_cifar10','fl_mnist']:
    local_lr = 0.01
else:
    local_lr = 0.1
local_epoch = 10 # 5 15
online_ratio = 1.0
column_mean_acc_list = ['method', 'paragroup'] + ['epoch'+str(i) for i in range(comm_epoch)]+['MEAN','MAX','C']
not_include_method = ['moon']
baseline = 'fedprox'
envils = 'RandomNoise' # SymFlip None RandomNoise AddNoise PairFlip
envils_rato = '0.2' # 0.2 0.4

bound_acc = 56.02 * 0.95

def load_acc_list(structure_path):
    acc_dict = {}
    experiment_index = 0
    for model in os.listdir(structure_path):
        if model != '' and model not in not_include_method:
            model_path = os.path.join(structure_path, model)
            if os.path.isdir(model_path):
                for para in os.listdir(model_path):
                    para_path = os.path.join(model_path, para)
                    args_path = para_path+'/args.csv'
                    args_pd = pd.read_table(args_path,sep=",")
                    args_pd = args_pd.loc[:, args_pd.columns]
                    args_comm_epoch = args_pd['communication_epoch'][0]
                    args_loca_epoch = args_pd['local_epoch'][0]
                    args_lr = args_pd['local_lr'][0]
                    args_online_ratio = args_pd['online_ratio'][0]
                    if args_comm_epoch == comm_epoch and \
                            args_loca_epoch ==local_epoch \
                            and args_lr==local_lr and \
                            args_online_ratio ==online_ratio:
                        if os.path.exists(para_path + '/weight.txt'):
                            with open(para_path + '/weight.txt', 'r') as f:
                                lines = f.readlines()
                            data_dict = {}
                            for line in lines:
                                # 删除换行符并将以逗号分隔的字符串拆分为一个列表
                                line_list = line.strip().split(':')
                                item_list = line_list[1].split(',')
                                item_list = [float(num) for i, num in
                                                           enumerate(item_list[:])]
                                good = np.sum(item_list[:8])
                                bad = np.sum(item_list[-2:])
                                data_dict[line_list[0]] = [good,bad]
                                # data_dict[line_list[0]] = item_list
                            acc_dict[model+para]=data_dict
    '''
    获得最终epoch
    '''
    # for model in os.listdir(structure_path):
    #     if model != '' and model not in not_include_method:
    #         model_path = os.path.join(structure_path, model)
    #         if os.path.isdir(model_path):
    #             for para in os.listdir(model_path):
    #                 para_path = os.path.join(model_path, para)
    #                 args_path = para_path+'/args.csv'
    #                 args_pd = pd.read_table(args_path,sep=",")
    #                 args_pd = args_pd.loc[:, args_pd.columns]
    #                 args_comm_epoch = args_pd['communication_epoch'][0]
    #                 args_loca_epoch = args_pd['local_epoch'][0]
    #                 args_lr = args_pd['local_lr'][0]
    #                 args_online_ratio = args_pd['online_ratio'][0]
    #                 if args_comm_epoch == comm_epoch and \
    #                         args_loca_epoch ==local_epoch \
    #                         and args_lr==local_lr and \
    #                         args_online_ratio ==online_ratio:
    #                     if os.path.exists(para_path + '/weight.txt'):
    #                         print(model)
    #                         print(acc_dict[model]['96'])
    return acc_dict

if __name__=='__main__':
        print('**************************************************************')
        scenario_path = os.path.join(path, scenario)
        print('Scenario: ' + scenario+' Beta: '+ str(beta) + ' Robust: '+envils + 'Ratio: ' + envils_rato)
        scenario_beta_path = os.path.join(scenario_path, str(beta))
        scenario_evils_path = os.path.join(scenario_beta_path, envils)
        scenario_evils_rato_path = os.path.join(scenario_evils_path, envils_rato)
        baseline_path = os.path.join(scenario_evils_rato_path,baseline)
        mean_acc_dict = load_acc_list(baseline_path)
        mean_df = pd.DataFrame(mean_acc_dict)
        # mean_df = mean_df.T
        # mean_df.columns = column_mean_acc_list
        # pd.set_option('display.max_columns', None)
        print(mean_df)
        mean_df.to_excel(os.path.join(baseline_path+'weight.xls'), na_rep=True)
        print('**************************************************************')
