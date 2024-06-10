import pandas as pd
import os
import numpy as np
import csv

path = './data/' # data_old data_submission data
scenario = 'fl_cifar10'  # fl_cifar10,fl_mnist fl_cifar10
beta = 0.5  # 1.0 0.5
if scenario in ['fl_cifar10', 'fl_cifar100']:
    comm_epoch = 100
else:
    comm_epoch = 50
if scenario in ['fl_cifar10', 'fl_mnist']:
    local_lr = 0.01
else:
    local_lr = 0.1
local_epoch = 10  # 5 15
online_ratio = 1.0
public_batch = 64
column_mean_acc_list = ['method', 'paragroup'] + ['epoch' + str(i) for i in range(comm_epoch)] + ['MEAN', 'MAX']
# column_mean_acc_list = ['method', 'paragroup'] + ['epoch' + str(i) for i in range(comm_epoch)] + ['MEAN', 'MAX', 'C']
not_include_method = ['ins', 'inshe', 'inswofinch']
baseline = 'fedprox' # fedprox fedavg
envils = 'RandomNoise'  # PairFlip SymFlip RandomNoise AddNoise lie_attack min_max min_sum    None
envils_rato = '0.2'  # 0.2 0.4
bound_acc = 56.02 * 0.95


def load_acc_list(structure_path):
    acc_dict = {}
    experiment_index = 0
    for model in os.listdir(structure_path):
        if model != '' and model not in not_include_method:
            model_path = os.path.join(structure_path, model)
            if os.path.isdir(model_path):
                for para in os.listdir(model_path):
                    if para != 'weight.txt':
                        para_path = os.path.join(model_path, para)
                        args_path = para_path + '/args.csv'
                        args_pd = pd.read_table(args_path, sep=",")
                        args_pd = args_pd.loc[:, args_pd.columns]
                        args_comm_epoch = args_pd['communication_epoch'][0]
                        args_loca_epoch = args_pd['local_epoch'][0]
                        args_public_batch = args_pd['public_batch_size'][0]
                        args_lr = args_pd['local_lr'][0]
                        args_online_ratio = args_pd['online_ratio'][0]
                        if args_comm_epoch == comm_epoch and \
                                args_loca_epoch == local_epoch and \
                                args_public_batch == public_batch and\
                                args_lr == local_lr and \
                                args_online_ratio == online_ratio:
                            if os.path.exists(para_path + '/acc.csv'):
                                # if len(os.listdir(para_path)) != 1:
                                data = pd.read_table(para_path + '/acc.csv', sep=",")
                                data = data.loc[:, data.columns]
                                acc_value = data.values
                                times = acc_value.shape[0]
                                mean_acc_value = np.mean(acc_value, axis=0)
                                mean_acc_value = mean_acc_value.tolist()
                                mean_acc_value = [round(item, 2) for item in mean_acc_value]
                                max_acc_value = max(mean_acc_value)
                                last_acc_vale = mean_acc_value[-5:]
                                last_acc_vale = np.mean(last_acc_vale)

                                if last_acc_vale > bound_acc:
                                    epoch_result = list(filter(lambda k: k > bound_acc, mean_acc_value))[0]
                                    epoch_index = mean_acc_value.index(epoch_result)
                                else:
                                    epoch_index = 'Nan'

                                mean_acc_value.append(round(last_acc_vale, 3))
                                mean_acc_value.append(max_acc_value)
                                # mean_acc_value.append(epoch_index)
                                # acc_dict[experiment_index] = [model + str(times) ,
                                #                               para] + mean_acc_value
                                # public_len
                                acc_dict[experiment_index] = [model + str(times) + args_pd['public_dataset'][0] + str(args_pd['public_batch_size'][0]) + str(
                                    args_pd['public_len'][0]), para] + mean_acc_value
                                experiment_index += 1
    return acc_dict


if __name__ == '__main__':
    print('**************************************************************')
    scenario_path = os.path.join(path, scenario)
    print('Scenario: ' + scenario + ' Beta: ' + str(beta) + ' Robust: ' + envils + 'Ratio: ' + envils_rato)
    scenario_beta_path = os.path.join(scenario_path, str(beta))
    scenario_evils_path = os.path.join(scenario_beta_path, envils)
    scenario_evils_rato_path = os.path.join(scenario_evils_path, envils_rato)
    baseline_path = os.path.join(scenario_evils_rato_path, baseline)
    mean_acc_dict = load_acc_list(baseline_path)
    mean_df = pd.DataFrame(mean_acc_dict)
    mean_df = mean_df.T
    mean_df.columns = column_mean_acc_list
    # pd.set_option('display.max_columns', None)
    print(mean_df)
    mean_df.to_excel(os.path.join(scenario_evils_path, scenario + str(beta) + 'output.xls'), na_rep=True)
    print('**************************************************************')
