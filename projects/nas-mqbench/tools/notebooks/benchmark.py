import os

import collections
import numpy as np
import matplotlib.pyplot as plt

import nats_bench

######################GLOBAL SETTINGS#################################
WORK_DIR = os.path.dirname(os.path.abspath(__file__))
print('NATS-Bench version: {:}'.format(nats_bench.version()))
# Create the API for tologoy search space
nas_mqbench_pth = '/home/wangshiguang/mnodes/NAS-MQBench/mmrazor/work_dirs/NAS-MQ-Bench-NATS-tss'
# nas_mqbench_pth = '/alg-data/ftp-upload/private/wangshiguang/datasets/NATS/NATS-tss-v1_0-3ffb9-full'
api = nats_bench.create(nas_mqbench_pth, 'tss', fast_mode=True, verbose=True)
api.verbose = False
metric_marker_mapping = {
      'ori-test': '.',
      # 'ptq_per-tensor_w-minmax_a-minmax_nats': '*',
      # 'ptq_per-channel_w-minmax_a-minmax_nats': 'v',
      'mq_avg': '*',
      'deployability_index': 'v'
}
metric_on_set = 'ori-test'
evaluated_indexes = list(range(0, 1000))  #len(api)))
######################GLOBAL SETTINGS#################################


def find_best(api, dataset, metric_on_set, hp, fp32_random=777, mq_random=False, mq_args=None):

      def find_best_with_metric(dataset, metric, is_random):
            best_index = -1
            if isinstance(dataset, (list, tuple)):
                  assert isinstance(metric, (list, tuple))
                  assert len(dataset) == len(metric)
            for arch_index in evaluated_indexes:
                  # print(arch_index, hp, is_random, metric, dataset)
                  api._prepare_info(arch_index)
                  arch_info = api.arch2infos_dict[arch_index][hp]
                  if isinstance(dataset, (list, tuple)):
                        accuracy = []
                        for ds, mc in zip(dataset, metric):
                              xinfo = arch_info.get_metrics(ds, mc, is_random=is_random)
                              accuracy.append(xinfo['accuracy'])
                        accuracy = np.mean(accuracy)
                  else:
                        xinfo = arch_info.get_metrics(dataset, metric, is_random=is_random)
                        accuracy = xinfo['accuracy']
                  if best_index == -1:
                        best_index, highest_accuracy = arch_index, accuracy
                  elif highest_accuracy < accuracy:
                        best_index, highest_accuracy = arch_index, accuracy
            return best_index, highest_accuracy

      results = {}
      # 1. Find the best architecture with fp32 setting
      results[metric_on_set] = find_best_with_metric(dataset, metric_on_set, fp32_random)
      # 2. Find the best architecture with quantize settings
      mq_datasets = []
      if mq_args is not None:
        if not isinstance(mq_args, (list, tuple)):
            mq_args = [mq_args]
        for mq in mq_args:
            mq_dataset = dataset + '_' + mq
            results[mq] = find_best_with_metric(mq_dataset, mq, mq_random)
            mq_datasets.append(mq_dataset)            
      results['mq_avg'] = find_best_with_metric(mq_datasets, mq_args, mq_random)
      return results

def get_sorted_indices(arr, reverse=True):
      sorted_arr = sorted(arr, reverse=reverse)
      sorted_indices = [sorted_arr.index(num) for num in arr]
      return sorted_indices

def get_metrics(api, dataset, metric_on_set, hp, fp32_random=777, mq_random=False, mq_args=None):

      def _get_metrics(dataset, metric, is_random):
            metric_list = []
            if isinstance(dataset, (list, tuple)):
                  assert isinstance(metric, (list, tuple))
                  assert len(dataset) == len(metric)            
            for arch_index in evaluated_indexes:
                  api._prepare_info(arch_index)
                  arch_info = api.arch2infos_dict[arch_index][hp]
                  if isinstance(dataset, (list, tuple)):
                        accuracy = []
                        for ds, mc in zip(dataset, metric):
                              xinfo = arch_info.get_metrics(ds, mc, is_random=is_random)
                              accuracy.append(xinfo['accuracy'])
                        accuracy = np.mean(accuracy)
                  else:                  
                        xinfo = arch_info.get_metrics(dataset, metric, is_random=is_random)
                        accuracy = xinfo['accuracy']
                  metric_list.append(accuracy)
            return metric_list

      results = collections.OrderedDict()
      results[metric_on_set] = _get_metrics(dataset, metric_on_set, fp32_random)
      mq_datasets = []
      if mq_args is not None:
            for mq in mq_args:
                  mq_dataset = dataset + '_' + mq
                  results[mq] = _get_metrics(mq_dataset, mq, mq_random)
                  mq_datasets.append(mq_dataset)
      results['mq_avg'] = _get_metrics(mq_datasets, mq_args, mq_random)
      results['deployability_index'] = list(20 * (np.array(results[metric_on_set]) - np.array(results['mq_avg'])))
      # results['deployability_index'] = list((np.array(results[metric_on_set]) - np.array(results['mq_avg'])) / np.array(results[metric_on_set]))
      return results

def get_computation_cost(api, dataset, indicator='flops'):
      cost_list = []
      for arch_index in evaluated_indexes:
            api._prepare_info(arch_index)
            arch_info = api.arch2infos_dict[arch_index][hp]
            if isinstance(dataset, (list, tuple)):
                  cost = []
                  arch_info.get_compute_costs(dataset)
                  for ds in dataset:
                        xinfo = arch_info.get_compute_costs(ds)
                        cost.append(xinfo[indicator])
                  cost = np.mean(cost)
            else:
                  xinfo = arch_info.get_compute_costs(dataset)
                  cost = xinfo[indicator]
            cost_list.append(cost)
      return cost_list


# step1: find the best architecture
# # hp = '12'
# # mq_args = 'ptq_per-tensor_minmax_nats'   # ptq_(quantize_granularity)_(calib)_(search_space)
# # results = find_best(api, dataset=dataset, metric_on_set=metric_on_set, hp=hp, mq_args=mq_args)
# # print(f'{nats_bench.api_utils.time_string()} The best architecture on {dataset} ' +
# #       f'with hp: {hp} training is: {results}')

datasets = ['cifar10', 'cifar100', 'ImageNet16-120']
mq_random = False
hp = '200'
for dataset in datasets:
      mq_args = ('ptq_per-tensor_w-minmax_a-minmax_nats', 'ptq_per-channel_w-minmax_a-minmax_nats')   # ptq_(quantize_granularity)_(calib)_(search_space)
      # best_arch_index, highest_valid_accuracy = api.find_best(dataset=dataset, metric_on_set=metric_on_set, hp=hp)
      results = find_best(api, dataset=dataset, metric_on_set=metric_on_set, hp=hp, mq_random=mq_random, mq_args=mq_args)
      print(f'{nats_bench.api_utils.time_string()} The best architecture on {dataset} ' +
            f'with hp: {hp}, seed: {mq_random} training is: {results}')

# step2: relative rank
# 2.1 architecture rank between float32 and model quantization settings.
hp = '200'
mq_random = False
datasets = ['cifar10', 'cifar100', 'ImageNet16-120']

for idx, dataset in enumerate(datasets):
      mq_args = ('ptq_per-tensor_w-minmax_a-minmax_nats', 'ptq_per-channel_w-minmax_a-minmax_nats')
      results = get_metrics(api, dataset=dataset, metric_on_set=metric_on_set, hp=hp, mq_args=mq_args)
      sorted_fp32 = get_sorted_indices(results[metric_on_set])
      fig, ax = plt.subplots()
      for jdx, metric in enumerate(results):
            if metric not in metric_marker_mapping:
                  continue
            sorted_rst = get_sorted_indices(results[metric])
            ax.scatter(sorted_fp32, sorted_rst, alpha=0.5, label=metric,
                       marker=metric_marker_mapping[metric])

      ax.set_xlabel('Architecture ranking for float32')
      ax.set_ylabel('Architecture ranking')
      ax.set_title(f'Architecture ranking on {dataset}')

      plt.tight_layout()
      plt.legend()
      plt.show()
      fig.savefig(f'{WORK_DIR}/architecture_rank_mq_{dataset}.png')

# 2.2. architecture rank between different hps
hps_fp32random = [('12', 111), ('200', 777)]
mq_random = False
dataset = 'cifar10'

fig, ax = plt.subplots()
for idx, (hp, fp32_random) in enumerate(hps_fp32random):
      mq_args = ('ptq_per-tensor_w-minmax_a-minmax_nats', )
      results = get_metrics(api, dataset=dataset, metric_on_set=metric_on_set, hp=hp,
                            fp32_random=fp32_random, mq_args=mq_args)
      if idx == 0:
            sorted_fp32 = get_sorted_indices(results[metric_on_set])

      for jdx, metric in enumerate(results):
            if metric not in metric_marker_mapping:
                  continue            
            sorted_rst = get_sorted_indices(results[metric])
            ax.scatter(sorted_fp32, sorted_rst, alpha=0.5, label=f'{hp}_{metric}',
                       marker=metric_marker_mapping[metric])

ax.set_xlabel(f'Architecture ranking for float32 and hp {hps_fp32random[0][0]}')
ax.set_ylabel('Architecture ranking')
ax.set_title(f'Architecture ranking between different hps')

plt.tight_layout()
plt.legend()
plt.show()
fig.savefig(f'{WORK_DIR}/architecture_rank_hp_{dataset}.png')


# 3. an overview of architecutre performance (float32, mq, and deployability_index) with respect to parameters / FLOPs.
# 3.1 respect to parameters
# 3.2 respect to FLOPs
hp = '200'
mq_random = False
datasets = ['cifar10', 'cifar100', 'ImageNet16-120']
cost_indicators = ['flops', 'params']

for indicator in cost_indicators:
      for idx, dataset in enumerate(datasets):
            mq_args = ('ptq_per-tensor_w-minmax_a-minmax_nats', 'ptq_per-channel_w-minmax_a-minmax_nats')
            costs = get_computation_cost(api, dataset, indicator=indicator)
            results = get_metrics(api, dataset=dataset, metric_on_set=metric_on_set, hp=hp, mq_args=mq_args)
            fig, ax = plt.subplots()
            for jdx, metric in enumerate(results):
                  if metric not in metric_marker_mapping:
                        continue
                  ax.scatter(costs, results[metric], alpha=0.5, label=metric,
                        marker=metric_marker_mapping[metric])

            ax.set_xlabel(f'{indicator}')
            ax.set_ylabel('Architecture Accuracy')
            ax.set_title(f'Results of architecture accuracy on {dataset}')

            plt.tight_layout()
            plt.legend()
            plt.show()
            fig.savefig(f'{WORK_DIR}/architecture_results_{indicator}_{dataset}.png')
