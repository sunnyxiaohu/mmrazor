import os

import collections
import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import kendalltau, spearmanr
from sklearn.metrics import mean_squared_error

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
      'ori-test': ('.', 'red'),
      # 'ptq_per-tensor_w-minmax_a-minmax_nats': ('^', 'yellow'),
      # 'ptq_per-channel_w-minmax_a-minmax_nats': ('v', 'purple'),
      'mq_avg': ('*', 'green'),
      'deployability_index': ('o', 'blue')
}
metric_on_set = 'ori-test'
EVALUATED_INDEXES = list(range(0, len(api)))
######################GLOBAL SETTINGS#################################


def find_best(api, dataset, metric_on_set, hp, fp32_random=777, mq_random=False, mq_args=None,
              evaluated_indexes=None):

      evaluated_indexes = evaluated_indexes or EVALUATED_INDEXES

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

def get_metrics(api, dataset, metric_on_set, hp, fp32_random=777, mq_random=False, mq_args=None,
                evaluated_indexes=None):
      evaluated_indexes = evaluated_indexes or EVALUATED_INDEXES

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
      # results['deployability_index'] = list(20 * (np.array(results[metric_on_set]) - np.array(results['mq_avg'])))
      results['deployability_index'] = list(1 - (np.array(results[metric_on_set]) - np.array(results['mq_avg'])) / np.array(results[metric_on_set]))
      return results

def get_computation_cost(api, dataset, indicator='flops', constraints=None,
                         evaluated_indexes=None):
      evaluated_indexes = evaluated_indexes or EVALUATED_INDEXES

      cost_list = []
      constraints_indexs = []
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
            if constraints is not None and (cost > constraints[1] or cost < constraints[0]):
                  continue
            cost_list.append(cost)
            constraints_indexs.append(arch_index)
      return cost_list, constraints_indexs


# 1 an overview of architecutre performance (float32, mq, and deployability_index).
# 1.1 respect to parameters and FLOPs, respectively
hp = '200'
mq_random = False
datasets = ['cifar10', 'cifar100', 'ImageNet16-120']
cost_indicators = ['flops', 'params']
for indicator in cost_indicators:
      for idx, dataset in enumerate(datasets):
            mq_args = ('ptq_per-tensor_w-minmax_a-minmax_nats', 'ptq_per-channel_w-minmax_a-minmax_nats')
            costs, _ = get_computation_cost(api, dataset, indicator=indicator)
            results = get_metrics(api, dataset=dataset, metric_on_set=metric_on_set, hp=hp, mq_args=mq_args)

            fig, ax = plt.subplots()
            metric = 'deployability_index'
            ax.scatter(costs, results.pop(metric), alpha=0.5,
                  marker=metric_marker_mapping[metric][0],
                  color=metric_marker_mapping[metric][1])
            ax.set_xlabel(f'#{indicator}(M)')
            ax.set_ylabel('Architecture deploy index')
            ax.set_title(f'Results of architecture deploy index on {dataset}')
            plt.tight_layout()
            plt.show()
            fig.savefig(f'{WORK_DIR}/architecture_deploy_{indicator}_{dataset}.png')

            fig, ax = plt.subplots()
            for jdx, metric in enumerate(results):
                  if metric not in metric_marker_mapping:
                        continue
                  ax.scatter(costs, results[metric], alpha=0.5, label=metric,
                        marker=metric_marker_mapping[metric][0],
                        color=metric_marker_mapping[metric][1])
            ax.set_xlabel(f'#{indicator}(M)')
            ax.set_ylabel('Architecture accuracy')
            ax.set_title(f'Results of architecture accuracy on {dataset}')
            plt.tight_layout()
            plt.legend()
            plt.show()
            fig.savefig(f'{WORK_DIR}/architecture_results_{indicator}_{dataset}.png')
plt.close()

# 1.2 find best
datasets = ['cifar10', 'cifar100', 'ImageNet16-120']
hps_fp32random = [('200', 777)]   # [('12', 111), ('200', 777)]
mq_random = False
for dataset in datasets:
      for idx, (hp, fp32_random) in enumerate(hps_fp32random):
            mq_args = ('ptq_per-tensor_w-minmax_a-minmax_nats', 'ptq_per-channel_w-minmax_a-minmax_nats')
            results = find_best(api, dataset=dataset, metric_on_set=metric_on_set, hp=hp,
                                fp32_random=fp32_random, mq_random=mq_random, mq_args=mq_args)
      print(f'{nats_bench.api_utils.time_string()} The best architecture on {dataset} ' +
            f'with hp: {hp}, seed: {mq_random} training is: {results}')

# 1.3 mdiff(Mean of difference) and xrange between float32 and mq
hp = '200'
mq_random = False
datasets = ['cifar10', 'cifar100', 'ImageNet16-120']
for idx, dataset in enumerate(datasets):
      mq_args = ('ptq_per-tensor_w-minmax_a-minmax_nats', 'ptq_per-channel_w-minmax_a-minmax_nats')
      results = get_metrics(api, dataset=dataset, metric_on_set=metric_on_set, hp=hp, mq_args=mq_args)
      for jdx, metric in enumerate(results):
            xrange = np.array(results[metric_on_set]) - np.array(results[metric])
            max_idx, min_idx = np.argmax(xrange), np.argmin(xrange)
            print(f"Dataset {dataset}, Xrange between {metric_on_set} and {metric}: "
                  f"MDiff, Max, Min: {xrange.mean():.2f}, "
                  f"{xrange[max_idx]:.2f}[{results[metric_on_set][max_idx]:.2f} vs {results[metric][max_idx]:.2f}], "
                  f"{xrange[min_idx]:.2f}[{results[metric_on_set][min_idx]:.2f} vs {results[metric][min_idx]:.2f}]")

# 2 architecture ranking
# 2.1 Kendall and Spearmanr Rank
# 2.2 relative ranking between float32 and model quantization settings
hp = '200'
mq_random = False
datasets = ['cifar10', 'cifar100', 'ImageNet16-120']
for idx, dataset in enumerate(datasets):
      mq_args = ('ptq_per-tensor_w-minmax_a-minmax_nats', 'ptq_per-channel_w-minmax_a-minmax_nats')
      results = get_metrics(api, dataset=dataset, metric_on_set=metric_on_set, hp=hp, mq_args=mq_args)
      # Kendall and Spearmanr Rank.
      num_rows, num_cols = len(results), len(results)
      k_corr = np.zeros((num_rows, num_cols))
      s_corr = np.zeros((num_rows, num_cols))
      for i, i_metric in enumerate(results):
            i_sorted = results[i_metric]  # get_sorted_indices(results[i_metric])
            for j, j_metric in enumerate(results):
                  j_sorted = results[j_metric]  # get_sorted_indices(results[j_metric])
                  correlation, p_value = kendalltau(i_sorted, j_sorted)
                  k_corr[i, j] = correlation
                  correlation, p_value = spearmanr(i_sorted, j_sorted)
                  s_corr[i, j] = correlation
      fig, ax = plt.subplots()                  
      im = ax.imshow(k_corr, cmap='coolwarm', vmin=0.5, vmax=1)
      for i in range(num_rows):
            for j in range(num_cols):
                  text = ax.text(j, i, f'{k_corr[i, j]:.2f}',
                                 ha='center', va='center', color='w')
      plt.colorbar(im)
      plt.title('Kendall Rank Matrix')
      plt.show()
      fig.savefig(f'{WORK_DIR}/kendall_rank_matrix_{dataset}.png')
      fig, ax = plt.subplots()      
      im = ax.imshow(s_corr, cmap='coolwarm', vmin=0.5, vmax=1)
      for i in range(num_rows):
            for j in range(num_cols):
                  text = ax.text(j, i, f'{s_corr[i, j]:.2f}',
                                 ha='center', va='center', color='w')      
      plt.colorbar(im)
      plt.title('Spearmanr Rank Matrix')
      plt.show()
      fig.savefig(f'{WORK_DIR}/spearmanr_rank_matrix_{dataset}.png')      

      sorted_fp32 = get_sorted_indices(results[metric_on_set])
      fig, ax = plt.subplots()
      metric = 'deployability_index'
      sorted_rst = get_sorted_indices(results.pop(metric))
      ax.scatter(sorted_fp32, sorted_rst, alpha=0.5,
            marker=metric_marker_mapping[metric][0],
            color=metric_marker_mapping[metric][1])
      ax.set_xlabel('Architecture ranking for float32 accuracy')
      ax.set_ylabel('Architecture ranking for deploy index')
      ax.set_title(f'Architecture ranking deploy index on {dataset}')
      plt.tight_layout()
      plt.show()
      fig.savefig(f'{WORK_DIR}/architecture_rank_deploy_{indicator}_{dataset}.png')
      
      fig, ax = plt.subplots()
      for jdx, metric in enumerate(results):
            if metric not in metric_marker_mapping:
                  continue
            sorted_rst = get_sorted_indices(results[metric])
            ax.scatter(sorted_fp32, sorted_rst, alpha=0.5, label=metric,
                       marker=metric_marker_mapping[metric][0],
                       color=metric_marker_mapping[metric][1])
      ax.set_xlabel('Architecture ranking for float32 accuracy')
      ax.set_ylabel('Architecture ranking accuracy')
      ax.set_title(f'Architecture ranking accuracy on {dataset}')
      plt.tight_layout()
      plt.legend()
      plt.show()
      fig.savefig(f'{WORK_DIR}/architecture_rank_mq_{dataset}.png')
plt.close()

# 2.3 relative ranking between different hps
hps_fp32random = [('12', 111), ('200', 777)]
mq_random = False
datasets = ['cifar10', 'ImageNet16-120']
for idx, dataset in enumerate(datasets):
      fig, ax = plt.subplots()
      for jdx, (hp, fp32_random) in enumerate(hps_fp32random):
            mq_args = ('ptq_per-tensor_w-minmax_a-minmax_nats', 'ptq_per-channel_w-minmax_a-minmax_nats')
            results = get_metrics(api, dataset=dataset, metric_on_set=metric_on_set, hp=hp,
                                  fp32_random=fp32_random, mq_args=mq_args)
            metric = 'deployability_index'
            results.pop(metric)
            if jdx == 0:
                  sorted_fp32 = get_sorted_indices(results[metric_on_set])

            for jdx, metric in enumerate(results):
                  if metric not in metric_marker_mapping:
                        continue            
                  sorted_rst = get_sorted_indices(results[metric])
                  ax.scatter(sorted_fp32, sorted_rst, alpha=0.5, label=f'{hp}_{metric}')
                        #      marker=metric_marker_mapping[metric][0],
                        #      color=metric_marker_mapping[metric][1])

      ax.set_xlabel(f'Architecture ranking for float32 and hp {hps_fp32random[0][0]}')
      ax.set_ylabel('Architecture ranking')
      ax.set_title(f'Architecture ranking between different hps on dataset {dataset}')

      plt.tight_layout()
      plt.legend()
      plt.show()
      fig.savefig(f'{WORK_DIR}/architecture_rank_hp_{dataset}.png')
plt.close()