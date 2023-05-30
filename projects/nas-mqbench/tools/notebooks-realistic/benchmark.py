import os

import collections
import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import kendalltau, spearmanr
from sklearn.metrics import mean_squared_error
from mmengine import fileio
from mmrazor.structures import Candidates

######################GLOBAL SETTINGS#################################
WORK_DIR = os.path.dirname(os.path.abspath(__file__))
# Create the API for realistic search space
nas_mqbench_pth = 'work_dirs/ptq_per-tensor_w-minmax_a-minmax_bignas_resnet18_8xb256_in1k_calib32xb16_flops0-1000/search_epoch_5.pkl'
api = fileio.load(nas_mqbench_pth)['export_candidates']
metric_marker_mapping = {
      'score': ('.', 'red'),
      'per-tensor_w-minmax_a-minmax': ('^', 'yellow'),
      'per-channel_w-minmax_a-minmax': ('v', 'purple'),
      'mq_avg': ('*', 'green'),
      'deployability_index': ('o', 'blue')
}
Candidates._indicators = tuple(set(Candidates._indicators + tuple(metric_marker_mapping.keys())))
metric_on_set = 'score'
EVALUATED_INDEXES = list(range(0, 25))  #len(api)))
######################GLOBAL SETTINGS#################################


def find_best(api, metric_on_set, mq_args=None,
              evaluated_indexes=None):

      evaluated_indexes = evaluated_indexes or EVALUATED_INDEXES

      def find_best_with_metric(metric):
            metric_list = []
            if isinstance(metric, (list, tuple)):
                  metric_list = []
                  for mc in metric:
                        metric_list.append(api.resources(mc))
                  metric_list = np.array(metric_list).mean(axis=0)
            else:
                  metric_list = np.array(api.resources(metric))
            # metric_list = list(metric_list[evaluated_indexes])
            sorted_idx = metric_list.argsort()[:-1]
            best_idx = 0
            while(sorted_idx[best_idx] not in evaluated_indexes):
                  best_idx += 1
            return sorted_idx[best_idx], metric_list[sorted_idx[best_idx]]

      results = {}
      # 1. Find the best architecture with fp32 setting
      results[metric_on_set] = find_best_with_metric(metric_on_set)
      # 2. Find the best architecture with quantize settings
      if mq_args is not None:
        if not isinstance(mq_args, (list, tuple)):
            mq_args = [mq_args]
        for mq in mq_args:
            mq_dataset = dataset + '_' + mq
            results[mq] = find_best_with_metric(mq)
      results['mq_avg'] = find_best_with_metric(mq_args)
      return results

def get_sorted_indices(arr, reverse=True):
      sorted_arr = sorted(arr, reverse=reverse)
      sorted_indices = [sorted_arr.index(num) for num in arr]
      return sorted_indices

def get_metrics(api, metric_on_set, mq_args=None, evaluated_indexes=None):
      evaluated_indexes = evaluated_indexes or EVALUATED_INDEXES

      def _get_metrics(metric):
            metric_list = []
            if isinstance(metric, (list, tuple)):
                  metric_list = []
                  for mc in metric:
                        metric_list.append(api.resources(mc))
                  metric_list = np.array(metric_list).mean(axis=0)
            else:
                  metric_list = np.array(api.resources(metric))
            metric_list = list(metric_list[evaluated_indexes])
            return metric_list

      results = collections.OrderedDict()
      results[metric_on_set] = _get_metrics(metric_on_set)
      if mq_args is not None:
            for mq in mq_args:
                  results[mq] = _get_metrics(mq)
      results['mq_avg'] = _get_metrics(mq_args)
      # results['deployability_index'] = list(20 * (np.array(results[metric_on_set]) - np.array(results['mq_avg'])))
      results['deployability_index'] = list(1 - (np.array(results[metric_on_set]) - np.array(results['mq_avg'])) / np.array(results[metric_on_set]))
      return results


def get_computation_cost(api, indicator='flops', constraints=None,
                         evaluated_indexes=None):
      evaluated_indexes = evaluated_indexes or EVALUATED_INDEXES

      metric_list = np.array(api.resources(indicator))
      metric_list = list(metric_list[evaluated_indexes])
      return metric_list

# 1 an overview of architecutre performance (float32, mq, and deployability_index).
# 1.1 respect to parameters and FLOPs, respectively
hp = '200'
mq_random = False
datasets = ['in1k']
results = dict()
cost_indicators = ['flops', 'params']
for indicator in cost_indicators:
      for idx, dataset in enumerate(datasets):
            mq_args = ('per-tensor_w-minmax_a-minmax', )
            costs = get_computation_cost(api, indicator)
            results = get_metrics(api, metric_on_set=metric_on_set, mq_args=mq_args)
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

# 1.2 find best
datasets = ['in1k']
hps_fp32random = [('200', 777)]   # [('12', 111), ('200', 777)]
mq_random = False
for dataset in datasets:
      for idx, (hp, fp32_random) in enumerate(hps_fp32random):
            mq_args = ('per-tensor_w-minmax_a-minmax', )
            results = find_best(api, metric_on_set=metric_on_set, mq_args=mq_args)
      print(f'The best architecture on {dataset} ' +
            f'with hp: {hp}, seed: {mq_random} training is: {results}')

# 1.3 mdiff(Mean of difference) and xrange between float32 and mq
hp = '200'
mq_random = False
datasets = ['in1k']
for idx, dataset in enumerate(datasets):
      mq_args = ('per-tensor_w-minmax_a-minmax', )
      results = get_metrics(api, metric_on_set=metric_on_set, mq_args=mq_args)
      for jdx, metric in enumerate(results):
            xrange = np.array(results[metric_on_set]) - np.array(results[metric])
            max_idx, min_idx = np.argmax(xrange), np.argmin(xrange)
            print(f"Dataset {dataset}, Xrange between {metric_on_set} and {metric}: "
                  f"MDiff, Max, Min: {xrange.mean():.2f}, "
                  f"{xrange[max_idx]:.2f}[{results[metric_on_set][max_idx]:.2f} vs {results[metric][max_idx]:.2f}], "
                  f"{xrange[min_idx]:.2f}[{results[metric_on_set][min_idx]:.2f} vs {results[metric][min_idx]:.2f}]")

# 2 architecture ranking
# 2.1 relative ranking between float32 and model quantization settings
# 2.2 Kendall and Spearmanr Rank
hp = '200'
mq_random = False
datasets = ['in1k']
for idx, dataset in enumerate(datasets):
      mq_args = ('per-tensor_w-minmax_a-minmax', )
      results = get_metrics(api, metric_on_set=metric_on_set, mq_args=mq_args)
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

# 2.3 relative ranking between sliced subnsts and retraining subnets
