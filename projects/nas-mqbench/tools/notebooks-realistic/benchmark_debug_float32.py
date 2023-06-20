import os
import glob
from copy import deepcopy
import collections
import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import kendalltau, spearmanr
from sklearn.metrics import mean_squared_error
from mmengine import fileio
from mmrazor.structures import Candidates

######################GLOBAL SETTINGS#################################
WORK_DIR = os.path.dirname(os.path.abspath(__file__))
EVALUATED_INDEXES = list(range(0, 100))  # len(api)))
# Create the API for realistic search space
nas_mqbench_pth = [
    'work_dirs/bignas_resnet18_benchmark/done_logs/ptq_per-channel_w-minmax_a-minmax_bignas_resnet18_8xb256_in1k_calib32xb16_flops0-1000/search_epoch_5.pkl',
    'work_dirs/bignas_resnet18_benchmark/done_logs/ptq_per-channel_w-minmax_a-minmax_bignas_resnet18_8xb256_in1k_calib32xb16_flops1000-2000/search_epoch_5.pkl',
    'work_dirs/bignas_resnet18_benchmark/done_logs/ptq_per-channel_w-minmax_a-minmax_bignas_resnet18_8xb256_in1k_calib32xb16_flops2000-3000/search_epoch_5.pkl',
    'work_dirs/bignas_resnet18_benchmark/done_logs/ptq_per-channel_w-minmax_a-minmax_bignas_resnet18_8xb256_in1k_calib32xb16_flops3000-4000/search_epoch_5.pkl',
]
indexes_per_pth = len(EVALUATED_INDEXES) // len(nas_mqbench_pth)
api = Candidates()
for pth in nas_mqbench_pth:
      _api = fileio.load(pth)['export_candidates'][:indexes_per_pth]
      api.extend(_api)
metric_marker_mapping = {
      'score': ('.', 'red'),
      'per-tensor_w-minmax_a-minmax': ('^', 'yellow'),
      'per-channel_w-minmax_a-minmax': ('v', 'purple'),
      'mq_avg': ('*', 'green'),
      'deployability_index': ('o', 'blue'),
      'retrained.score': ('.', 'red'),
      # 'retrained.per-tensor_w-minmax_a-minmax': ('^', 'yellow'),
      # 'retrained.per-channel_w-minmax_a-minmax': ('v', 'purple'),
      'retrained.mq_avg': ('*', 'green'),
      'retrained.deployability_index': ('o', 'blue'),
      'sliced.score': ('.', 'red'),
      # 'sliced.per-tensor_w-minmax_a-minmax': ('^', 'yellow'),
      # 'sliced.per-channel_w-minmax_a-minmax': ('v', 'purple'),
      'sliced.mq_avg': ('*', 'green'),
      'sliced.deployability_index': ('o', 'blue')        
}
Candidates._indicators = tuple(set(Candidates._indicators + tuple(metric_marker_mapping.keys())))
metric_on_set = 'score'
hp = '200'
mq_random = False
base_metric = f'retrained.{metric_on_set}'

######################GLOBAL SETTINGS#################################
nas_mqbench_pth2 = [
    'work_dirs/bignas_resnet18_benchmark/done_logs/ptq_base_bignas_resnet18_8xb256_in1k_flops0-1000',
    'work_dirs/bignas_resnet18_benchmark/done_logs/ptq_base_bignas_resnet18_8xb256_in1k_flops1000-2000',
    'work_dirs/bignas_resnet18_benchmark/done_logs/ptq_base_bignas_resnet18_8xb256_in1k_flops2000-3000',
    'work_dirs/bignas_resnet18_benchmark/done_logs/ptq_base_bignas_resnet18_8xb256_in1k_flops3000-4000',
]
assert len(nas_mqbench_pth) == len(nas_mqbench_pth2)
api2 = Candidates()
for i, pth in enumerate(nas_mqbench_pth2):
      for j in range(indexes_per_pth):
            subnet_folder = pth + f'/subnet{j}'
            # float32
            xsubnet_folder = os.path.join(subnet_folder, 'float32')
            exp_folder = [f for f in os.listdir(xsubnet_folder) if os.path.isdir(os.path.join(xsubnet_folder, f))]
            assert len(exp_folder) == 1, f'Get multiple exp_folder: {xsubnet_folder}'
            exp_folder = exp_folder[0]
            filename = os.path.join(xsubnet_folder, exp_folder, 'vis_data', f'{exp_folder}.json')
            results = eval(open(filename).readlines()[-1])
            subnet_yaml = glob.glob(f'{pth}/subnet_{j}_*.yaml')[0]
            subnet_cfg = fileio.load(subnet_yaml)
            assert api.subnets[i*indexes_per_pth + j] == subnet_cfg
            api2.append(deepcopy(api[i*indexes_per_pth + j]))
            api2.set_resource(i*indexes_per_pth + j, results['accuracy/top1'], key_indicator='score')

            mq_args = ('per-tensor_w-minmax_a-minmax', 'per-channel_w-minmax_a-minmax')
            for mq in mq_args:
                  xsubnet_folder = os.path.join(subnet_folder, mq)
                  exp_folder = [f for f in os.listdir(xsubnet_folder) if os.path.isdir(os.path.join(xsubnet_folder, f))][0]
                  filename = os.path.join(xsubnet_folder, exp_folder, 'vis_data', f'{exp_folder}.json')
                  results = eval(open(filename).readlines()[-1])
                  subnet_yaml = glob.glob(f'{pth}/subnet_{j}_*.yaml')[0]
                  subnet_cfg = fileio.load(subnet_yaml)
                  assert api.subnets[i*indexes_per_pth + j] == subnet_cfg
                  api2.set_resource(i*indexes_per_pth + j, results['accuracy/top1'], key_indicator=mq)

######################################################################

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
datasets = ['in1k']
results = dict()
cost_indicators = ['flops', 'params']
for indicator in cost_indicators:
      for idx, dataset in enumerate(datasets):
            mq_args = ('per-tensor_w-minmax_a-minmax', 'per-channel_w-minmax_a-minmax')
            costs = get_computation_cost(api, indicator)
            results1 = get_metrics(api, metric_on_set=metric_on_set, mq_args=mq_args)
            results2 = get_metrics(api2, metric_on_set=metric_on_set, mq_args=mq_args)
            from mmrazor.models.utils import add_prefix
            results = collections.OrderedDict()
            results.update(add_prefix(results2, 'retrained'))
            results.update(add_prefix(results1, 'sliced'))
            results = collections.OrderedDict(filter(lambda x: x[0] in metric_marker_mapping, results.items()))            

            metric = 'deployability_index'
            if f'retrained.{metric}' in results:
                  fig, ax = plt.subplots()
                  ax.scatter(costs, results.pop(f'retrained.{metric}'), alpha=0.5, label=f'retrained.{metric}')
                  ax.scatter(costs, results.pop(f'sliced.{metric}'), alpha=0.5, label=f'sliced.{metric}')
                  ax.set_xlabel(f'#{indicator}(M)')
                  ax.set_ylabel('Architecture deploy index')
                  ax.set_title(f'Results of architecture deploy index on {dataset}')
                  plt.tight_layout()
                  plt.legend()
                  plt.show()
                  fig.savefig(f'{WORK_DIR}/architecture_deploy_{indicator}_{dataset}.png')

            fig, ax = plt.subplots()
            for jdx, metric in enumerate(results):
                  if metric not in metric_marker_mapping:
                        continue
                  ax.scatter(costs, results[metric], alpha=0.5, label=metric)
                        # marker=metric_marker_mapping[metric][0],
                        # color=metric_marker_mapping[metric][1])
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
            mq_args = ('per-tensor_w-minmax_a-minmax', 'per-channel_w-minmax_a-minmax')
            results = find_best(api, metric_on_set=metric_on_set, mq_args=mq_args)
      print(f'The best architecture on {dataset} ' +
            f'with hp: {hp}, seed: {mq_random} training is: {results}')

# 1.3 mdiff(Mean of difference) and xrange between float32 and mq
datasets = ['in1k']
for idx, dataset in enumerate(datasets):
      mq_args = ('per-tensor_w-minmax_a-minmax', 'per-channel_w-minmax_a-minmax')
      # results = get_metrics(api, metric_on_set=metric_on_set, mq_args=mq_args)
      results1 = get_metrics(api, metric_on_set=metric_on_set, mq_args=mq_args)
      results2 = get_metrics(api2, metric_on_set=metric_on_set, mq_args=mq_args)
      from mmrazor.models.utils import add_prefix
      results = collections.OrderedDict()
      results.update(add_prefix(results2, 'retrained'))
      results.update(add_prefix(results1, 'sliced'))      
      # results = collections.OrderedDict(filter(lambda x: x[0] in metric_marker_mapping, results.items()))
      for bmetric in ['retrained.score', 'sliced.score', 'retrained.per-channel_w-minmax_a-minmax']:
            for jdx, metric in enumerate(results):
                  xrange = np.array(results[bmetric]) - np.array(results[metric])
                  max_idx, min_idx = np.argmax(xrange), np.argmin(xrange)
                  print(f"Dataset {dataset}, Xrange between {bmetric} and {metric}: "
                        f"MDiff, Max, Min: {xrange.mean():.2f}, "
                        f"{xrange[max_idx]:.2f}[{results[bmetric][max_idx]:.2f} vs {results[metric][max_idx]:.2f}], "
                        f"{xrange[min_idx]:.2f}[{results[bmetric][min_idx]:.2f} vs {results[metric][min_idx]:.2f}]")

# 2 architecture ranking
# 2.1 Kendall and Spearmanr Rank
# 2.2 relative ranking between float32 and model quantization settings
datasets = ['in1k']
for idx, dataset in enumerate(datasets):
      mq_args = ('per-tensor_w-minmax_a-minmax', 'per-channel_w-minmax_a-minmax')
      results1 = get_metrics(api, metric_on_set=metric_on_set, mq_args=mq_args)
      results2 = get_metrics(api2, metric_on_set=metric_on_set, mq_args=mq_args)
      from mmrazor.models.utils import add_prefix
      results = collections.OrderedDict()
      results.update(add_prefix(results2, 'retrained'))
      results.update(add_prefix(results1, 'sliced'))         
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
      fig.set_size_inches(10, 10)
      im = ax.imshow(k_corr, cmap='coolwarm', vmin=0.5, vmax=1)
      for i in range(num_rows):
            for j in range(num_cols):
                  text = ax.text(j, i, f'{k_corr[i, j]:.2f}',
                                 ha='center', va='center', color='w')
      plt.colorbar(im)
      ax.set_xticks(np.arange(num_cols))
      ax.set_xticklabels(list(results.keys()), rotation=90)
      ax.set_yticks(np.arange(num_rows))
      ax.set_yticklabels(list(results.keys()), rotation=45)
      plt.title('Kendall Rank Matrix')
      ax.tick_params(axis='x', labelsize='xx-small')
      ax.tick_params(axis='y', labelsize='xx-small')      
      plt.show()
      fig.savefig(f'{WORK_DIR}/kendall_rank_matrix_{dataset}.png')
      fig, ax = plt.subplots()
      fig.set_size_inches(10, 10)
      im = ax.imshow(s_corr, cmap='coolwarm', vmin=0.5, vmax=1)
      for i in range(num_rows):
            for j in range(num_cols):
                  text = ax.text(j, i, f'{s_corr[i, j]:.2f}',
                                 ha='center', va='center', color='w')      
      plt.colorbar(im)
      ax.set_xticks(np.arange(num_cols))
      ax.set_xticklabels(list(results.keys()), rotation=90)
      ax.set_yticks(np.arange(num_rows))
      ax.set_yticklabels(list(results.keys()), rotation=45)
      plt.title('Spearmanr Rank Matrix')
      ax.tick_params(axis='x', labelsize='xx-small')
      ax.tick_params(axis='y', labelsize='xx-small')
      plt.show()
      fig.savefig(f'{WORK_DIR}/spearmanr_rank_matrix_{dataset}.png')      

      sorted_fp32 = get_sorted_indices(results[base_metric])
      fig, ax = plt.subplots()
      metric = 'deployability_index'
      sorted_rst = get_sorted_indices(results.pop(f'retrained.{metric}'))
      ax.scatter(sorted_fp32, sorted_rst, alpha=0.5, label=f'retrained.{metric}')
      sorted_rst = get_sorted_indices(results.pop(f'sliced.{metric}'))
      ax.scatter(sorted_fp32, sorted_rst, alpha=0.5, label=f'sliced.{metric}')      
            # marker=metric_marker_mapping[metric][0],
            # color=metric_marker_mapping[metric][1])
      ax.set_xlabel('Architecture ranking for retrained float32 accuracy')
      ax.set_ylabel('Architecture ranking for deploy index')
      ax.set_title(f'Architecture ranking deploy index on {dataset}')
      plt.tight_layout()
      plt.legend()
      plt.show()
      fig.savefig(f'{WORK_DIR}/architecture_rank_deploy_{indicator}_{dataset}.png')
      
      fig, ax = plt.subplots()
      for jdx, metric in enumerate(results):
            if metric not in metric_marker_mapping:
                  continue
            sorted_rst = get_sorted_indices(results[metric])
            ax.scatter(sorted_fp32, sorted_rst, alpha=0.5, label=metric,)
                  #      marker=metric_marker_mapping[metric][0],
                  #      color=metric_marker_mapping[metric][1])
      ax.set_xlabel('Architecture ranking for retrained float32 accuracy')
      ax.set_ylabel('Architecture ranking accuracy')
      ax.set_title(f'Architecture ranking accuracy on {dataset}')
      plt.tight_layout()
      plt.legend()
      plt.show()
      fig.savefig(f'{WORK_DIR}/architecture_rank_mq_{dataset}.png')
