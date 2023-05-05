import nats_bench


def find_best(api, dataset, metric_on_set, hp, mq_args=None):
    
      evaluated_indexes = list(range(len(api)))
      results = {}
      def find_best_with_metric(dataset, metric):
            best_index = -1
            for arch_index in evaluated_indexes:
                  api._prepare_info(arch_index)
                  arch_info = api.arch2infos_dict[arch_index][hp]
                  try:
                        xinfo = arch_info.get_metrics(dataset, metric)
                  except:
                       import pdb; pdb.set_trace()
                  accuracy = xinfo['accuracy']
                  if best_index == -1:
                        best_index, highest_accuracy = arch_index, accuracy
                  elif highest_accuracy < accuracy:
                        best_index, highest_accuracy = arch_index, accuracy
            return best_index, highest_accuracy

      # 1. Find the best architecture with fp32 setting
      results[metric_on_set] = find_best_with_metric(dataset, metric_on_set)
      # 2. Find the best architecture with quantize settings      
      if mq_args is not None:
        if not isinstance(mq_args, (list, tuple)):
            mq_args = [mq_args]
        for mq in mq_args:
            mq_dataset = dataset + '_' + mq
            # import pdb; pdb.set_trace()
            results[mq] = find_best_with_metric(mq_dataset, mq)

      return results


print('NATS-Bench version: {:}'.format(nats_bench.version()))
# Create the API for tologoy search space
nas_mqbench_pth = '/home/wangshiguang/NAS-MQBench/mmrazor/work_dirs/NASMQ_NATS-tss'
api = nats_bench.create(nas_mqbench_pth, 'tss', fast_mode=True, verbose=True)


dataset = 'cifar10'
metric_on_set = 'ori-test'
hp = '12'
mq_args = 'ptq_per-tensor_minmax_nats'   # ptq_(quantize_granularity)_(calib)_(search_space)
api.verbose = False
# results = find_best(api, dataset=dataset, metric_on_set=metric_on_set, hp=hp, mq_args=mq_args)
# print(f'{nats_bench.api_utils.time_string()} The best architecture on {dataset} ' +
#       f'with hp: {hp} training is: {results}')

# hp = '200'
# mq_args = ('ptq_per-tensor_histogram_nats', )   # ptq_(quantize_granularity)_(calib)_(search_space)
# api.verbose = False
# # best_arch_index, highest_valid_accuracy = api.find_best(dataset=dataset, metric_on_set=metric_on_set, hp=hp)
# results = find_best(api, dataset=dataset, metric_on_set=metric_on_set, hp=hp, mq_args=mq_args)
# print(f'{nats_bench.api_utils.time_string()} The best architecture on {dataset} ' +
#       f'with hp: {hp} training is: {results}')


dataset = 'cifar100'
hp = '200'
mq_args = ('ptq_per-tensor_minmax_nats', 'ptq_per-channel_minmax_nats')   # ptq_(quantize_granularity)_(calib)_(search_space)
api.verbose = False
# best_arch_index, highest_valid_accuracy = api.find_best(dataset=dataset, metric_on_set=metric_on_set, hp=hp)
results = find_best(api, dataset=dataset, metric_on_set=metric_on_set, hp=hp, mq_args=mq_args)
print(f'{nats_bench.api_utils.time_string()} The best architecture on {dataset} ' +
      f'with hp: {hp} training is: {results}')


dataset = 'ImageNet16-120'
hp = '200'
mq_args = ('ptq_per-tensor_minmax_nats', 'ptq_per-channel_minmax_nats')   # ptq_(quantize_granularity)_(calib)_(search_space)
api.verbose = False
# best_arch_index, highest_valid_accuracy = api.find_best(dataset=dataset, metric_on_set=metric_on_set, hp=hp)
results = find_best(api, dataset=dataset, metric_on_set=metric_on_set, hp=hp, mq_args=mq_args)
print(f'{nats_bench.api_utils.time_string()} The best architecture on {dataset} ' +
      f'with hp: {hp} training is: {results}')