import torch
from collections import defaultdict


def cal_torch_model_params(model):
    """
    Count PyTorch model parameters by coarse granularity
    :param model: nn.Module
    :return: None, prints directly
    """
    total_params = 0
    total_trainable_params = 0

    module_params = defaultdict(int)
    module_trainable_params = defaultdict(int)

    for name, param in model.named_parameters():
        if '.' in name:
            # Extract first two levels of module names, e.g., image_encoder.layer1
            module_name = '.'.join(name.split('.')[:2])
        else:
            module_name = name  # No submodules

        num_params = param.numel()
        module_params[module_name] += num_params
        total_params += num_params

        if param.requires_grad:
            module_trainable_params[module_name] += num_params
            total_trainable_params += num_params

    print("\n")
    print(f"{'Module':<30} | {'Total Params':>12} | {'Trainable Params':>16}")
    print("-" * 65)

    for module in sorted(module_params.keys()):
        total = module_params[module]
        trainable = module_trainable_params.get(module, 0)
        print(f"{module:<30} | {total:12,} | {trainable:16,}")

    print("-" * 65)
    print(f"{'Total':<30} | {total_params:12,} | {total_trainable_params:16,}\n")


# device
def setup_device(n_gpu_use):
    n_gpu = torch.cuda.device_count()
    if n_gpu_use > 0 and n_gpu == 0:
        print("Warning: There\'s no GPU available on this machine, training will be performed on CPU.")
        n_gpu_use = 0
    if n_gpu_use > n_gpu:
        print(
            "Warning: The number of GPU\'s configured to use is {}, but only {} are available on this machine.".format(
                n_gpu_use, n_gpu))
        n_gpu_use = n_gpu
    device = torch.device('cuda:0' if n_gpu_use > 0 else 'cpu')
    list_ids = list(range(n_gpu_use))
    return device, list_ids


def is_left_better_right(left_num, right_num, standard):
    '''

    :param left_num:
    :param right_num:
    :param standard: if max, left_num > right_num is true, if min, left_num < right_num is true.
    :return:
    '''
    assert standard in ["max", "min"]
    if standard == "max":
        return left_num > right_num
    elif standard == "min":
        return left_num < right_num
