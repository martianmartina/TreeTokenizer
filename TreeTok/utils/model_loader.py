import glob
import torch
import os


def load_model(model, model_path, strict=True):
    state_dict = torch.load(model_path, map_location=lambda a, b: a)
    transfered_state_dict = {}
    for k, v in state_dict.items():
        new_k = k.replace('module.', '')
        transfered_state_dict[new_k] = v
    model.load_state_dict(transfered_state_dict, strict=strict)


def load_checkpoint(modules, files, output_dir):
    for module, file in zip(modules, files):
        path = os.path.join(output_dir, file)
        module.load_state_dict(torch.load(path, map_location="cpu"))


def get_max_epoch(output_dir, pattern, acc=False):
    fn_dir_list = glob.glob(os.path.join(output_dir, pattern))
    if not fn_dir_list:
        return -1

    def get_epoch_num(fn, acc):
        if "/" in fn:
            fn = fn.rsplit("/", 1)[1]
        if acc: # model30_9552: epoch 30 accuracy 95.52
            epoch_acc = fn.replace("model", "").replace(".bin", "")
            epoch, acc = epoch_acc.split("_")
            epoch, acc = int(epoch), float(acc)
        else:
            epoch = int(fn.replace("model", "").replace(".bin", ""))
        return epoch # acc is not considered in this func

    epoch_set = set([get_epoch_num(fn, acc) for fn in fn_dir_list])
    if epoch_set:
        return max(epoch_set)
    else:
        return -1

def get_max_epoch_step(output_dir, pattern):
    fn_dir_list = glob.glob(os.path.join(output_dir, pattern))
    if not fn_dir_list:
        return -1, -1

    max_epoch = -1
    max_step = -1
    for fn in fn_dir_list:
        if "/" in fn:
            fn = fn.rsplit("/", 1)[1]
        
        epoch = int(fn.replace("model", "").replace(".bin", "").split('_')[0])
        step = int(fn.replace("model", "").replace(".bin", "").split('_')[1])
        if epoch >= max_epoch:
            if step > max_step or epoch > max_epoch:
                max_epoch = epoch
                max_step = step

    return max_epoch, max_step