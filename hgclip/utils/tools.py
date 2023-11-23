import mmcv
import torch
import tqdm
import numpy as np
import os

def update_from_config(args):
    cfg = mmcv.Config.fromfile(args.config)
    for _, cfg_item in cfg._cfg_dict.items():
        for k, v in cfg_item.items():
            setattr(args, k, v)

    return args

def map_dic(d):
    mapping_dict = {}
    for high_dimension, low_dimension in d.items():
        if low_dimension in mapping_dict:
            mapping_dict[low_dimension].append(high_dimension)
        else:
            mapping_dict[low_dimension] = [high_dimension]
    return mapping_dict

def get_low_dimension_result(high_dimension_result,mapping_dict):
    low_dimension_result = []
    for row in high_dimension_result:
        low_dimension_row = [sum(row[indices]) for indices in mapping_dict.values()]
        low_dimension_result.append(low_dimension_row)
    return low_dimension_result
