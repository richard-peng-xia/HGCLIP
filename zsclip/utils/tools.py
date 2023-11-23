import mmcv
import torch
import tqdm
import numpy as np
import clip
import os

def update_from_config(args):
    cfg = mmcv.Config.fromfile(args.config)
    for _, cfg_item in cfg._cfg_dict.items():
        for k, v in cfg_item.items():
            setattr(args, k, v)

    return args

def one_hot_label(labels, labels_name):

    # one_hot_codes = np.eye(len(labels))
    one_hot_codes = np.eye(len(labels_name))

    one_hot_labels = []
    for l in labels:
        one_hot_label = one_hot_codes[l]
        one_hot_labels.append(one_hot_label)

    labels = torch.tensor(np.array(one_hot_labels))

    return labels

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

def transform_mapping(father, son, mapping_dict):
    transformed_mapping_dict = {}
    
    for key, value in mapping_dict.items():
        mapped_key = father[key]
        mapped_values = []
        for val in value:
            if val.split('_')[-1] not in son.keys():
                continue
            mapped_values.append(son[val.split('_')[-1]])
        transformed_mapping_dict[mapped_key] = mapped_values 
    
    return transformed_mapping_dict

def zeroshot_classifier(classnames, templates, model):
    with torch.no_grad():
        zeroshot_weights = []
        # for classname in tqdm.tqdm(classnames):
        for classname in classnames:
            texts = [template.format(classname) for template in templates] #format with class
            texts = clip.tokenize(texts).cuda() #tokenize
            class_embeddings = model.encode_text(texts) #embed with text encoder
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()
            zeroshot_weights.append(class_embedding)
        zeroshot_weights = torch.stack(zeroshot_weights, dim=1).cuda()
    return zeroshot_weights

def zeroshot_classifier_cifar(classnames, templates, model, fine2coarse):
    with torch.no_grad():
        zeroshot_weights = []
        for i in tqdm.tqdm(range(len(classnames))):
            classname = classnames[i]
            texts = [template.format(classname, fine2coarse[int(i)]) for template in templates] #format with class
            texts = clip.tokenize(texts).cuda() #tokenize
            class_embeddings = model.encode_text(texts) #embed with text encoder
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()
            zeroshot_weights.append(class_embedding)
        zeroshot_weights = torch.stack(zeroshot_weights, dim=1).cuda()
    return zeroshot_weights

def zeroshot_classifier_coarse_with_fine(classnames, templates, model, mapping_dict):
    with torch.no_grad():
        zeroshot_weights = []
        for classname in tqdm.tqdm(classnames):
            texts = [template.format(classname, ', a '.join(mapping_dict[classname]))[:77] for template in templates] #format with class
            texts = clip.tokenize(texts).cuda() #tokenize
            class_embeddings = model.encode_text(texts) #embed with text encoder
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()
            zeroshot_weights.append(class_embedding)
        zeroshot_weights = torch.stack(zeroshot_weights, dim=1).cuda()
    return zeroshot_weights

def zeroshot_classifier_ethec(classnames, templates, model, child_of_family, child_of_subfamily, child_of_genus, num):
    with torch.no_grad():
        zeroshot_weights = []
        for classname in tqdm.tqdm(classnames):
            level3 = list(filter(lambda k: classname in child_of_genus[k], child_of_genus.keys()))
            level2 = list(filter(lambda k: level3 in child_of_subfamily[k], child_of_subfamily.keys()))
            level1 = list(filter(lambda k: level2 in child_of_family[k], child_of_family.keys()))
            if num == 4:
                texts = [template.format(classname, level3, level2, level1) for template in templates] #format with class
            elif num == 3:
                texts = [template.format(level3, level2, level1) for template in templates]
            elif num == 2:
                texts = [template.format(level2, level1) for template in templates]
            texts = clip.tokenize(texts).cuda() #tokenize
            class_embeddings = model.encode_text(texts) #embed with text encoder
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()
            zeroshot_weights.append(class_embedding)
        zeroshot_weights = torch.stack(zeroshot_weights, dim=1).cuda()
    return zeroshot_weights

