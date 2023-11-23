import clip
from PIL import Image
import torch
import argparse
import warnings
import os
import tqdm
import numpy as np
import pandas as pd
import pickle
import torchvision.transforms as transforms
import torchvision
from utils import *
import torch.nn as nn
from torch.autograd import Variable
import json,sklearn
from sklearn.model_selection import train_test_split

warnings.filterwarnings('ignore')

def get_args_parser():
    parser = argparse.ArgumentParser('zero-shot CLIP on hierarchical classification', add_help=False)
    parser.add_argument('--config', required=True, type=str, help='config')
    parser.add_argument('--device', default='cuda', help='device to use for training / testing')
    parser.add_argument('--gpu_id', default=0, type=int, help='gpu id')
    parser.add_argument('--seed', default='0', type=int, help='seed')
    parser.add_argument('--pretrain_clip_path', default='pretrained/ViT-B-16.pt', type=str, help='path of pretrained clip ckpt')
    parser.add_argument('--dataset', default='living17', type=str, help='dataset name', choices=['cifar-100','ethec','imagenet'])
    parser.add_argument('--batch_size', default=2048, type=int, help='batch size')
    parser.add_argument('--multi_label', default=False, type=bool)
    parser.add_argument('--marginalization', default=False, type=bool)
    return parser

def main(args):
    print(args)

    assert torch.cuda.is_available()
    torch.cuda.set_device(args.gpu_id)

    torch.manual_seed(args.seed)

    model, preprocess = clip.load(os.path.join('hgclip/',args.pretrain_clip_path), device=args.device)

    if args.dataset == 'cifar-100':
        dataset_test = torchvision.datasets.CIFAR100('hgclip/data',train = False,download = False,transform = preprocess)
    elif args.dataset == 'ethec':
        dataset_test = ETHECDataset(root_dir='hgclip/data/ETHEC_dataset', split='test', transform=transform)
    elif args.dataset == 'imagenet':
        # dataset_test = torchvision.datasets.ImageFolder(root='hgclip/data/imagenet-314/val', transform=preprocess)
        dataset_test = torchvision.datasets.ImageFolder(root='hgclip/data/imagenet-rendition/imagenet-r/', transform=preprocess)
        with open('hgclip/data/ImageNet/imagenet_class_index.json', 'r') as file:
            target_labels_data = json.load(file)
        target_labels = [label_info[0] for label_info in target_labels_data.values()]
        target_labels_name = [label_info[1].replace('_',' ') for label_info in target_labels_data.values()]
        list_200_1 = os.listdir('hgclip/data/imagenet-adversarial/imagenet-a/')
        list_200_2 = os.listdir('hgclip/data/imagenet-rendition/imagenet-r/')
        list_all = list(set(list_200_1 + list_200_2))
        target_labels_name = [target_labels_name[target_labels.index(i)] for i in list_all]
        target_labels = list_all
        target_labels_name = [target_labels_name[target_labels.index(i)] for i in list(dataset_test.class_to_idx.keys())]
    elif args.dataset == 'car':
        dataset_test = StanfordCars(root = "data/", split = "test", transform=preprocess)
    elif args.dataset == 'air':
        dataset_test = AircraftDataset(data_dir = "hgclip/data/fgvc-aircraft-2013b/data/", split = "test", transform=preprocess)
    elif args.dataset == 'caltech-101':
        dataset = torchvision.datasets.ImageFolder('hgclip/data/caltech101/101_ObjectCategories')
        labels_names_l3 = [i.lower().strip().replace('_',' ') for i in list(dataset.class_to_idx.keys())]
        _, dataset_test = train_test_split(dataset, test_size=0.2, random_state=42)
        dataset_test = caltech(dataset_test, preprocess, 'hgclip/data/caltech101/caltech101-hierarchy.txt', labels_names_l3)
    elif args.dataset == 'caltech-256':
        dataset = torchvision.datasets.ImageFolder('hgclip/data/caltech256/')
        labels_names_l3 = [i.lower().split('.')[-1].strip().replace('-101','').replace('_',' ').replace('-',' ') for i in list(dataset.class_to_idx.keys())]
        _, dataset_test = train_test_split(dataset, test_size=0.2, random_state=42)
        dataset_test = caltech(dataset_test, preprocess, '256', labels_names_l3)
    elif args.dataset == 'food-101':
        dataset = torchvision.datasets.ImageFolder('hgclip/data/food-101/food-101/images/')
        labels_names_l2 = [i.lower().strip().replace('_',' ').replace('-',' ') for i in list(dataset.class_to_idx.keys())]
        _, dataset_test = train_test_split(dataset, test_size=0.2, random_state=42)
        dataset_test = food101_dataset(dataset_test, preprocess, 'hgclip/data/food-101/food-101/food-101_hierarchy.txt', labels_names_l2)
    elif args.dataset == 'fruits-360':
        dataset_test = torchvision.datasets.ImageFolder('hgclip/data/fruits-360/fruits-360_dataset/fruits-360/Test')
        labels_names_l3 = [i.lower().strip().replace('_',' ').replace('-',' ') for i in list(dataset_test.class_to_idx.keys())]
        dataset_test = caltech(dataset_test, preprocess, 'hgclip/data/fruits-360/fruits-360_dataset/fruit-360_hierarchy.txt', labels_names_l3)
    elif args.dataset == 'pets':
        dataset_test = torchvision.datasets.OxfordIIITPet(root = 'data', split = 'test')
        labels_names_l2 = [i.lower().strip().replace('_',' ').replace('-',' ') for i in list(dataset_test.class_to_idx)]
        dataset_test = food101_dataset(dataset_test, preprocess, 'data/oxford-iiit-pet/pets-hierarchy.txt', labels_names_l2)
    elif args.dataset in ['living17', 'entity13', 'nonliving26', 'entity30']:
        from robustness import datasets
        from robustness.tools.breeds_helpers import make_living17, make_entity13, make_entity30, make_nonliving26
        info_dir = 'hgclip/data/ImageNet/breeds'
        data_dir = 'hgclip/data/ImageNet'
        if args.dataset == 'living17':
            ret = make_living17(info_dir, split="good")
        elif args.dataset == 'entity13':
            ret = make_entity13(info_dir, split="good")
        elif args.dataset == 'entity30':
            ret = make_entity30(info_dir, split="good")
        elif args.dataset == 'nonliving26':
            ret = make_nonliving26(info_dir, split="good")
        superclasses, subclass_split, label_map = ret
        train_subclasses, test_subclasses = subclass_split
        dataset_source = datasets.CustomImageNet(data_dir, train_subclasses)
        loaders_source = dataset_source.make_loaders(workers=2, batch_size=args.batch_size)
        _, test_loader_source = loaders_source
        dataset_target = datasets.CustomImageNet(data_dir, test_subclasses)
        loaders_target = dataset_target.make_loaders(workers=2, batch_size=args.batch_size)
        _, test_loader_target = loaders_target
    if args.dataset not in ['living17', 'entity13', 'nonliving26', 'entity30']:
        test_loader = torch.utils.data.DataLoader(dataset_test, batch_size = args.batch_size, shuffle = False, num_workers = 2)

    if args.dataset == 'cifar-100':
        def load_labels_name(filename):
            with open(filename, 'rb') as f:
                obj = pickle.load(f)
            return obj
        labels_name_dic = load_labels_name('hgclip/data/cifar-100-python/meta')
        labels_name_fine = labels_name_dic['fine_label_names']
        labels_name_coarse = labels_name_dic['coarse_label_names']
        fine2coarse = pd.read_pickle('hgclip/data/cifar-100-python/label.pkl')
        mapping_dict = map_dic(fine2coarse)
        mapping_dict_name = {}
        for i in mapping_dict.keys():
            mapping_dict_name[labels_name_coarse[i]] = [labels_name_fine[l] for l in mapping_dict[i]]
    elif args.dataset == 'ethec':
        label_map = ETHECLabelMap()
        labels_name_family = list(label_map.family.keys())
        labels_name_subfamily = list(label_map.subfamily.keys())
        labels_name_genus = list(label_map.genus.keys())
        labels_name_specific_epithet = list(label_map.genus_specific_epithet.keys())
        mapping_dict_genus = label_map.child_of_genus
        mapping_dict_subfamily = label_map.child_of_subfamily
        mapping_dict_family = label_map.child_of_family
        transformed_mapping_dict_genus = transform_mapping(label_map.genus, label_map.genus_specific_epithet, mapping_dict_genus)
        transformed_mapping_dict_subfamily = transform_mapping(label_map.subfamily, label_map.genus, mapping_dict_subfamily)
        transformed_mapping_dict_family = transform_mapping(label_map.family, label_map.subfamily, mapping_dict_family)
        mapping_dict_genus = {}
        mapping_dict_subfamily = {}
        mapping_dict_family = {}
        for i in transformed_mapping_dict_genus.keys():
            mapping_dict_genus[labels_name_genus[i]] = [labels_name_specific_epithet[l] for l in transformed_mapping_dict_genus[i]]
        for i in transformed_mapping_dict_subfamily.keys():
            mapping_dict_subfamily[labels_name_subfamily[i]] = [labels_name_genus[l] for l in transformed_mapping_dict_subfamily[i]]
        for i in transformed_mapping_dict_family.keys():
            mapping_dict_family[labels_name_family[i]] = [labels_name_subfamily[l] for l in transformed_mapping_dict_family[i]]
    elif args.dataset == 'imagenet':
        with open('hgclip/data/ImageNet/wordnet.is_a.txt', 'r') as f:
            hierarchy = f.readlines()
        hierarchy = [i.strip().split(' ') for i in hierarchy]
        with open('hgclip/data/ImageNet/words.txt', 'r') as f:
            words = f.readlines()
        word_id = [word.strip().split('\t')[0] for word in words]
        word_name = [word.strip().split('\t')[1].split(',')[0] for word in words]
        word_id_new = target_labels.copy()
        word_name_new = target_labels_name.copy()
        print(f'has {len(word_name_new)} labels and {len(word_name_new)} label names')
        x = []
        y = []
        for i in hierarchy: # len(hierarchy) = 75850
            start, end = i
            if end in target_labels:
                if start not in word_id_new:
                    word_id_new.append(start)
                    word_name_new.append(word_name[word_id.index(start)])
                x.append(word_id_new.index(start))
                y.append(word_id_new.index(end))
        print(f'has {len(word_name_new)} labels and {len(word_name_new)} label names after adding hierarchies')
        labels_name = word_name_new
    elif args.dataset == 'car':
        label_map = StanfordCarsLabelMap(root = "hgclip/data/")
        labels_name_fine = label_map.fine_classes
        labels_name_coarse = label_map.coarse_classes
        mapping_dict = label_map.trees
    elif args.dataset == 'air':
        label_map = AircraftMap(data_dir = "hgclip/data/fgvc-aircraft-2013b/data/")
        labels_name_model = label_map.labels_names_variant
        labels_name_family = label_map.labels_names_family
        labels_name_maker = label_map.labels_names_maker
    elif args.dataset == 'caltech-101' or args.dataset == 'caltech-256' or args.dataset == 'fruits-360':
        labels_names_l2, labels_names_l1 = dataset_test.load_data()
        print(f'level 1: {len(labels_names_l1)} level 2: {len(labels_names_l2)} level 3: {len(labels_names_l3)}')
    elif args.dataset == 'food-101' or args.dataset == 'pets':
        labels_names_l1 = dataset_test.load_data()
        print(f'level 1: {len(labels_names_l1)} level 2: {len(labels_names_l2)}')
    elif args.dataset in ['living17', 'entity13', 'nonliving26', 'entity30']:
        labels_names = list(label_map.values())

    if args.dataset == 'cifar-100':
        if args.multi_label:
            zeroshot_weights_fine = zeroshot_classifier(labels_name_fine, ['a photo of a {}.'], model)
            # zeroshot_weights_fine = zeroshot_classifier_cifar(labels_name_fine, ['a photo of a {}, which is a kind of {}.'], model, fine2coarse)
            zeroshot_weights_coarse = zeroshot_classifier(labels_name_coarse, ['a photo of a {}.'], model)
            # zeroshot_weights_coarse = zeroshot_classifier_coarse_with_fine(labels_name_coarse, ['a photo of a {}, possibly a {}.'], model, mapping_dict_name)#which consists of
        else:
            # zeroshot_weights = zeroshot_classifier_cifar(labels_name_fine, ['a photo of a {}, which is a kind of {}.'], model, fine2coarse) # ['a photo of a {}.'] cifar100_templates
            zeroshot_weights = zeroshot_classifier(labels_name_fine, cifar100_templates, model)
        with torch.no_grad():
            top1_fine, top5_fine, top1_coarse, top5_coarse, n = 0., 0., 0., 0., 0.
            for i, (images, target) in enumerate(tqdm.tqdm(test_loader)):
                images = images.cuda()
                target = target.cuda()
                target_coarse = torch.tensor([fine2coarse[int(i)] for i in target]).cuda()
                
                # predict
                image_features = model.encode_image(images)
                image_features /= image_features.norm(dim=-1, keepdim=True)
                if args.multi_label:
                    logits_fine = 100. * image_features @ zeroshot_weights_fine
                    logits_coarse = 100. * image_features @ zeroshot_weights_coarse
                elif args.marginalization:
                    logits_fine = (image_features @ zeroshot_weights)
                    logits_coarse = torch.tensor(get_low_dimension_result(logits_fine,mapping_dict)).cuda()

                # measure accuracy
                acc1_fine, acc5_fine = acc(logits_fine, target, topk=(1, 5))
                acc1_coarse, acc5_coarse = acc(logits_coarse, target_coarse, topk=(1, 5))
                top1_fine += acc1_fine
                top5_fine += acc5_fine
                top1_coarse += acc1_coarse
                top5_coarse += acc5_coarse
                n += images.size(0)

        top1_fine = (top1_fine / n) * 100
        top5_fine = (top5_fine / n) * 100
        top1_coarse = (top1_coarse / n) * 100
        top5_coarse = (top5_coarse / n) * 100

        print(f"Fine Top-1 accuracy: {top1_fine:.2f} Top-5 accuracy: {top5_fine:.2f}")
        print(f"Coarse Top-1 accuracy: {top1_coarse:.2f} Top-5 accuracy: {top5_coarse:.2f}")
        
    elif args.dataset == 'car':
        if args.multi_label:
            zeroshot_weights_fine = zeroshot_classifier(labels_name_fine, ['a photo of a {}.'], model)
            # zeroshot_weights_fine = zeroshot_classifier_cifar(labels_name_fine, ['a photo of a {}, which is a kind of {}.'], model, fine2coarse)
            zeroshot_weights_coarse = zeroshot_classifier(labels_name_coarse, ['a photo of a {}.'], model)
            # zeroshot_weights_coarse = zeroshot_classifier_coarse_with_fine(labels_name_coarse, ['a photo of a {}, possibly a {}.'], model, mapping_dict_name)#which consists of
        else:
            # zeroshot_weights = zeroshot_classifier_cifar(labels_name_fine, ['a photo of a {}, which is a kind of {}.'], model, fine2coarse) # ['a photo of a {}.'] cifar100_templates
            zeroshot_weights = zeroshot_classifier(labels_name_fine, cifar100_templates, model)
        with torch.no_grad():
            top1_fine, top5_fine, top1_coarse, top5_coarse, n = 0., 0., 0., 0., 0.
            for i, (images, target, target_coarse) in enumerate(tqdm.tqdm(test_loader)):
                images = images.cuda()
                target = target.cuda()
                target_coarse = target_coarse.cuda()

                # predict
                image_features = model.encode_image(images)
                image_features /= image_features.norm(dim=-1, keepdim=True)
                if args.multi_label:
                    logits_fine = 100. * image_features @ zeroshot_weights_fine
                    logits_coarse = 100. * image_features @ zeroshot_weights_coarse
                elif args.marginalization:
                    logits_fine = (image_features @ zeroshot_weights)
                    logits_coarse = torch.tensor(get_low_dimension_result(logits_fine,mapping_dict)).cuda()

                # measure accuracy
                acc1_fine, acc5_fine = acc(logits_fine, target, topk=(1, 5))
                acc1_coarse, acc5_coarse = acc(logits_coarse, target_coarse, topk=(1, 5))
                top1_fine += acc1_fine
                top5_fine += acc5_fine
                top1_coarse += acc1_coarse
                top5_coarse += acc5_coarse
                n += images.size(0)

        top1_fine = (top1_fine / n) * 100
        top5_fine = (top5_fine / n) * 100
        top1_coarse = (top1_coarse / n) * 100
        top5_coarse = (top5_coarse / n) * 100

        print(f"Fine Top-1 accuracy: {top1_fine:.2f} Top-5 accuracy: {top5_fine:.2f}")
        print(f"Coarse Top-1 accuracy: {top1_coarse:.2f} Top-5 accuracy: {top5_coarse:.2f}")
        
    elif args.dataset == 'air':
        if args.multi_label:
            zeroshot_weights_model = zeroshot_classifier(labels_name_model, ['a photo of a {} aircraft.'], model)
            # zeroshot_weights_fine = zeroshot_classifier_cifar(labels_name_fine, ['a photo of a {}, which is a kind of {}.'], model, fine2coarse)
            zeroshot_weights_family = zeroshot_classifier(labels_name_family, ['a photo of a {} aircraft.'], model)
            # zeroshot_weights_coarse = zeroshot_classifier_coarse_with_fine(labels_name_coarse, ['a photo of a {}, possibly a {}.'], model, mapping_dict_name)#which consists of
            zeroshot_weights_maker = zeroshot_classifier(labels_name_maker, ['a photo of a {} aircraft.'], model)
        else:
            # zeroshot_weights = zeroshot_classifier_cifar(labels_name_fine, ['a photo of a {}, which is a kind of {}.'], model, fine2coarse) # ['a photo of a {}.'] cifar100_templates
            zeroshot_weights = zeroshot_classifier(labels_name_model, cifar100_templates, model)
        with torch.no_grad():
            top1_model, top1_family, top1_maker, n = 0., 0., 0., 0.
            for i, (images, target_model, target_family, target_maker) in enumerate(tqdm.tqdm(test_loader)):
                images = images.cuda()
                target_model = torch.tensor([int(labels_name_model.index(l)) for l in target_model]).cuda()
                target_family = torch.tensor([int(labels_name_family.index(l)) for l in target_family]).cuda()
                target_maker = torch.tensor([int(labels_name_maker.index(l)) for l in target_maker]).cuda()

                # predict
                image_features = model.encode_image(images)
                image_features /= image_features.norm(dim=-1, keepdim=True)
                if args.multi_label:
                    logits_model = 100. * image_features @ zeroshot_weights_model
                    logits_family = 100. * image_features @ zeroshot_weights_family
                    logits_maker = 100. * image_features @ zeroshot_weights_maker
                elif args.marginalization:
                    logits_fine = (image_features @ zeroshot_weights)
                    logits_coarse = torch.tensor(get_low_dimension_result(logits_fine,mapping_dict)).cuda()

                # measure accuracy
                acc1_model, _ = acc(logits_model, target_model, topk=(1, 5))
                acc1_family, _ = acc(logits_family, target_family, topk=(1, 5))
                acc1_maker, _ = acc(logits_maker, target_maker, topk=(1, 5))
                top1_model += acc1_model
                top1_family += acc1_family
                top1_maker += acc1_maker
                n += images.size(0)

        top1_model = (top1_model / n) * 100
        top1_family = (top1_family / n) * 100
        top1_maker = (top1_maker / n) * 100

        print(f"Model Top-1 accuracy: {top1_model:.2f} Family: {top1_family:.2f} Maker: {top1_maker:.2f}")
        
    elif args.dataset == 'ethec':
        child_of_family = label_map.child_of_family
        child_of_subfamily = label_map.child_of_subfamily
        child_of_genus = label_map.child_of_genus
        if args.multi_label: # ['a photo of a {}, a type of butterfly specimens.']
            zeroshot_weights_specific_epithet = zeroshot_classifier(labels_name_specific_epithet, ['a photo of a {}, a type of butterfly specimens.'], model) # ethec_templates
            # zeroshot_weights_genus = zeroshot_classifier(labels_name_genus, ['a photo of a {}, a type of butterfly specimens.'], model)
            # zeroshot_weights_subfamily = zeroshot_classifier(labels_name_subfamily, ['a photo of a {}, a type of butterfly specimens.'], model)
            # zeroshot_weights_family = zeroshot_classifier(labels_name_family, ['a photo of a {}, a type of butterfly specimens.'], model)

            # zeroshot_weights_specific_epithet = zeroshot_classifier_ethec(labels_name_specific_epithet, ['a photo of a {}, a type of butterfly specimens, which is a kind of {} {} {}.'], model, child_of_family, child_of_subfamily, child_of_genus, 4)
            # zeroshot_weights_genus = zeroshot_classifier_ethec(labels_name_genus, ['a photo of a {}, a type of butterfly specimens, which is a kind of {} {}.'], model, child_of_family, child_of_subfamily, child_of_genus, 3)
            # zeroshot_weights_subfamily = zeroshot_classifier_ethec(labels_name_subfamily, ['a photo of a {}, a type of butterfly specimens, which is a kind of {}.'], model, child_of_family, child_of_subfamily, child_of_genus, 2)
            # zeroshot_weights_family = zeroshot_classifier(labels_name_family, ['a photo of a {}, a type of butterfly specimens.'], model)

            zeroshot_weights_genus = zeroshot_classifier_coarse_with_fine(labels_name_genus, ['a photo of a {}, a type of butterfly specimens, which consists of a {}.'], model, mapping_dict_genus)#which consists of possibly
            zeroshot_weights_subfamily = zeroshot_classifier_coarse_with_fine(labels_name_subfamily, ['a photo of a {}, a type of butterfly specimens, which consists of a {}.'], model, mapping_dict_subfamily)
            zeroshot_weights_family = zeroshot_classifier_coarse_with_fine(labels_name_family, ['a photo of a {}, a type of butterfly specimens, which consists of a {}.'], model, mapping_dict_family)
        else:
            zeroshot_weights_specific_epithet = zeroshot_classifier_ethec(labels_name_specific_epithet, ['a photo of a {}, which is a kind of {} {} {}.'], model, child_of_family, child_of_subfamily, child_of_genus) # ['a photo of a {}.'] cifar100_templates
            # zeroshot_weights_specific_epithet = zeroshot_classifier(labels_name_specific_epithet, imagenet_templates, model)
        with torch.no_grad():
            top1_family, top5_family, top1_subfamily, top5_subfamily, top1_genus, top5_genus, top1_specific_epithet, top5_specific_epithet, n = 0., 0., 0., 0., 0., 0.,0., 0., 0.
            for i, (images, target_family, target_subfamily, target_genus, target_specific_epithet) in enumerate(tqdm.tqdm(test_loader)):
                images = images.cuda()
                target_family = torch.tensor([int(label_map.family[l]) for l in target_family]).cuda()
                target_subfamily = torch.tensor([int(label_map.subfamily[l]) for l in target_subfamily]).cuda()
                target_genus = torch.tensor([int(label_map.genus[l]) for l in target_genus]).cuda()
                target_specific_epithet = torch.tensor([int(label_map.genus_specific_epithet[l]) for l in target_specific_epithet]).cuda()
                
                # predict
                image_features = model.encode_image(images)
                image_features /= image_features.norm(dim=-1, keepdim=True)
                if args.multi_label:
                    logits_specific_epithet = 100. * image_features @ zeroshot_weights_specific_epithet
                    logits_genus = 100. * image_features @ zeroshot_weights_genus
                    logits_subfamily = 100. * image_features @ zeroshot_weights_subfamily
                    logits_family = 100. * image_features @ zeroshot_weights_family
                elif args.marginalization:
                    logits_specific_epithet = (image_features @ zeroshot_weights_specific_epithet)
                    logits_genus = torch.tensor(get_low_dimension_result(logits_specific_epithet,transformed_mapping_dict_genus)).cuda()
                    logits_subfamily = torch.tensor(get_low_dimension_result(logits_genus,transformed_mapping_dict_subfamily)).cuda()
                    logits_family = torch.tensor(get_low_dimension_result(logits_subfamily,transformed_mapping_dict_family)).cuda()

                # measure accuracy
                acc1_specific_epithet, acc5_specific_epithet = acc(logits_specific_epithet, target_specific_epithet, topk=(1, 5))
                acc1_genus, acc5_genus = acc(logits_genus, target_genus, topk=(1, 5))
                acc1_subfamily, acc5_subfamily = acc(logits_subfamily, target_subfamily, topk=(1, 5))
                acc1_family, acc5_family = acc(logits_family, target_family, topk=(1, 5))
                top1_specific_epithet += acc1_specific_epithet
                top5_specific_epithet += acc5_specific_epithet
                top1_genus += acc1_genus
                top5_genus += acc5_genus
                top1_subfamily += acc1_subfamily
                top5_subfamily += acc5_subfamily
                top1_family += acc1_family
                top5_family += acc5_family
                n += images.size(0)

        top1_specific_epithet = (top1_specific_epithet / n) * 100
        top5_specific_epithet = (top5_specific_epithet / n) * 100
        top1_genus = (top1_genus / n) * 100
        top5_genus = (top5_genus / n) * 100
        top1_subfamily = (top1_subfamily / n) * 100
        top5_subfamily = (top5_subfamily / n) * 100
        top1_family = (top1_family / n) * 100
        top5_family = (top5_family / n) * 100

        print(f"specific_epithet Top-1 accuracy: {top1_specific_epithet:.2f} Top-5 accuracy: {top5_specific_epithet:.2f}")
        print(f"genus Top-1 accuracy: {top1_genus:.2f} Top-5 accuracy: {top5_genus:.2f}")
        print(f"subfamily Top-1 accuracy: {top1_subfamily:.2f} Top-5 accuracy: {top5_subfamily:.2f}")
        print(f"family Top-1 accuracy: {top1_family:.2f} Top-5 accuracy: {top5_family:.2f}")
        
    elif args.dataset == 'imagenet':
        zeroshot_weights_l2 = zeroshot_classifier(labels_name[:200], ['a photo of a {}.'], model) # imagenet_templates ['a photo of a {}.']
        zeroshot_weights_l1 = zeroshot_classifier(labels_name[200:], ['a photo of a {}.'], model)
        with torch.no_grad():
            top1_l1, top1_l2, n = 0., 0., 0.
            for i, (images, target) in enumerate(tqdm.tqdm(test_loader)):
                images = images.cuda()
                labels_l2 = target.to(args.device)
                labels_l1 = torch.tensor([x[y.index(int(l))]-200 for l in target]).to(args.device)
                
                # predict
                image_features = model.encode_image(images)
                image_features /= image_features.norm(dim=-1, keepdim=True)
                logits_l1 = 100. * image_features @ zeroshot_weights_l1
                logits_l2 = 100. * image_features @ zeroshot_weights_l2

                # measure accuracy
                acc1_l2 = acc(logits_l2, labels_l2, topk=(1, 5))[0]
                acc1_l1 = acc(logits_l1, labels_l1, topk=(1, 5))[0]
                top1_l2 += acc1_l2
                top1_l1 += acc1_l1
                n += images.size(0)

        top1_l1 = (top1_l1 / n) * 100
        top1_l2 = (top1_l2 / n) * 100

        print(f"Top-1 accuracy l1: {top1_l1:.2f} l2: {top1_l2:.2f}")
        
    elif args.dataset == 'caltech-101' or args.dataset == 'caltech-256' or args.dataset == 'fruits-360':
        if args.multi_label:
            zeroshot_weights_l1 = zeroshot_classifier(labels_names_l1, ['a photo of a {}.'], model)
            zeroshot_weights_l2 = zeroshot_classifier(labels_names_l2, ['a photo of a {}.'], model)
            zeroshot_weights_l3 = zeroshot_classifier(labels_names_l3, ['a photo of a {}.'], model)
        else:
            # zeroshot_weights = zeroshot_classifier_cifar(labels_name_fine, ['a photo of a {}, which is a kind of {}.'], model, fine2coarse) # ['a photo of a {}.'] cifar100_templates
            zeroshot_weights = zeroshot_classifier(labels_name_model, cifar100_templates, model)
        with torch.no_grad():
            top1_l1, top1_l2, top1_l3, n = 0., 0., 0., 0.
            for i, (images, labels_l3, labels_l2, labels_l1) in enumerate(tqdm.tqdm(test_loader)):
                images = images.cuda()
                label_l3 = torch.tensor([int(labels_names_l3.index(l)) for l in labels_l3]).to(args.device)
                label_l2 = torch.tensor([int(labels_names_l2.index(l)) for l in labels_l2]).to(args.device)
                label_l1 = torch.tensor([int(labels_names_l1.index(l)) for l in labels_l1]).to(args.device)

                # predict
                image_features = model.encode_image(images)
                image_features /= image_features.norm(dim=-1, keepdim=True)
                if args.multi_label:
                    logits_l1 = 100. * image_features @ zeroshot_weights_l1
                    logits_l2 = 100. * image_features @ zeroshot_weights_l2
                    logits_l3 = 100. * image_features @ zeroshot_weights_l3
                elif args.marginalization:
                    logits_fine = (image_features @ zeroshot_weights)
                    logits_coarse = torch.tensor(get_low_dimension_result(logits_fine,mapping_dict)).cuda()

                # measure accuracy
                acc1_l1 = acc(logits_l1, label_l1, topk=(1,))[0]
                acc1_l2 = acc(logits_l2, label_l2, topk=(1, 5))[0]
                acc1_l3 = acc(logits_l3, label_l3, topk=(1, 5))[0]
                top1_l1 += acc1_l1
                top1_l2 += acc1_l2
                top1_l3 += acc1_l3
                n += images.size(0)

        top1_l1 = (top1_l1 / n) * 100
        top1_l2 = (top1_l2 / n) * 100
        top1_l3 = (top1_l3 / n) * 100

        print(f"l1 Top-1 accuracy: {top1_l1:.2f} l2: {top1_l2:.2f} l3: {top1_l3:.2f}")
        
    elif args.dataset == 'food-101' or args.dataset == 'pets':
        if args.multi_label:
            zeroshot_weights_l1 = zeroshot_classifier(labels_names_l1, ['a photo of a {}.'], model)
            zeroshot_weights_l2 = zeroshot_classifier(labels_names_l2, ['a photo of a {}.'], model)
        else:
            # zeroshot_weights = zeroshot_classifier_cifar(labels_name_fine, ['a photo of a {}, which is a kind of {}.'], model, fine2coarse) # ['a photo of a {}.'] cifar100_templates
            zeroshot_weights = zeroshot_classifier(labels_name_model, cifar100_templates, model)
        with torch.no_grad():
            top1_l1, top1_l2, n = 0., 0., 0.
            for i, (images, labels_l2, labels_l1) in enumerate(tqdm.tqdm(test_loader)):
                images = images.cuda()
                label_l2 = torch.tensor([int(labels_names_l2.index(l)) for l in labels_l2]).to(args.device)
                label_l1 = torch.tensor([int(labels_names_l1.index(l)) for l in labels_l1]).to(args.device)

                # predict
                image_features = model.encode_image(images)
                image_features /= image_features.norm(dim=-1, keepdim=True)
                if args.multi_label:
                    logits_l1 = 100. * image_features @ zeroshot_weights_l1
                    logits_l2 = 100. * image_features @ zeroshot_weights_l2
                elif args.marginalization:
                    logits_fine = (image_features @ zeroshot_weights)
                    logits_coarse = torch.tensor(get_low_dimension_result(logits_fine,mapping_dict)).cuda()

                # measure accuracy
                acc1_l1 = acc(logits_l1, label_l1, topk=(1,))[0]
                acc1_l2 = acc(logits_l2, label_l2, topk=(1, 5))[0]
                top1_l1 += acc1_l1
                top1_l2 += acc1_l2
                n += images.size(0)

        top1_l1 = (top1_l1 / n) * 100
        top1_l2 = (top1_l2 / n) * 100

        print(f"l1 Top-1 accuracy: {top1_l1:.2f} l2: {top1_l2:.2f}")
        
    elif args.dataset in ['living17', 'entity13', 'nonliving26', 'entity30']:
        if args.multi_label:
            zeroshot_weights = zeroshot_classifier(labels_names, ['a photo of a {}.'], model)
        with torch.no_grad():
            top1_source, n_source = 0., 0.
            for i, (images, labels) in enumerate(tqdm.tqdm(test_loader_source)):
                images = images.to(args.device)
                labels = labels.to(args.device)
                # predict
                image_features = model.encode_image(images)
                image_features /= image_features.norm(dim=-1, keepdim=True)
                if args.multi_label:
                    logits = 100. * image_features @ zeroshot_weights
                # measure accuracy
                acc1_source = acc(logits, labels, topk=(1,))[0]
                top1_source += acc1_source
                n_source += images.size(0)
        with torch.no_grad():
            top1_target, n_target = 0., 0.
            for i, (images, labels) in enumerate(tqdm.tqdm(test_loader_target)):
                images = images.to(args.device)
                labels = labels.to(args.device)
                # predict
                image_features = model.encode_image(images)
                image_features /= image_features.norm(dim=-1, keepdim=True)
                if args.multi_label:
                    logits = 100. * image_features @ zeroshot_weights
                # measure accuracy
                acc1_target = acc(logits, labels, topk=(1,))[0]
                top1_target += acc1_target
                n_target += images.size(0)
        top1_source = (top1_source / n_source) * 100
        top1_target = (top1_target / n_target) * 100
        print(f"Top-1 source accuracy: {top1_source:.2f} target: {top1_target:.2f}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Zero-shot CLIP on hierarchical classification', parents=[get_args_parser()])
    args = parser.parse_args()
    args = update_from_config(args)
    main(args)
