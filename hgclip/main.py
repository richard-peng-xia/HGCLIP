import os
from clip import clip
from PIL import Image
import torch
import argparse
import warnings
import tqdm
import json
import numpy as np
import pandas as pd
import pickle
import torchvision
from utils import *
from models import *
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.optim.lr_scheduler import _LRScheduler
from torch.nn import functional as F
from torch.autograd import Variable
import dgl
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split

warnings.filterwarnings('ignore')

def get_args_parser():
    parser = argparse.ArgumentParser('HGCLIP', add_help=False)
    parser.add_argument('--config', required=True, type=str, help='config')
    parser.add_argument('--device', default='cuda', help='device to use for training / testing')
    parser.add_argument('--gpu_id', default=0, type=int, help='gpu id')
    parser.add_argument('--seed', default='0', type=int, help='seed')
    parser.add_argument('--pretrain_clip', default='ViT-B-16.pt', type=str, help='path of pretrained clip ckpt')
    parser.add_argument('--dataset', default='cifar-100', type=str, help='dataset name')
    parser.add_argument('--batch_size', default=32, type=int, help='batch size')
    parser.add_argument('--epochs', default=50, type=int, help='train epochs')
    parser.add_argument('--trainer', type=str, choices=['CoOp', 'CoCoOp', 'HGCLIP', 'MaPLe', 'PromptSRC'], help='training method')
    parser.add_argument('--ctx_init', default='a photo of a', type=str, help='context initialization')
    parser.add_argument('--n_ctx', default=2, type=int, help='context length for random initialization')
    parser.add_argument('--lr', default=0.001, type=float, help='learning rate for optim')
    parser.add_argument('--optimizer', default='sgd', type=str, choices=['sgd', 'adam', 'adamw'], help='optimizer')
    parser.add_argument('--prompt_depth', default=9, type=int, help='prompt depth for multi-modal prompt learning')
    return parser

def main(args):
    print(args)

    assert torch.cuda.is_available()
    
    torch.cuda.set_device(args.gpu_id)

    torch.manual_seed(args.seed)

    preprocess = {
            'train': transforms.Compose([transforms.Resize((224,224)),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                        ]),
            'test': transforms.Compose([
                                    transforms.Resize((224,224)),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                    ])
            }

    if args.dataset == 'cifar-100':
        dataset_train = torchvision.datasets.CIFAR100('data/', train = True, download = False, transform = preprocess['train'])
        dataset_test = torchvision.datasets.CIFAR100('data/', train = False, download = False, transform = preprocess['test'])

    elif args.dataset == 'ethec':
        dataset_train = ETHECDataset(root_dir='data/ETHEC_dataset', split='train', transform=preprocess['train'])
        dataset_test = ETHECDataset(root_dir='data/ETHEC_dataset', split='val', transform=preprocess['test'])

    elif args.dataset == 'imagenet':
        """
        Original ImageNet-1K
        dataset_train = torchvision.datasets.ImageFolder(root='data/ImageNet/train', transform=preprocess['train'])
        dataset_test = torchvision.datasets.ImageFolder(root='data/ImageNet/val', transform=preprocess['test'])
        """
        dataset_train = torchvision.datasets.ImageFolder(root='data/imagenet-314/train', transform=preprocess['train'])
        dataset_test = torchvision.datasets.ImageFolder(root='data/imagenet-314/val', transform=preprocess['test'])
        with open('data/ImageNet/imagenet_class_index.json', 'r') as file:
            target_labels_data = json.load(file)
        target_labels = [label_info[0] for label_info in target_labels_data.values()]
        target_labels_name = [label_info[1].replace('_',' ') for label_info in target_labels_data.values()]
        # imagenet-a -r
        list_200_1 = os.listdir('data/imagenet-adversarial/imagenet-a/')
        list_200_2 = os.listdir('data/imagenet-rendition/imagenet-r/')
        list_all = list(set(list_200_1 + list_200_2))
        target_labels_name = [target_labels_name[target_labels.index(i)] for i in list_all]
        target_labels = list_all
        target_labels_name = [target_labels_name[target_labels.index(i)] for i in list(dataset_train.class_to_idx.keys())]
    
    elif args.dataset == 'car':
        dataset_train = StanfordCars(root = "data/", split = "train", transform=preprocess['train'])
        dataset_test = StanfordCars(root = "data/", split = "test", transform=preprocess['test'])

    elif args.dataset == 'air':
        dataset_train = AircraftDataset(data_dir = "data/fgvc-aircraft-2013b/data/", split = "train", transform=preprocess['train'])
        dataset_test = AircraftDataset(data_dir = "data/fgvc-aircraft-2013b/data/", split = "test", transform=preprocess['test'])
    
    elif args.dataset == 'caltech-101':
        dataset = torchvision.datasets.ImageFolder('data/caltech101/101_ObjectCategories')
        labels_names_l3 = [i.lower().strip().replace('_',' ') for i in list(dataset.class_to_idx.keys())]
        dataset_train, dataset_test = train_test_split(dataset, test_size=0.2, random_state=42)
        dataset_train = caltech(dataset_train, preprocess['train'], 'data/caltech101/caltech101-hierarchy.txt', labels_names_l3)
        dataset_test = caltech(dataset_test, preprocess['test'], 'data/caltech101/caltech101-hierarchy.txt', labels_names_l3)

    elif args.dataset == 'food-101':
        dataset = torchvision.datasets.ImageFolder('data/food-101/food-101/images/')
        labels_names_l2 = [i.lower().strip().replace('_',' ').replace('-',' ') for i in list(dataset.class_to_idx.keys())]
        dataset_train, dataset_test = train_test_split(dataset, test_size=0.2, random_state=42)
        dataset_train = food101_dataset(dataset_train, preprocess['train'], 'data/food-101/food-101/food-101_hierarchy.txt', labels_names_l2)
        dataset_test = food101_dataset(dataset_test, preprocess['test'], 'data/food-101/food-101/food-101_hierarchy.txt', labels_names_l2)
    
    elif args.dataset == 'fruits-360':
        dataset_train = torchvision.datasets.ImageFolder('data/fruits-360/fruits-360_dataset/fruits-360/Training')
        dataset_test = torchvision.datasets.ImageFolder('data/fruits-360/fruits-360_dataset/fruits-360/Test')
        labels_names_l3 = [i.lower().strip().replace('_',' ').replace('-',' ') for i in list(dataset_train.class_to_idx.keys())]
        dataset_train = caltech(dataset_train, preprocess['train'], 'data/fruits-360/fruits-360_dataset/fruit-360_hierarchy.txt', labels_names_l3)
        dataset_test = caltech(dataset_test, preprocess['test'], 'data/fruits-360/fruits-360_dataset/fruit-360_hierarchy.txt', labels_names_l3)

    elif args.dataset == 'flowers-102':
        dataset_train = torchvision.datasets.Flowers102(root = 'data', split = 'train')
        dataset_test = torchvision.datasets.Flowers102(root = 'data', split = 'test')
        with open('data/flowers-102/cat_to_name.json', 'r') as f:
            idx_to_class = json.load(f)
        print(idx_to_class.values())

    elif args.dataset == 'pets':
        dataset_train = torchvision.datasets.OxfordIIITPet(root = 'data', split = 'trainval')
        dataset_test = torchvision.datasets.OxfordIIITPet(root = 'data', split = 'test')
        labels_names_l2 = [i.lower().strip().replace('_',' ').replace('-',' ') for i in list(dataset_train.class_to_idx)]
        dataset_train = food101_dataset(dataset_train, preprocess['train'], 'data/oxford-iiit-pet/pets-hierarchy.txt', labels_names_l2)
        dataset_test = food101_dataset(dataset_test, preprocess['test'], 'data/oxford-iiit-pet/pets-hierarchy.txt', labels_names_l2)

    elif args.dataset in ['living17', 'entity13', 'nonliving26', 'entity30']:
        breeds_factory = BREEDSFactory(args=args, info_dir="data/ImageNet/breeds", data_dir="data/ImageNet")
        train_loader, test_loader_source, test_loader_target = breeds_factory.get_breeds(split='good')

    if args.dataset not in ['living17', 'entity13', 'nonliving26', 'entity30']:
        train_loader = torch.utils.data.DataLoader(
                                dataset_train,
                                batch_size = args.batch_size,
                                shuffle = True,
                                num_workers = 2,
                                drop_last = True
                            )
        test_loader = torch.utils.data.DataLoader(
                                dataset_test,
                                batch_size = args.batch_size * 2,
                                shuffle = False,
                                num_workers = 2,
                                drop_last = True
                            )

    clip_model = torch.jit.load(os.path.join('pretrained', args.pretrain_clip), map_location="cpu").eval()
    if args.trainer == 'CoOp':
        design_details = {"trainer": 'CoOp',
                        "vision_depth": 0,
                        "language_depth": 0, "vision_ctx": 0,
                        "language_ctx": 0, "attention": 0}
    elif args.trainer == 'CoCoOp':
        design_details = {"trainer": 'CoCoOp',
                        "vision_depth": 0,
                        "language_depth": 0, "vision_ctx": 0,
                        "language_ctx": 0, "attention": 0}
    elif args.trainer == 'HGCLIP':
        design_details = {"trainer": 'HGCLIP',
                      "vision_depth": 9,
                      "language_depth": 9, "vision_ctx": 4,
                      "language_ctx": 4,
                      "attention": 1}
    elif args.trainer == 'VP':
        design_details = {"trainer": 'VP_CLIP',
                      "vision_depth": 9,
                      "language_depth": 0, "vision_ctx": 4,
                      "language_ctx": 0, "attention": 0}
    elif args.trainer == 'MaPLe':
        design_details = {"trainer": 'MaPLe',
                        "vision_depth": 9,
                        "language_depth": 9, "vision_ctx": 4,
                        "language_ctx": 4,
                        "maple_length": 4, "attention": 0}
    clip_model = clip.build_model(clip_model.state_dict(), design_details)

    if args.dataset == 'cifar-100':
        def load_labels_name(filename):
            with open(filename, 'rb') as f:
                obj = pickle.load(f)
            return obj
        labels_name_dic = load_labels_name('data/cifar-100-python/meta')
        labels_name_fine = labels_name_dic['fine_label_names']
        labels_name_coarse = labels_name_dic['coarse_label_names']
        fine2coarse = pd.read_pickle('data/cifar-100-python/label.pkl')
        mapping_dict = map_dic(fine2coarse)
        mapping_dict_name = {}
        for i in mapping_dict.keys():
            mapping_dict_name[labels_name_coarse[i]] = [labels_name_fine[l] for l in mapping_dict[i]]
        x = []
        y = []
        for fine in fine2coarse.keys():
            y.append(fine+20)
            x.append(fine2coarse[fine])
    
    elif args.dataset == 'ethec':
        label_map = ETHECLabelMap()
        labels_name_family = [i.split('_')[-1].lower() for i in list(label_map.family.keys())]
        labels_name_subfamily = [i.split('_')[-1].lower() for i in list(label_map.subfamily.keys())]
        labels_name_genus = [i.split('_')[-1].lower() for i in list(label_map.genus.keys())]
        labels_name_specific_epithet = [i.split('_')[-1].lower() for i in list(label_map.genus_specific_epithet.keys())]
        mapping_dict_genus = label_map.child_of_genus
        mapping_dict_subfamily = label_map.child_of_subfamily
        mapping_dict_family = label_map.child_of_family
        x = []
        y = []
        for family in mapping_dict_family.keys():
            for subfamily in mapping_dict_family[family]:
                subfamily = subfamily.split('_')[-1].lower()
                x.append(labels_name_family.index(family.lower()))
                y.append(labels_name_subfamily.index(subfamily.lower())+len(labels_name_family))
        for subfamily in mapping_dict_subfamily.keys():
            for genus in mapping_dict_subfamily[subfamily]:
                genus = genus.split('_')[-1].lower()
                x.append(labels_name_subfamily.index(subfamily.lower())+len(labels_name_family))
                y.append(labels_name_genus.index(genus.lower())+len(labels_name_subfamily)+len(labels_name_family))
        for genus in mapping_dict_genus.keys():
            for specific_epithet in mapping_dict_genus[genus]:
                specific_epithet = specific_epithet.split('_')[-1].lower()
                x.append(labels_name_genus.index(genus.lower())+len(labels_name_family)+len(labels_name_subfamily))
                y.append(labels_name_specific_epithet.index(specific_epithet.lower())+len(labels_name_genus)+len(labels_name_family)+len(labels_name_subfamily))

    elif args.dataset == 'car':
        label_map = StanfordCarsLabelMap(root = "data/")
        labels_name_fine = label_map.fine_classes
        labels_name_coarse = label_map.coarse_classes
        x = []
        y = []
        for rel in label_map.trees:
            fine, coarse = rel
            x.append(coarse-1)
            y.append(fine-1+9)
    
    elif args.dataset == 'air':
        label_map = AircraftMap(data_dir = "data/fgvc-aircraft-2013b/data/")
        labels_names_variant = label_map.labels_names_variant
        labels_names_family = label_map.labels_names_family
        labels_names_maker = label_map.labels_names_maker
        x = []
        y = []
        for rel in label_map.trees:
            models, families, makers = rel
            x.append(makers-1)
            y.append(families-1+30)
            x.append(families-1+30)
            y.append(models-1+100)

    elif args.dataset == 'caltech-101' or args.dataset == 'fruits-360':
        labels_names_l2, labels_names_l1 = dataset_train.load_data()
        print(f'level 1: {len(labels_names_l1)} level 2: {len(labels_names_l2)} level 3: {len(labels_names_l3)}')
        x = []
        y = []
        if args.dataset == 'food-101':
            hierarchy_file = 'data/food-101/food-101/food-101_hierarchy.txt'
        elif args.dataset == 'fruits-360':
            hierarchy_file = 'data/fruits-360/fruits-360_dataset/fruit-360_hierarchy.txt'
        else:
            hierarchy_file = f"data/{args.dataset.replace('-','')}/{args.dataset.replace('-','')}-hierarchy.txt"
        with open(hierarchy_file, 'r') as f:
            for i in f.readlines():
                level3, level2, level1 = i.split(',')
                x.append(labels_names_l1.index(level1.lower().strip().replace('_',' ').replace('-',' ')))
                y.append(labels_names_l2.index(level2.lower().strip().replace('_',' ').replace('-',' '))+len(labels_names_l1))
                x.append(labels_names_l2.index(level2.lower().strip().replace('_',' ').replace('-',' '))+len(labels_names_l1))
                y.append(labels_names_l3.index(level3.lower().strip().replace('_',' ').replace('-',' '))+len(labels_names_l1)+len(labels_names_l2))
    
    elif args.dataset == 'food-101' or args.dataset == 'pets':
        labels_names_l1 = dataset_train.load_data()
        print(f'level 1: {len(labels_names_l1)} level 2: {len(labels_names_l2)}')
        x = []
        y = []
        if args.dataset == 'food-101':
            hierarchy_file = 'data/food-101/food-101/food-101_hierarchy.txt'
        elif args.dataset == 'pets':
            hierarchy_file = 'data/oxford-iiit-pet/pets-hierarchy.txt'
        with open(hierarchy_file, 'r') as f:
            for i in f.readlines():
                level2, level1 = i.split(',')
                x.append(labels_names_l1.index(level1.lower().strip().replace('_',' ').replace('-',' ')))
                y.append(labels_names_l2.index(level2.lower().strip().replace('_',' ').replace('-',' '))+len(labels_names_l1))
    
    elif args.dataset in ['living17', 'entity13', 'nonliving26', 'entity30']:
        labels_name = list(breeds_factory.label_map.values())
        with open(f'data/ImageNet/{args.dataset}_fineclassnames.json', encoding='utf-8') as f:
            labels_name_fine = json.load(f)
        print(f'{len(labels_name)} {len(labels_name_fine)}')
        labels_name = labels_name + labels_name_fine
        with open(f'data/ImageNet/{args.dataset}_coarse2fine.json', encoding='utf-8') as f:
            coarse2fine = json.load(f)
        x = []
        y = []
        for k,v in coarse2fine.items():
            for j in range(len(v)):
                x.append(int(k))
            y.extend(v)

    # Create a graph
    edges = torch.tensor(x), torch.tensor(y)
    g = dgl.graph(edges)
    g = dgl.add_self_loop(g)
    g = g.to(args.device)

    if args.dataset == 'cifar-100' or args.dataset == 'car':
        labels_name = labels_name_coarse + labels_name_fine
    elif args.dataset == 'ethec':
        labels_name = labels_name_family + labels_name_subfamily + labels_name_genus + labels_name_specific_epithet
    elif args.dataset == 'air':
        labels_name = labels_names_maker + labels_names_family + labels_names_variant
    elif args.dataset == 'caltech-101' or args.dataset == 'fruits-360':
        labels_name = labels_names_l1 + labels_names_l2 + labels_names_l3
    elif args.dataset == 'food-101' or args.dataset == 'pets':
        labels_name = labels_names_l1 + labels_names_l2

    print("Building custom CLIP")
    if args.trainer == 'CoOp':
        model = CoOp_CLIP(args, clip_model, labels_name)
    elif args.trainer == 'CoCoOp':
        model = CoCoOp_CLIP(args, clip_model, labels_name)
    elif args.trainer == 'MaPLe':
        model = MaPLe(args, clip_model, labels_name)
    elif args.trainer == 'HGCLIP':
        model = HGCLIP(args, g, clip_model, labels_name, design_details)
    elif args.trainer == 'VP':
        model = VP_CLIP(args, g, clip_model, labels_name, design_details)
    model.to(args.device)

    print("Turning off gradients in both the image and the text encoder")
    for name, param in model.named_parameters():
        if "prompt_learner" not in name and "gnn" not in name:
            if "VPT" in name:
                param.requires_grad_(True)
            else:
                param.requires_grad_(False)

    # Double check
    enabled = set()
    for name, param in model.named_parameters():
        if param.requires_grad:
            enabled.add(name)
    print(f"Parameters to be updated: {enabled}")

    total = sum([param.nelement() for param in model.parameters()])
    total_update = sum([param.nelement() for param in [p for p in model.parameters() if p.requires_grad]])
    print("Number of parameter: %.2fM" % (total/1e6))
    print("Number of parameter to be updated: %.2fM" % (total_update/1e6))

    if args.optimizer == 'sgd':
        optimizer = optim.SGD(params=[p for p in model.parameters() if p.requires_grad], lr=args.lr, momentum=0.9, weight_decay=1e-4)
         
    elif args.optimizer == 'adam':
        optimizer = optim.Adam(params=[p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=1e-4) 

    elif args.optimizer == 'adamw':
        optimizer = optim.AdamW(params=[p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=1e-4)

    exp_lr_scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, patience = 5, mode = 'max', verbose = True, min_lr = 1e-7)

    loss_function = nn.CrossEntropyLoss()

    best_acc = 0.0
    if design_details['attention'] == 1:
        with open(f'{args.dataset}_prototypes.pkl', 'rb') as f:
            class_image_features = pickle.load(f)
        class_image_features = class_image_features.to(args.device)
    for epoch in range(args.epochs):
        if args.dataset == 'cifar-100':
            top1_fine, top5_fine, top1_coarse, top5_coarse, n = 0., 0., 0., 0., 0.
            model.train()
            for data in tqdm.tqdm(train_loader):
                inputs, labels = data

                label_coarse = torch.tensor([fine2coarse[int(i)] for i in labels]).to(args.device)
                label_fine = labels.to(args.device)

                inputs = Variable(inputs.to(args.device))

                optimizer.zero_grad()

                if design_details['attention'] == 1:
                    outputs_coarse, outputs_fine = model(inputs, class_image_features)
                else:
                    outputs_coarse, outputs_fine = model(inputs)
                
                loss_coarse = loss_function(outputs_coarse, label_coarse)
                loss_fine = loss_function(outputs_fine, label_fine)
                loss = (loss_coarse + 2 * loss_fine) / 2
                loss.backward(retain_graph=True)
                optimizer.step()

                acc1_fine, acc5_fine = acc(outputs_fine, label_fine, topk=(1, 5))
                acc1_coarse, acc5_coarse = acc(outputs_coarse, label_coarse, topk=(1, 5))
                top1_fine += acc1_fine
                top5_fine += acc5_fine
                top1_coarse += acc1_coarse
                top5_coarse += acc5_coarse
                n += inputs.size(0)

            top1_fine = (top1_fine / n) * 100
            top5_fine = (top5_fine / n) * 100
            top1_coarse = (top1_coarse / n) * 100
            top5_coarse = (top5_coarse / n) * 100
            print("train epoch[{}/{}] loss:{:.3f} Fine Top-1:{:.2f} Top-5: {:.2f} Coarse Top-1:{:.2f} Top-5:{:.2f}".format(epoch + 1, args.epochs, loss, top1_fine, top5_fine, top1_coarse, top5_coarse))

            model.eval()

            with torch.no_grad():
                top1_fine, top5_fine, top1_coarse, top5_coarse, n = 0., 0., 0., 0., 0.
                for data in tqdm.tqdm(test_loader):
                    inputs, labels = data
                    
                    label_coarse = torch.tensor([fine2coarse[int(i)] for i in labels]).to(args.device)
                    label_fine = labels.to(args.device)

                    inputs = Variable(inputs.to(args.device))

                    if design_details['attention'] == 1:
                        outputs_coarse, outputs_fine = model(inputs, class_image_features)
                    else:
                        outputs_coarse, outputs_fine = model(inputs)

                    loss_coarse = loss_function(outputs_coarse, label_coarse)
                    loss_fine = loss_function(outputs_fine, label_fine)
                    loss = (loss_coarse + loss_fine) / 2

                    acc1_fine, acc5_fine = acc(outputs_fine, label_fine, topk=(1, 5))
                    acc1_coarse, acc5_coarse = acc(outputs_coarse, label_coarse, topk=(1, 5))
                    top1_fine += acc1_fine
                    top5_fine += acc5_fine
                    top1_coarse += acc1_coarse
                    top5_coarse += acc5_coarse
                    n += inputs.size(0)

                top1_fine = (top1_fine / n) * 100
                top5_fine = (top5_fine / n) * 100
                top1_coarse = (top1_coarse / n) * 100
                top5_coarse = (top5_coarse / n) * 100
                print("test epoch[{}/{}] loss:{:.3f} Fine Top-1:{:.2f} Top-5: {:.2f} Coarse Top-1:{:.2f} Top-5:{:.2f}".format(epoch + 1, args.epochs, loss, top1_fine, top5_fine, top1_coarse, top5_coarse))
                current_acc = (top1_fine + top1_coarse) / 2
                exp_lr_scheduler.step(current_acc)
                if current_acc > best_acc:
                    best_acc = current_acc
                    checkpoint_path = f'checkpoint/{args.dataset}_ViT-B-16_{args.trainer}.pt'
                    torch.save(model.state_dict(), checkpoint_path)
                    print(f'checkpoint saved at: {checkpoint_path}')

        elif args.dataset == 'ethec':
            top1_specific_epithet, top5_specific_epithet, top1_genus, top5_genus, top1_subfamily, top5_subfamily, top1_family, top5_family, n = 0., 0., 0., 0., 0., 0., 0., 0., 0.
            model.train()
            for data in tqdm.tqdm(train_loader):
                inputs, labels_family, labels_subfamily, labels_genus, labels_specific_epithet = data

                labels_specific_epithet = [i.split("_")[-1].lower() for i in labels_specific_epithet]
                labels_genus = [i.split("_")[-1].lower() for i in labels_genus]
                labels_subfamily = [i.split("_")[-1].lower() for i in labels_subfamily]
                labels_family = [i.split("_")[-1].lower() for i in labels_family]

                label_family = torch.tensor([int(labels_name_family.index(l.lower())) for l in labels_family]).to(args.device)
                label_subfamily = torch.tensor([int(labels_name_subfamily.index(l.lower())) for l in labels_subfamily]).to(args.device)
                label_genus = torch.tensor([int(labels_name_genus.index(l.lower())) for l in labels_genus]).to(args.device)
                label_specific_epithet = torch.tensor([int(labels_name_specific_epithet.index(l.lower())) for l in labels_specific_epithet]).to(args.device)

                inputs = Variable(inputs.to(args.device))

                optimizer.zero_grad()
                if design_details['attention'] == 1:
                    outputs_family, outputs_subfamily, outputs_genus, outputs_specific_epithet = model(inputs, class_image_features)
                else:
                    outputs_family, outputs_subfamily, outputs_genus, outputs_specific_epithet = model(inputs)
                
                loss_family = loss_function(outputs_family, label_family)
                loss_subfamily = loss_function(outputs_subfamily, label_subfamily)
                loss_genus = loss_function(outputs_genus, label_genus)
                loss_specific_epithet = loss_function(outputs_specific_epithet, label_specific_epithet)
                loss = (loss_family + loss_subfamily + loss_genus + loss_specific_epithet * 2) / 4
                loss.backward(retain_graph=True)

                optimizer.step()

                acc1_family, acc5_family = acc(outputs_family, label_family, topk=(1, 5))
                acc1_subfamily, acc5_subfamily = acc(outputs_subfamily, label_subfamily, topk=(1, 5))
                acc1_genus, acc5_genus = acc(outputs_genus, label_genus, topk=(1, 5))
                acc1_specific_epithet, acc5_specific_epithet = acc(outputs_specific_epithet, label_specific_epithet, topk=(1, 5))
                top1_family += acc1_family
                top5_family += acc5_family
                top1_subfamily += acc1_subfamily
                top5_subfamily += acc5_subfamily
                top1_genus += acc1_genus
                top5_genus += acc5_genus
                top1_specific_epithet += acc1_specific_epithet
                top5_specific_epithet += acc5_specific_epithet
                n += inputs.size(0)

            top1_family = (top1_family / n) * 100
            top5_family = (top5_family / n) * 100
            top1_subfamily = (top1_subfamily / n) * 100
            top5_subfamily = (top5_subfamily / n) * 100
            top1_genus = (top1_genus / n) * 100
            top5_genus = (top5_genus / n) * 100
            top1_specific_epithet = (top1_specific_epithet / n) * 100
            top5_specific_epithet = (top5_specific_epithet / n) * 100
            print("train epoch[{}/{}] loss:{:.3f} family Top-1:{:.2f} Top-5: {:.2f} subfamily Top-1:{:.2f} Top-5:{:.2f} genus Top-1:{:.2f} Top-5:{:.2f} specific_epithet Top-1:{:.2f} Top-5:{:.2f}".format(epoch + 1, args.epochs, loss, top1_family, top5_family, top1_subfamily, top5_subfamily, top1_genus, top5_genus, top1_specific_epithet, top5_specific_epithet))

            model.eval()

            with torch.no_grad():
                top1_specific_epithet, top5_specific_epithet, top1_genus, top5_genus, top1_subfamily, top5_subfamily, top1_family, top5_family, n = 0., 0., 0., 0., 0., 0., 0., 0., 0.
                for data in tqdm.tqdm(test_loader):
                    inputs, labels_family, labels_subfamily, labels_genus, labels_specific_epithet = data

                    labels_specific_epithet = [i.split("_")[-1].lower() for i in labels_specific_epithet]
                    labels_genus = [i.split("_")[-1].lower() for i in labels_genus]
                    labels_subfamily = [i.split("_")[-1].lower() for i in labels_subfamily]
                    labels_family = [i.split("_")[-1].lower() for i in labels_family]

                    label_family = torch.tensor([int(labels_name_family.index(l.lower())) for l in labels_family]).to(args.device)
                    label_subfamily = torch.tensor([int(labels_name_subfamily.index(l.lower())) for l in labels_subfamily]).to(args.device)
                    label_genus = torch.tensor([int(labels_name_genus.index(l.lower())) for l in labels_genus]).to(args.device)
                    label_specific_epithet = torch.tensor([int(labels_name_specific_epithet.index(l.lower())) for l in labels_specific_epithet]).to(args.device)

                    inputs = Variable(inputs.to(args.device))

                    if design_details['attention'] == 1:
                        outputs_family, outputs_subfamily, outputs_genus, outputs_specific_epithet = model(inputs, class_image_features)
                    else:
                        outputs_family, outputs_subfamily, outputs_genus, outputs_specific_epithet = model(inputs)
                    
                    loss_family = loss_function(outputs_family, label_family)
                    loss_subfamily = loss_function(outputs_subfamily, label_subfamily)
                    loss_genus = loss_function(outputs_genus, label_genus)
                    loss_specific_epithet = loss_function(outputs_specific_epithet, label_specific_epithet)
                    loss = (loss_family + loss_subfamily + loss_genus + loss_specific_epithet * 2) / 4

                    acc1_family, acc5_family = acc(outputs_family, label_family, topk=(1, 5))
                    acc1_subfamily, acc5_subfamily = acc(outputs_subfamily, label_subfamily, topk=(1, 5))
                    acc1_genus, acc5_genus = acc(outputs_genus, label_genus, topk=(1, 5))
                    acc1_specific_epithet, acc5_specific_epithet = acc(outputs_specific_epithet, label_specific_epithet, topk=(1, 5))
                    top1_family += acc1_family
                    top5_family += acc5_family
                    top1_subfamily += acc1_subfamily
                    top5_subfamily += acc5_subfamily
                    top1_genus += acc1_genus
                    top5_genus += acc5_genus
                    top1_specific_epithet += acc1_specific_epithet
                    top5_specific_epithet += acc5_specific_epithet
                    n += inputs.size(0)

                top1_family = (top1_family / n) * 100
                top5_family = (top5_family / n) * 100
                top1_subfamily = (top1_subfamily / n) * 100
                top5_subfamily = (top5_subfamily / n) * 100
                top1_genus = (top1_genus / n) * 100
                top5_genus = (top5_genus / n) * 100
                top1_specific_epithet = (top1_specific_epithet / n) * 100
                top5_specific_epithet = (top5_specific_epithet / n) * 100
                print("test epoch[{}/{}] loss:{:.3f} family Top-1:{:.2f} Top-5: {:.2f} subfamily Top-1:{:.2f} Top-5:{:.2f} genus Top-1:{:.2f} Top-5:{:.2f} specific_epithet Top-1:{:.2f} Top-5:{:.2f}".format(epoch + 1, args.epochs, loss, top1_family, top5_family, top1_subfamily, top5_subfamily, top1_genus, top5_genus, top1_specific_epithet, top5_specific_epithet))
                current_acc = (top1_family + top1_subfamily + top1_genus + top1_specific_epithet) / 4
                exp_lr_scheduler.step(current_acc)
                if current_acc > best_acc:
                    best_acc = current_acc
                    checkpoint_path = f'checkpoint/{args.dataset}_ViT-B-16_{args.trainer}.pt'
                    torch.save(model.state_dict(), checkpoint_path)
                    print(f'checkpoint saved at: {checkpoint_path}')
        
        elif args.dataset == 'car':
            top1_fine, top5_fine, top1_coarse, top5_coarse, n = 0., 0., 0., 0., 0.
            model.train()
            for data in tqdm.tqdm(train_loader):
                inputs, labels_fine, labels_coarse = data

                label_coarse = labels_coarse.to(args.device)
                label_fine = labels_fine.to(args.device)

                inputs = Variable(inputs.to(args.device))

                optimizer.zero_grad()

                if design_details['attention'] == 1:
                    outputs_coarse, outputs_fine = model(inputs, class_image_features)
                else:
                    outputs_coarse, outputs_fine = model(inputs)
                
                loss_coarse = loss_function(outputs_coarse, label_coarse)
                loss_fine = loss_function(outputs_fine, label_fine)
                loss = (loss_coarse + 2 * loss_fine) / 2
                loss.backward(retain_graph=True)
                optimizer.step()

                acc1_fine, acc5_fine = acc(outputs_fine, label_fine, topk=(1, 5))
                acc1_coarse, acc5_coarse = acc(outputs_coarse, label_coarse, topk=(1, 5))
                top1_fine += acc1_fine
                top5_fine += acc5_fine
                top1_coarse += acc1_coarse
                top5_coarse += acc5_coarse
                n += inputs.size(0)

            top1_fine = (top1_fine / n) * 100
            top5_fine = (top5_fine / n) * 100
            top1_coarse = (top1_coarse / n) * 100
            top5_coarse = (top5_coarse / n) * 100
            print("train epoch[{}/{}] loss:{:.3f} Fine Top-1:{:.2f} Coarse Top-1:{:.2f}".format(epoch + 1, args.epochs, loss, top1_fine, top1_coarse))

            model.eval()

            with torch.no_grad():
                top1_fine, top5_fine, top1_coarse, top5_coarse, n = 0., 0., 0., 0., 0.
                for data in tqdm.tqdm(test_loader):
                    inputs, labels_fine, labels_coarse = data
                    
                    label_coarse = labels_coarse.to(args.device)
                    label_fine = labels_fine.to(args.device)

                    inputs = Variable(inputs.to(args.device))

                    if design_details['attention'] == 1:
                        outputs_coarse, outputs_fine = model(inputs, class_image_features)
                    else:
                        outputs_coarse, outputs_fine = model(inputs)

                    loss_coarse = loss_function(outputs_coarse, label_coarse)
                    loss_fine = loss_function(outputs_fine, label_fine)
                    loss = (loss_coarse + loss_fine) / 2

                    acc1_fine, acc5_fine = acc(outputs_fine, label_fine, topk=(1, 5))
                    acc1_coarse, acc5_coarse = acc(outputs_coarse, label_coarse, topk=(1, 5))
                    top1_fine += acc1_fine
                    top5_fine += acc5_fine
                    top1_coarse += acc1_coarse
                    top5_coarse += acc5_coarse
                    n += inputs.size(0)

                top1_fine = (top1_fine / n) * 100
                top5_fine = (top5_fine / n) * 100
                top1_coarse = (top1_coarse / n) * 100
                top5_coarse = (top5_coarse / n) * 100
                print("test epoch[{}/{}] loss:{:.3f} Fine Top-1:{:.2f} Coarse Top-1:{:.2f}".format(epoch + 1, args.epochs, loss, top1_fine, top1_coarse))
                current_acc = (top1_fine + top1_coarse) / 2
                exp_lr_scheduler.step(current_acc)
                if current_acc > best_acc:
                    best_acc = current_acc
                    checkpoint_path = f'checkpoint/{args.dataset}_ViT-B-16_{args.trainer}.pt'
                    torch.save(model.state_dict(), checkpoint_path)
                    print(f'checkpoint saved at: {checkpoint_path}')
    
        elif args.dataset == 'air':
            top1_model, top1_family, top1_maker, n = 0., 0., 0., 0.
            model.train()
            for data in tqdm.tqdm(train_loader):
                inputs, labels_model, labels_family, labels_maker = data

                label_model = torch.tensor([int(labels_names_variant.index(l)) for l in labels_model]).to(args.device)
                label_family = torch.tensor([int(labels_names_family.index(l)) for l in labels_family]).to(args.device)
                label_maker = torch.tensor([int(labels_names_maker.index(l)) for l in labels_maker]).to(args.device)

                inputs = Variable(inputs.to(args.device))

                optimizer.zero_grad()

                if design_details['attention'] == 1:
                    outputs_model, outputs_family, outputs_maker = model(inputs, class_image_features)
                else:
                    outputs_model, outputs_family, outputs_maker = model(inputs)
                
                loss_model = loss_function(outputs_model, label_model)
                loss_family = loss_function(outputs_family, label_family)
                loss_maker = loss_function(outputs_maker, label_maker)
                loss = (2*loss_model + loss_family + loss_maker) / 3
                loss.backward(retain_graph=True)
                optimizer.step()

                acc1_model = acc(outputs_model, label_model, topk=(1,))[0]
                acc1_family = acc(outputs_family, label_family, topk=(1,))[0]
                acc1_maker = acc(outputs_maker, label_maker, topk=(1,))[0]
                top1_model += acc1_model
                top1_family += acc1_family
                top1_maker += acc1_maker
                n += inputs.size(0)

            top1_model = (top1_model / n) * 100
            top1_family = (top1_family / n) * 100
            top1_maker = (top1_maker / n) * 100
            print("train epoch[{}/{}] loss:{:.3f} model:{:.2f} family:{:.2f} maker:{:.2f}".format(epoch + 1, args.epochs, loss, top1_model, top1_family, top1_maker))

            model.eval()

            with torch.no_grad():
                top1_model, top1_family, top1_maker, n = 0., 0., 0., 0.
                for data in tqdm.tqdm(test_loader):
                    inputs, labels_model, labels_family, labels_maker = data

                    label_model = torch.tensor([int(labels_names_variant.index(l)) for l in labels_model]).to(args.device)
                    label_family = torch.tensor([int(labels_names_family.index(l)) for l in labels_family]).to(args.device)
                    label_maker = torch.tensor([int(labels_names_maker.index(l)) for l in labels_maker]).to(args.device)

                    inputs = Variable(inputs.to(args.device))

                    if design_details['attention'] == 1:
                        outputs_model, outputs_family, outputs_maker = model(inputs, class_image_features)
                    else:
                        outputs_model, outputs_family, outputs_maker = model(inputs)

                    loss_model = loss_function(outputs_model, label_model)
                    loss_family = loss_function(outputs_family, label_family)
                    loss_maker = loss_function(outputs_maker, label_maker)
                    loss = (2*loss_model + loss_family + loss_maker) / 3

                    acc1_model = acc(outputs_model, label_model, topk=(1,))[0]
                    acc1_family = acc(outputs_family, label_family, topk=(1,))[0]
                    acc1_maker = acc(outputs_maker, label_maker, topk=(1,))[0]
                    top1_model += acc1_model
                    top1_family += acc1_family
                    top1_maker += acc1_maker
                    n += inputs.size(0)

                top1_model = (top1_model / n) * 100
                top1_family = (top1_family / n) * 100
                top1_maker = (top1_maker / n) * 100
                print("test epoch[{}/{}] loss:{:.3f} model:{:.2f} family:{:.2f} maker:{:.2f}".format(epoch + 1, args.epochs, loss, top1_model, top1_family, top1_maker))
                current_acc = (top1_model + top1_family + top1_maker ) / 3
                exp_lr_scheduler.step(current_acc)
                if current_acc > best_acc:
                    best_acc = current_acc
                    checkpoint_path = f'checkpoint/{args.dataset}_ViT-B-16_GCN_3layers_{args.trainer}.pt'
                    torch.save(model.state_dict(), checkpoint_path)
                    print(f'checkpoint saved at: {checkpoint_path}')
    
        elif args.dataset == 'caltech-101' or args.dataset == 'fruits-360':
            top1_l1, top1_l2, top1_l3, n = 0., 0., 0., 0.
            model.train()
            for data in tqdm.tqdm(train_loader):
                inputs, labels_l3, labels_l2, labels_l1 = data

                label_l3 = torch.tensor([int(labels_names_l3.index(l)) for l in labels_l3]).to(args.device)
                label_l2 = torch.tensor([int(labels_names_l2.index(l)) for l in labels_l2]).to(args.device)
                label_l1 = torch.tensor([int(labels_names_l1.index(l)) for l in labels_l1]).to(args.device)
                inputs = Variable(inputs.to(args.device))

                optimizer.zero_grad()

                if design_details['attention'] == 1:
                    outputs_l1, outputs_l2, outputs_l3 = model(inputs, class_image_features)
                else:
                    outputs_l1, outputs_l2, outputs_l3 = model(inputs)
                
                loss_l1 = loss_function(outputs_l1, label_l1)
                loss_l2 = loss_function(outputs_l2, label_l2)
                loss_l3 = loss_function(outputs_l3, label_l3)
                loss = (2*loss_l3 + loss_l2 + loss_l1) / 3
                loss.backward(retain_graph=True)
                optimizer.step()

                acc1_l1 = acc(outputs_l1, label_l1, topk=(1,))[0]
                acc1_l2 = acc(outputs_l2, label_l2, topk=(1,))[0]
                acc1_l3 = acc(outputs_l3, label_l3, topk=(1,))[0]
                top1_l1 += acc1_l1
                top1_l2 += acc1_l2
                top1_l3 += acc1_l3
                n += inputs.size(0)

            top1_l1 = (top1_l1 / n) * 100
            top1_l2 = (top1_l2 / n) * 100
            top1_l3 = (top1_l3 / n) * 100
            print("train epoch[{}/{}] loss:{:.3f} l1:{:.2f} l2:{:.2f} l3:{:.2f}".format(epoch + 1, args.epochs, loss, top1_l1, top1_l2, top1_l3))

            model.eval()

            with torch.no_grad():
                top1_l1, top1_l2, top1_l3, n = 0., 0., 0., 0.
                for data in tqdm.tqdm(test_loader):
                    inputs, labels_l3, labels_l2, labels_l1 = data

                    label_l3 = torch.tensor([int(labels_names_l3.index(l)) for l in labels_l3]).to(args.device)
                    label_l2 = torch.tensor([int(labels_names_l2.index(l)) for l in labels_l2]).to(args.device)
                    label_l1 = torch.tensor([int(labels_names_l1.index(l)) for l in labels_l1]).to(args.device)
                    inputs = Variable(inputs.to(args.device))

                    if design_details['attention'] == 1:
                        outputs_l1, outputs_l2, outputs_l3 = model(inputs, class_image_features)
                    else:
                        outputs_l1, outputs_l2, outputs_l3 = model(inputs)

                    loss_l1 = loss_function(outputs_l1, label_l1)
                    loss_l2 = loss_function(outputs_l2, label_l2)
                    loss_l3 = loss_function(outputs_l3, label_l3)
                    loss = (2*loss_l3 + loss_l2 + loss_l1) / 3

                    acc1_l1 = acc(outputs_l1, label_l1, topk=(1,))[0]
                    acc1_l2 = acc(outputs_l2, label_l2, topk=(1,))[0]
                    acc1_l3 = acc(outputs_l3, label_l3, topk=(1,))[0]
                    top1_l1 += acc1_l1
                    top1_l2 += acc1_l2
                    top1_l3 += acc1_l3
                    n += inputs.size(0)

                top1_l1 = (top1_l1 / n) * 100
                top1_l2 = (top1_l2 / n) * 100
                top1_l3 = (top1_l3 / n) * 100
                print("test epoch[{}/{}] loss:{:.3f} l1:{:.2f} l2:{:.2f} l3:{:.2f}".format(epoch + 1, args.epochs, loss, top1_l1, top1_l2, top1_l3))
                current_acc = (top1_l3 + top1_l2 + top1_l1 ) / 3
                exp_lr_scheduler.step(current_acc)
                if current_acc > best_acc:
                    best_acc = current_acc
                    checkpoint_path = f'checkpoint/{args.dataset}_ViT-B-16_{args.trainer}.pt'
                    torch.save(model.state_dict(), checkpoint_path)
                    print(f'checkpoint saved at: {checkpoint_path}')

        elif args.dataset == 'food-101' or args.dataset == 'pets':
            top1_l1, top1_l2, n = 0., 0., 0.
            model.train()
            for data in tqdm.tqdm(train_loader):
                inputs, labels_l2, labels_l1 = data

                label_l2 = torch.tensor([int(labels_names_l2.index(l)) for l in labels_l2]).to(args.device)
                label_l1 = torch.tensor([int(labels_names_l1.index(l)) for l in labels_l1]).to(args.device)
                inputs = Variable(inputs.to(args.device))

                optimizer.zero_grad()

                if design_details['attention'] == 1:
                    outputs_l1, outputs_l2 = model(inputs, class_image_features)
                else:
                    outputs_l1, outputs_l2 = model(inputs)
                
                loss_l1 = loss_function(outputs_l1, label_l1)
                loss_l2 = loss_function(outputs_l2, label_l2)
                loss = (2*loss_l2 + loss_l1) / 2
                loss.backward(retain_graph=True)
                optimizer.step()

                acc1_l1 = acc(outputs_l1, label_l1, topk=(1,))[0]
                acc1_l2 = acc(outputs_l2, label_l2, topk=(1,))[0]
                top1_l1 += acc1_l1
                top1_l2 += acc1_l2
                n += inputs.size(0)

            top1_l1 = (top1_l1 / n) * 100
            top1_l2 = (top1_l2 / n) * 100
            print("train epoch[{}/{}] loss:{:.3f} l1:{:.2f} l2:{:.2f}".format(epoch + 1, args.epochs, loss, top1_l1, top1_l2))

            model.eval()

            with torch.no_grad():
                top1_l1, top1_l2, n = 0., 0., 0.
                for data in tqdm.tqdm(test_loader):
                    inputs, labels_l2, labels_l1 = data

                    label_l2 = torch.tensor([int(labels_names_l2.index(l)) for l in labels_l2]).to(args.device)
                    label_l1 = torch.tensor([int(labels_names_l1.index(l)) for l in labels_l1]).to(args.device)
                    inputs = Variable(inputs.to(args.device))

                    if design_details['attention'] == 1:
                        outputs_l1, outputs_l2 = model(inputs, class_image_features)
                    else:
                        outputs_l1, outputs_l2 = model(inputs)

                    loss_l1 = loss_function(outputs_l1, label_l1)
                    loss_l2 = loss_function(outputs_l2, label_l2)
                    loss = (2*loss_l2 + loss_l1) / 2

                    acc1_l1 = acc(outputs_l1, label_l1, topk=(1,))[0]
                    acc1_l2 = acc(outputs_l2, label_l2, topk=(1,))[0]
                    top1_l1 += acc1_l1
                    top1_l2 += acc1_l2
                    n += inputs.size(0)

                top1_l1 = (top1_l1 / n) * 100
                top1_l2 = (top1_l2 / n) * 100
                print("test epoch[{}/{}] loss:{:.3f} l1:{:.2f} l2:{:.2f}".format(epoch + 1, args.epochs, loss, top1_l1, top1_l2))
                current_acc = (top1_l1 + top1_l2) / 2
                exp_lr_scheduler.step(current_acc)
                if current_acc > best_acc:
                    best_acc = current_acc
                    checkpoint_path = f'checkpoint/{args.dataset}_ViT-B-16_{args.trainer}.pt'
                    torch.save(model.state_dict(), checkpoint_path)
                    print(f'checkpoint saved at: {checkpoint_path}')
    
        elif args.dataset in ['living17', 'entity13', 'nonliving26', 'entity30']:
            top1_source, top1_target, n = 0., 0., 0.
            model.train()
            for data in tqdm.tqdm(train_loader):
                inputs, labels = data

                labels = labels.to(args.device)
                inputs = Variable(inputs.to(args.device))

                optimizer.zero_grad()

                if design_details['attention'] == 1:
                    outputs = model(inputs, class_image_features)
                else:
                    outputs = model(inputs)

                loss = loss_function(outputs, labels)
                loss.backward(retain_graph=True)
                optimizer.step()

                acc1_source = acc(outputs, labels, topk=(1,))[0]
                top1_source += acc1_source
                n += inputs.size(0)

            top1_source = (top1_source / n) * 100
            print("train epoch[{}/{}] loss:{:.3f} source:{:.2f}".format(epoch + 1, args.epochs, loss, top1_source))

            model.eval()

            with torch.no_grad():
                top1_source, n = 0., 0.
                for data in tqdm.tqdm(test_loader_source):
                    inputs, labels = data

                    labels = labels.to(args.device)
                    inputs = Variable(inputs.to(args.device))

                    if design_details['attention'] == 1:
                        outputs = model(inputs, class_image_features)
                    else:
                        outputs = model(inputs)

                    loss = loss_function(outputs, labels)

                    acc1_source = acc(outputs, labels, topk=(1,))[0]
                    top1_source += acc1_source
                    n += inputs.size(0)

                top1_source = (top1_source / n) * 100
                print("test epoch[{}/{}] loss:{:.3f} top1 source:{:.2f}".format(epoch + 1, args.epochs, loss, top1_source))
                current_acc = top1_source
                exp_lr_scheduler.step(current_acc)
                if current_acc > best_acc:
                    best_acc = current_acc
                    checkpoint_path = f'checkpoint/{args.dataset}_ViT-B-16_{args.trainer}.pt'
                    torch.save(model.state_dict(), checkpoint_path)
                    print(f'checkpoint saved at: {checkpoint_path}')
            with torch.no_grad():
                top1_target, n = 0., 0.
                for data in tqdm.tqdm(test_loader_target):
                    inputs, labels = data

                    labels = labels.to(args.device)
                    inputs = Variable(inputs.to(args.device))

                    if design_details['attention'] == 1:
                        outputs = model(inputs, class_image_features)
                    else:
                        outputs = model(inputs)
                    acc1_target = acc(outputs, labels, topk=(1,))[0]
                    top1_target += acc1_target
                    n += inputs.size(0)
                top1_target = (top1_target / n) * 100
                print("test epoch[{}/{}] top1 target:{:.2f}".format(epoch + 1, args.epochs, top1_target))

if __name__ == '__main__':
    parser = argparse.ArgumentParser('HGCLIP', parents=[get_args_parser()])
    args = parser.parse_args()
    args = update_from_config(args)
    main(args)
