import os
from PIL import Image
import clip
import torch
import argparse
import warnings
import tqdm
import numpy as np
import pandas as pd
import pickle
import torchvision
import torch.nn as nn
from torch.nn import functional as F
from torch.autograd import Variable
import torchvision.transforms as transforms
from hierarchical_prompt.utils import *
from sklearn.model_selection import train_test_split

warnings.filterwarnings('ignore')

def get_args_parser():
    parser = argparse.ArgumentParser('generate visual prototypes of all categories', add_help=False)
    parser.add_argument('--config', required=False, type=str, help='config')
    parser.add_argument('--device', default='cuda', help='device to use for training / testing')
    parser.add_argument('--gpu_id', default=1, type=int, help='gpu id')
    parser.add_argument('--seed', default='0', type=int, help='seed')
    parser.add_argument('--dataset', default='cifar-100', type=str, help='dataset name')
    parser.add_argument('--pretrain_clip', default='ViT-B-16.pt', type=str, help='path of pretrained clip ckpt')
    parser.add_argument('--batch_size', default=1300, type=int, help='batch size')
    return parser

def main(args):
    print(args)

    assert torch.cuda.is_available()

    torch.cuda.set_device(args.gpu_id)

    torch.manual_seed(args.seed)

    clip_model, _ = clip.load(os.path.join('hgclip/pretrained', args.pretrain_clip), device='cpu', jit=False)
    clip_model.to(args.device)

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
        dataset_train = torchvision.datasets.CIFAR100('hgclip/data', train = True, download = False, transform = preprocess['train'])
    elif args.dataset == 'ethec':
        dataset_train = ETHECDataset(root_dir='hgclip/data/ETHEC_dataset', split='train', transform=preprocess['train'])
    elif args.dataset == 'car':
        dataset_train = StanfordCars(root = "hgclip/data/", split = "train", transform=preprocess['train'])
    elif args.dataset == 'air':
        dataset_train = AircraftDataset(data_dir = "hgclip/data/fgvc-aircraft-2013b/data/", split = "train", transform=preprocess['train'])
    elif args.dataset == 'caltech-101':
        dataset = torchvision.datasets.ImageFolder('hgclip/data/caltech101/101_ObjectCategories')
        labels_names_l3 = [i.lower().strip().replace('_',' ') for i in list(dataset.class_to_idx.keys())]
        dataset_train, _ = train_test_split(dataset, test_size=0.2, random_state=42)
        dataset_train = caltech(dataset_train, preprocess['train'], 'hgclip/data/caltech101/caltech101-hierarchy.txt', labels_names_l3)
    elif args.dataset == 'food-101':
        dataset = torchvision.datasets.ImageFolder('hgclip/data/food-101/food-101/images')
        labels_names_l2 = [i.lower().strip().replace('_',' ').replace('-',' ') for i in list(dataset.class_to_idx.keys())]
        dataset_train, _ = train_test_split(dataset, test_size=0.2, random_state=42)
        dataset_train = food101_dataset(dataset_train, preprocess['train'], 'hgclip/data/food-101/food-101/food-101_hierarchy.txt', labels_names_l2)
    elif args.dataset == 'fruits-360':
        dataset_train = torchvision.datasets.ImageFolder('hgclip/data/fruits-360/fruits-360_dataset/fruits-360/Training')
        labels_names_l3 = [i.lower().strip() for i in list(dataset_train.class_to_idx.keys())]
        dataset_train = caltech(dataset_train, preprocess['train'], 'hgclip/data/fruits-360/fruits-360_dataset/fruit-360_hierarchy.txt', labels_names_l3)
    elif args.dataset == 'pets':
        dataset_train = torchvision.datasets.OxfordIIITPet(root = 'hgclip/data', split = 'trainval')
        labels_names_l2 = [i.lower().strip().replace('_',' ').replace('-',' ') for i in list(dataset_train.class_to_idx)]
        dataset_train = food101_dataset(dataset_train, preprocess['train'], 'hgclip/data/oxford-iiit-pet/pets-hierarchy.txt', labels_names_l2)
    elif args.dataset == 'imagenet':
        dataset_train = torchvision.datasets.ImageFolder(root='hgclip/data/imagenet-314/train', transform=preprocess['train'])
        with open('hgclip/data/ImageNet/imagenet_class_index.json', 'r') as file:
            target_labels_data = json.load(file)
        target_labels = [label_info[0] for label_info in target_labels_data.values()]
        target_labels_name = [label_info[1].replace('_',' ') for label_info in target_labels_data.values()]
        # imagenet-a -r
        list_200_1 = os.listdir('hgclip/data/imagenet-adversarial/imagenet-a/')
        list_200_2 = os.listdir('hgclip/data/imagenet-rendition/imagenet-r/')
        list_all = list(set(list_200_1 + list_200_2))
        target_labels_name = [target_labels_name[target_labels.index(i)] for i in list_all]
        target_labels = list_all
        target_labels_name = [target_labels_name[target_labels.index(i)] for i in list(dataset_train.class_to_idx.keys())]

    train_loader = torch.utils.data.DataLoader(dataset_train, batch_size = args.batch_size, shuffle = True, num_workers = 2)

    if args.dataset == 'cifar-100':
        def load_labels_name(filename):
            with open(filename, 'rb') as f:
                obj = pickle.load(f)
            return obj
        labels_name_dic = load_labels_name('hgclip/data/cifar-100-python/meta')
        labels_name_fine = labels_name_dic['fine_label_names']
        labels_name_coarse = labels_name_dic['coarse_label_names']
        fine2coarse = pd.read_pickle('hgclip/data/cifar-100-python/label.pkl')
    elif args.dataset == 'ethec':
        label_map = ETHECLabelMap()
        labels_name_family = [i.split('_')[-1].lower() for i in list(label_map.family.keys())]
        labels_name_subfamily = [i.split('_')[-1].lower() for i in list(label_map.subfamily.keys())]
        labels_name_genus = [i.split('_')[-1].lower() for i in list(label_map.genus.keys())]
        labels_name_specific_epithet = [i.split('_')[-1].lower() for i in list(label_map.genus_specific_epithet.keys())]
        print(f'{len(labels_name_family)} {len(labels_name_subfamily)} {len(labels_name_genus)} {len(labels_name_specific_epithet)}')
        mapping_dict_genus = label_map.child_of_genus
        mapping_dict_subfamily = label_map.child_of_subfamily
        mapping_dict_family = label_map.child_of_family
    elif args.dataset == 'car':
        label_map = StanfordCarsLabelMap(root = "hgclip/data/")
        labels_name_fine = label_map.fine_classes
        labels_name_coarse = label_map.coarse_classes
        mapping_dict = label_map.trees
    elif args.dataset == 'air':
        label_map = AircraftMap(data_dir = "hgclip/data/fgvc-aircraft-2013b/data/")
        labels_name_variant = label_map.labels_names_variant
        labels_name_family = label_map.labels_names_family
        labels_name_maker = label_map.labels_names_maker
    elif args.dataset == 'caltech-101' or args.dataset == 'caltech-256' or args.dataset == 'fruits-360':
        labels_names_l2, labels_names_l1 = dataset_train.load_data()
        print(f'level 1: {len(labels_names_l1)} level 2: {len(labels_names_l2)} level 3: {len(labels_names_l3)}')
    elif args.dataset == 'food-101' or args.dataset == 'pets':
        labels_names_l1 = dataset_train.load_data()
        print(f'level 1: {len(labels_names_l1)} level 2: {len(labels_names_l2)}')
    
    if args.dataset == 'cifar-100' or args.dataset == 'car':
        labels_name = labels_name_coarse + labels_name_fine
    elif args.dataset == 'ethec':
        labels_name = labels_name_family + labels_name_subfamily + labels_name_genus + labels_name_specific_epithet
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
        print(f'has {len(word_id_new)} labels and {len(word_name_new)} label names')
        # node_degree = [0 for i in range(len(target_labels))]
        x = []
        y = []
        for i in hierarchy: # len(hierarchy) = 75850
            start, end = i
            if end in target_labels:
                # if node_degree[target_labels.index(end)] == 0:
                if start not in word_id_new:
                    word_id_new.append(start)
                    word_name_new.append(word_name[word_id.index(start)])
                x.append(word_id_new.index(start))
                y.append(word_id_new.index(end))
                # node_degree[target_labels.index(end)] += 1
        labels_name = word_name_new
        print(f'has {len(word_id_new)} labels and {len(word_name_new)} label names after adding hierarchies')

    elif args.dataset == 'air':
        labels_name = labels_name_maker + labels_name_family + labels_name_variant
    elif args.dataset == 'caltech-101' or args.dataset == 'fruits-360':
        labels_name = labels_names_l1 + labels_names_l2 + labels_names_l3
    elif args.dataset == 'food-101' or args.dataset == 'pets':
        labels_name = labels_names_l1 + labels_names_l2

    print("Building CLIP image encoder")
    image_encoder = clip_model.visual
    image_encoder.to(args.device)

    print("Turning off gradients in the image encoder")
    for name, param in image_encoder.named_parameters():
        param.requires_grad = False

    class_features = [[] for i in range(len(labels_name))]
    if args.dataset == 'cifar-100':
        for data in tqdm.tqdm(train_loader):
            inputs, labels = data

            label_coarse = torch.tensor([fine2coarse[int(i)] for i in labels]).to(args.device)
            label_fine = labels.to(args.device)
            inputs = Variable(inputs.to(args.device))

            image_features = image_encoder(inputs)

            for i, target in enumerate(label_fine):
                class_features[int(target.item())+20].append(image_features[i]) 
            for i, target in enumerate(label_coarse):
                class_features[int(target.item())].append(image_features[i])

        class_f_fine = torch.stack([torch.stack(i) for i in class_features[:20]]) # [20, 2500, 512]
        class_f_fine = torch.mean(class_f_fine, dim=1) # [20, 2500, 512] => [20, 512]
        class_f_coarse = torch.stack([torch.stack(i) for i in class_features[20:]]) # [100, 500, 512]
        class_f_coarse = torch.mean(class_f_coarse, dim=1) # [100, 500, 512] => [100, 512]

        class_features = torch.cat([class_f_fine, class_f_coarse], dim=0) # [120, 512]
    
    elif args.dataset == 'ethec':
        for data in tqdm.tqdm(train_loader):
            inputs, labels_family, labels_subfamily, labels_genus, labels_specific_epithet = data # 6 21 135 561

            labels_specific_epithet = [i.split("_")[-1].lower() for i in labels_specific_epithet]
            labels_genus = [i.split("_")[-1].lower() for i in labels_genus]
            labels_subfamily = [i.split("_")[-1].lower() for i in labels_subfamily]
            labels_family = [i.split("_")[-1].lower() for i in labels_family]

            label_family = torch.tensor([int(labels_name_family.index(l.lower())) for l in labels_family]).to(args.device)
            label_subfamily = torch.tensor([int(labels_name_subfamily.index(l.lower())) for l in labels_subfamily]).to(args.device)
            label_genus = torch.tensor([int(labels_name_genus.index(l.lower())) for l in labels_genus]).to(args.device)
            label_specific_epithet = torch.tensor([int(labels_name_specific_epithet.index(l.lower())) for l in labels_specific_epithet]).to(args.device)

            inputs = Variable(inputs.to(args.device))

            image_features = image_encoder(inputs)
                
            for i, target in enumerate(label_family):
                class_features[int(target.item())].append(image_features[i])
            for i, target in enumerate(label_subfamily):
                class_features[int(target.item())+6].append(image_features[i])
            for i, target in enumerate(label_genus):
                class_features[int(target.item())+27].append(image_features[i])
            for i, target in enumerate(label_specific_epithet):
                class_features[int(target.item())+162].append(image_features[i])

        class_f = [[] for j in range(len(labels_name))]
        for i in range(len(class_features)):
            if len(class_features[i]) != 0:
                each_class_features = torch.stack(class_features[i])
                each_class_feature = torch.mean(each_class_features, dim=0)
                class_f[i].append(each_class_feature)
            else:
                class_f[i].append(torch.mean(torch.stack(class_features[i-1]), dim=0))
        class_features = torch.stack([i[0] for i in class_f])
    
    elif args.dataset == 'car':
        for data in tqdm.tqdm(train_loader):
            inputs, labels_fine, labels_coarse = data # 196 9
            labels_fine = labels_fine.to(args.device)
            labels_coarse = labels_coarse.to(args.device)
            inputs = Variable(inputs.to(args.device))

            image_features = image_encoder(inputs)
                
            for i, target in enumerate(labels_coarse):
                class_features[int(target.item())].append(image_features[i])
            for i, target in enumerate(labels_fine):
                class_features[int(target.item())+len(labels_name_coarse)].append(image_features[i])

        class_f = [[] for j in range(len(labels_name))]
        for i in range(len(class_features)):
            if len(class_features[i]) != 0:
                each_class_features = torch.stack(class_features[i])
                each_class_feature = torch.mean(each_class_features, dim=0)
                class_f[i].append(each_class_feature)
            else:
                class_f[i].append(torch.mean(torch.stack(class_features[i-1]), dim=0))
        class_features = torch.stack([i[0] for i in class_f])

    elif args.dataset == 'caltech-101' or args.dataset == 'fruits-360':
        for data in tqdm.tqdm(train_loader):
            inputs, labels_l3, labels_l2, labels_l1 = data
            label_l3 = torch.tensor([int(labels_names_l3.index(l)) for l in labels_l3]).to(args.device)
            label_l2 = torch.tensor([int(labels_names_l2.index(l)) for l in labels_l2]).to(args.device)
            label_l1 = torch.tensor([int(labels_names_l1.index(l)) for l in labels_l1]).to(args.device)
            inputs = Variable(inputs.to(args.device))

            image_features = image_encoder(inputs)
                
            for i, target in enumerate(label_l1):
                class_features[int(target)].append(image_features[i])
            for i, target in enumerate(label_l2):
                class_features[int(target)+len(labels_names_l1)].append(image_features[i])
            for i, target in enumerate(label_l3):
                class_features[int(target)+len(labels_names_l1)+len(labels_names_l2)].append(image_features[i])

        class_f = [[] for j in range(len(labels_name))]
        for i in range(len(class_features)):
            if len(class_features[i]) != 0:
                each_class_features = torch.stack(class_features[i])
                each_class_feature = torch.mean(each_class_features, dim=0)
                class_f[i].append(each_class_feature)
            else:
                class_f[i].append(torch.mean(torch.stack(class_features[i-1]), dim=0))
        class_features = torch.stack([i[0] for i in class_f])
    
    elif args.dataset == 'air':
        for data in tqdm.tqdm(train_loader):
            inputs, labels_model, labels_family, labels_maker = data
            labels_model = torch.tensor([int(labels_name_variant.index(l)) for l in labels_model]).to(args.device)
            labels_family = torch.tensor([int(labels_name_family.index(l)) for l in labels_family]).to(args.device)
            labels_maker = torch.tensor([int(labels_name_maker.index(l)) for l in labels_maker]).to(args.device)
            inputs = Variable(inputs.to(args.device))

            image_features = image_encoder(inputs)
                
            for i, target in enumerate(labels_maker):
                class_features[int(target.item())].append(image_features[i])
            for i, target in enumerate(labels_family):
                class_features[int(target.item())+len(labels_name_maker)].append(image_features[i])
            for i, target in enumerate(labels_model):
                class_features[int(target.item())+len(labels_name_maker)+len(labels_name_family)].append(image_features[i])

        class_f = [[] for j in range(len(labels_name))]
        for i in range(len(class_features)):
            if len(class_features[i]) != 0:
                each_class_features = torch.stack(class_features[i])
                each_class_feature = torch.mean(each_class_features, dim=0)
                class_f[i].append(each_class_feature)
            else:
                class_f[i].append(torch.mean(torch.stack(class_features[i-1]), dim=0))
        class_features = torch.stack([i[0] for i in class_f])

    elif args.dataset == 'food-101' or args.dataset == 'pets':
        for data in tqdm.tqdm(train_loader):
            inputs, labels_l2, labels_l1 = data
            label_l2 = torch.tensor([int(labels_names_l2.index(l)) for l in labels_l2]).to(args.device)
            label_l1 = torch.tensor([int(labels_names_l1.index(l)) for l in labels_l1]).to(args.device)
            inputs = Variable(inputs.to(args.device))

            image_features = image_encoder(inputs)
                
            for i, target in enumerate(label_l1):
                class_features[int(target)].append(image_features[i])
            for i, target in enumerate(label_l2):
                class_features[int(target)+len(labels_names_l1)].append(image_features[i])

        class_f = [[] for j in range(len(labels_name))]
        for i in range(len(class_features)):
            if len(class_features[i]) != 0:
                each_class_features = torch.stack(class_features[i])
                each_class_feature = torch.mean(each_class_features, dim=0)
                class_f[i].append(each_class_feature)
            else:
                class_f[i].append(torch.mean(torch.stack(class_features[i-1]), dim=0))
        class_features = torch.stack([i[0] for i in class_f])
    
    elif args.dataset == 'imagenet':
        num_class = [0 for i in range(len(labels_name))]
        for data in tqdm.tqdm(train_loader):
            inputs, labels = data

            labels_l2 = labels.to(args.device)
            labels_l1 = torch.tensor([x[y.index(int(l))] for l in labels]).to(args.device)

            for l in labels_l1:
                num_class[int(l)] += 1
            for l in labels_l2:
                num_class[int(l)] += 1

            inputs = Variable(inputs.to(args.device))

            image_features = image_encoder(inputs)
                
            for i, target in enumerate(labels_l2):
                class_features[int(target)].append(image_features[i])
            for i, target in enumerate(labels_l1):
                class_features[int(target)].append(image_features[i])

        class_f = [[] for j in range(len(labels_name))]
        for i in range(len(class_features)):
            if len(class_features[i]) != 0:
                each_class_features = torch.stack(class_features[i])
                each_class_feature = torch.mean(each_class_features, dim=0)
                class_f[i].append(each_class_feature)
            else:
                class_f[i].append(torch.mean(torch.stack(class_features[i-1]), dim=0))
        class_features = torch.stack([i[0] for i in class_f])

    print(class_features.shape)
    with open(f'hgclip/{args.dataset}_prototypes.pkl', 'wb') as f:
        pickle.dump(class_features.cpu(), f)
    print(f'image features of {len(labels_name)} categories saved at {args.dataset}_prototypes.pkl')


if __name__ == '__main__':
    parser = argparse.ArgumentParser('generate visual prototypes of all categories', parents=[get_args_parser()])
    args = parser.parse_args()
    # args = update_from_config(args)
    main(args)
