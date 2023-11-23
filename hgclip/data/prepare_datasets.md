We suggest putting all datasets under the same folder (say `data`) to ease management and following the instructions below to organize datasets to avoid modifying the source code. The file structure looks like

```
data/
|–– cifar-100-python
|–– caltech-101
|–– ETHEC_dataset
|–– fgvc-aircraft-2013b
|–– food-101
|–– fruits-360
|–– imagenet
|–– imagenet-adversarial
|–– imagenet-rendition
|–– imagenet-sketch
|–– imagenetv2
|–– oxford-iiit-pet/
|–– stanford_cars/
```

Besides ImageNet and CIFAR100 (which can be autoloaded through the `torchvision` API), each dataset can be downloaded through the standard websites for each: 

- [Caltech101](http://www.vision.caltech.edu/Image_Datasets/Caltech101/101_ObjectCategories.tar.gz)

- OxfordPets 
  
  - [images](https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz) 
  
  - [annotations](https://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz)

- [Food-101](https://www.kaggle.com/datasets/dansbecker/food-101)

- [Fruits360](https://www.kaggle.com/datasets/moltean/fruits)

- StanfordCars 
  
  - [train images](http://ai.stanford.edu/~jkrause/car196/cars_train.tgz)
  
  - [test images](http://ai.stanford.edu/~jkrause/car196/cars_test.tgz)
  
  - [train labels](https://ai.stanford.edu/~jkrause/cars/car_devkit.tgz)
  
  - [test labels](http://ai.stanford.edu/~jkrause/car196/cars_test_annos_withlabels.mat) 

- [FGVCAircraft](https://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/archives/fgvc-aircraft-2013b.tar.gz)

- [ETHEC](https://www.research-collection.ethz.ch/handle/20.500.11850/365379?show=full)

- [ImageNetV2](https://github.com/modestyachts/ImageNetV2)

- [ImageNet-Sketch](https://github.com/HaohanWang/ImageNet-Sketch)

- [ImageNet-A](https://github.com/hendrycks/natural-adv-examples)

- [ImageNet-R](https://github.com/hendrycks/imagenet-r)

- BREEDS
  
  - The dataset is based on ImageNet, so there is no need to download extra datasets.
  
  - Please refer to [GitHub Repo](https://github.com/MadryLab/BREEDS-Benchmarks) and [Documentation](https://robustness.readthedocs.io/en/latest/example_usage/breeds_datasets.html).
