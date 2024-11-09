#!/bin/bash

# Download and extract imagenet-r in sequence
wget -nc -P data https://people.eecs.berkeley.edu/~hendrycks/imagenet-r.tar && tar -xf data/imagenet-r.tar -C data/

# import gdown
# url = 'https://drive.google.com/file/d/1Mj0i5HBthqH1p_yeXzsg22gZduvgoNeA/view?usp=sharing'
# output_path = 'ImageNet-Sketch.zip'
# gdown.download(url, output_path, quiet=False,fuzzy=True)

# Download and extract cifar-10-c, cifar-100-c, and imagenet-c
python robustbench/zenodo_download.py "2535967" "CIFAR-10-C.tar" "data"
python robustbench/zenodo_download.py "3555552" "CIFAR-100-C.tar" "data"

python robustbench/zenodo_download.py "2235448" "blur.tar" "data/ImageNet-C"
python robustbench/zenodo_download.py "2235448" "digital.tar" "data/ImageNet-C"
python robustbench/zenodo_download.py "2235448" "extra.tar" "data/ImageNet-C"

python robustbench/zenodo_download.py "2235448" "weather.tar" "data/ImageNet-C"
rm data/ImageNet-C/content