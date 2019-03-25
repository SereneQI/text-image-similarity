# Multilingual visual semantic similarity

This work is an extension of the paper [Finding beans in burgers: Deep semantic-visual embedding with localization](http://openaccess.thecvf.com/content_cvpr_2018/CameraReady/3272.pdf) with multilingual support.

With Multi30K dataset, can learn English, French, German and Czech.

## Main dependencies

This code is written in python. All dependencies are in the Dockerfile. It will automatically install:

* Python 3.7
* Pytorch 1.0
* FastText
* SRU[cuda]
* Numpy
* Scipy
* Torchvision
* Ms Coco API (pycocotools)
* NLTK
* opencv

An environment file for conda is available in the repository (environment.yml).

See notebooks for how to use it.





