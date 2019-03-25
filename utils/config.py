
path = {
    # Path to the Ms-CoCo dataset folder (containing annotations and images subfolder)
    # http://cocodataset.org/#home
    "COCO_ROOT": "/data/datasets/coco/",

    # Data set split from "Deep Visual-Semantic Alignments for Generating Image Descriptions"  Karpathy et al.
    # Coco split can be found here https://cs.stanford.edu/people/karpathy/deepimagesent/coco.zip
    "COCO_RESTVAL_SPLIT": "/data/datasets/coco/dataset.json",

    # Path to the weights of classification model (resnet + weldon pooling) pretrained on imagenet
    # https://cloud.lip6.fr/index.php/s/sEiwuVj7UXWwSjf
    "WELDON_CLASSIF_PRETRAINED": "./data/pretrained_classif_152_2400.pth.tar",
    
    "WORD_DICT":"/data/m.portaz/wiki.multi.en.vec",

    # ## The path below are only required for pointing game evaluation ## #

    # Path to the folder containing the images of the visual genome dataset
    # https://visualgenome.org/
    "VG_IMAGE": "/path/to/dataset/visual_genome/VG_100K/",

    # Path to the folder containing the annotation for the the visual genome dataset (image data and regions description)
    # https://visualgenome.org/
    "VG_ANN": "/path/to/dataset/visual_genome/data"
}
