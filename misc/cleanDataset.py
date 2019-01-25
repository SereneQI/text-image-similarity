"""
    Verify that images exist in the root directory
"""
import argparse
import os.path
from PIL import Image

def removeImages(r, f, o):
    i = 0
    imList = []
    with open(f) as fin:
        for line in fin:
            line=line.rstrip()
            try:
                im, _ = line.split('\t')
            except Exception:
                print("Got exception on line :", line)
                i += 1
                continue
            im = im+'.jpg'
            path = os.path.join(r,im)
            if os.path.exists(path):
                try:
                    Image.open(path)
                except Exception:
                    print("Cannot open file :", path)
                    i += 1
                else:
                    imList.append(line)
            else:
                i+=1
    with open(o, 'w') as fout:
        for line in imList:
            fout.write(line+'\n')
    return i





if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Verify that images exist in the root directory')

    parser.add_argument("-r", '--root_dir', required=True, help='Main directory with images')
    parser.add_argument("-f", "--image_file", required=True, help='File contraining images list')
    parser.add_argument('-o', '--output', required=True, help="Ouput file to store new image list (can be the same as -f")

    args = parser.parse_args()
    
    i = removeImages(args.root_dir, args.image_file, args.output)
    print("Removed ", i, "images") 



