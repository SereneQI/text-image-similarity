"""
    Verify that images exist in the root directory
"""
import argparse
import os.path
from PIL import Image
from nltk.tokenize import word_tokenize

def file_len(fname):
    with open(fname) as f:
        for i, l in enumerate(f):
            pass
    return i + 1

def removeImages(r, f, o):
    print("Removing bad images")
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

def removeText(f, o):
    print("Removing bad text")
    i = 0
    imList = []
    nl = file_len(f)
    with open(f) as fin:
        for j, line in enumerate(fin):
            print("%2.2f" % (j/nl*100.0), end='\r')
            line=line.rstrip()
            try:
                _, cap = line.split('\t')
            except Exception:
                print("Got exception on line :", line)
                i += 1
                continue
            
            cap = word_tokenize(cap)
            if 1 <= len(cap) < 20:
                if len(cap) == 1:
                   if cap != '':
                    imList.append(line)
                   else:
                    print("Error on cap :", cap)
                    i += 1
                else:
                    imList.append(line)
            else:
                i += 1
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
    i = 0
    if args.root_dir != "None":
        i += removeImages(args.root_dir, args.image_file, args.output)
    i += removeText(args.image_file, args.output)
    print("Removed ", i, "images") 



