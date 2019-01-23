"""
    Verify that images exist in the root directory
"""
import argparse
import os.path

def removeImages(r, f, o):
    i = 0
    imList = []
    with open(f) as fin:
        for line in f:
            line=f.rstrip()
            im, _ = line.split('\t')
            im = im+'.jpg'
            path = os.path.join(r,im)
            if os.path.exists(path):
                i += 1
                imList.append(line)
    with open(o, 'w') as fout:
        for line in imList:
            fout.write(line+'\n')






if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Verify that images exist in the root directory')

    parser.add_argument("-r", '--root_dir', required=True, help='Main directory with images')
    parser.add_argument("-f", "--image_file", required=True, help='File contraining images list')
    parser.add_argument('-o', '--output', required=True, help="Ouput file to store new image list (can be the same as -f")

    args = parser.parse_args()
    
    i = removeImages(args.root_dir, args.image_file, args.output)
    print("Removed ", i, "images") 



