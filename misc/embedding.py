"""
    Replace word with embedding
"""

from nltk.tokenize import word_tokenize
import argparse
import fastText
from utils import encode_sentence_fasttext

def file_len(fname):
    with open(fname) as f:
        for i, l in enumerate(f):
            pass
    return i + 1

def tokenizeFile(filename, file_out):
    fout = open(file_out, "w")
    nl = file_len(filename)
    with open(filename) as f:
        for k, line in enumerate(f):
            print("%2.2f" % (k/nl*100.0), "\%", end='\r')
            i, t = line.split("\t")    
            t = word_tokenize(t)
            fout.write(i + '\t')
            for j, w in enumerate(t):
                if j < len(t)-1:
                    fout.write(w+' ')
                else:
                    fout.write(w)
            fout.write('\n')
            
                
        

def embedFile(args):
    print("Compute embeddings for the file:", args.file)
    embed = fastText.load_model(args.dict)
    nl = file_len(args.file)
    fout = open(args.output, 'w')
    with open(args.file) as f:
        for i, line in enumerate(f):
            line = line.rstrip()
            print("%2.2f" % (i/nl*100.0), "\%", end='\r')
            i, t = line.split("\t")
            es = encode_sentence_fasttext(t.split(' '), embed, tokenize=False)
            fout.write(i+'\t'+str(es)+'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')

    parser.add_argument("-f", "--file", help="File to encode", required=True)
    parser.add_argument("-o", "--output", help='Output file', required=True)
    parser.add_argument("-dict", dest='dict', help='Dictionnary link', default="./data/wiki.fr.bin")
    parser.add_argument("-t", "--tokenize_only", default="True")
       
    args = parser.parse_args()
    if args.tokenize_only == 'False':
        embedFile(args)
    else:
        tokenizeFile(args.file, args.output)
    
