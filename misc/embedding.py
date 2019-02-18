"""
    Replace word with embedding
"""

from nltk.tokenize import word_tokenize
import argparse
import fastText
from misc.utils import encode_sentence_fasttext, load_vec
import io
from scipy.spatial import distance

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


def CompareDict(path1, path2):
    print("Comparing model :", path1, " and ", path2)
    embed1, id2word1, word2id1 = load_vec(path1)
    embed2, id2word2, word2id2 = load_vec(path2)
    
    fm_embed, fm_id2word, fm_word2id = [], [], {}
    sim = []
    
    avg_cos_similarity = 0
    avg_euc_similarity = 0
    j = 0
    avg_cos_dissimilarity = 0
    avg_euc_dissimilarity = 0
    i = 0
    
    
    for word in word2id1:
        if word in word2id2: # same word present in both dictionaries
            sim.append(word)
            id1 = word2id1[word]
            id2 = word2id1[word]
            avg_cos_similarity += distance.cosine(embed1[id1], embed2[id2])
            avg_euc_similarity += distance.euclidean(embed1[id1], embed2[id2])
            j += 1
        else:
            #fm_embed.append(word)
            #fm_id2word.append(word)
            #fm_word2id[word] = i
            avg_cos_dissimilarity += distance.cosine(embed1[id1], embed2[id2])
            avg_euc_dissimilarity += distance.euclidean(embed1[id1], embed2[id2])
            i += 1
    #return fm_embed, fm_id2word, fm_word2id, sim
    return avg_cos_similarity/j, avg_euc_similarity/j, avg_cos_dissimilarity/i, avg_euc_dissimilarity/i

def main():
    CompareDict("data/wiki.multi.en.vec", "data/wiki.multi.fr.vec")
    CompareDict("data/wiki.multi.en.vec", "data/wiki.multi.de.vec")

if __name__ == '__main__':
    main()
