import json


def exportImagePath(filepath, listDownload, outpath):
    imageIds = []
    idsDescription = {}
    idsLinks = {}
    idsTitles = {}
    
    with open(filepath) as f:
        with open(listDownload, 'w') as fout:
            with open(outpath, 'w') as fullout:
                paths = []
                for i, line in enumerate(f):
                    if i%2 == 0:
                        print('%2.2f'% (i/3672625.0*100.0), '\%', end='\r')
                    
                    j = json.loads(line)
                    source = j['_source']
                    
                    #imageIds.append(j['_id'])
                    #idsLinks[j['_id']] = source['imageLink']
                    #idsDescription[j['_id']] = source['description']
                    #idsTitles[j['_id']] = source['title']
                    
                    fout.write(j['_id'] + '\t' + source['imageLink'] + '\n')
                    if not source['description'] is None:
                        fullout.write(j['_id'] +'\t'+source['description']+'\n')
                    if not source['title'] is None:
                        fullout.write(j['_id'] +'\t'+source['title']+'\n')
         
     

if __name__ == '__main__':
    exportImagePath('/data/fr_shopping.json', 'imageList.txt', 'shoppingDataset.txt')
