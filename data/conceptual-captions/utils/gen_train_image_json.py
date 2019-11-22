captions = []
urls = []

with open('Train_GCC-training.tsv') as fp:
    for cnt, line in enumerate(fp):
        s = line.split('\t')
        captions.append(s[0].split(' '))
        urls.append(s[1][:-1])
        
valids = set([])
with open('train_valid.txt') as fp:
    for cnt, line in enumerate(fp):
        valids.add(line[:-1])
        
import json
with open('train.json', 'w') as outfile:
    for cnt, (cap, url) in enumerate(zip(captions, urls)):
        im = "{:08d}.jpg".format(cnt)
        if (im in valids):
            d = {'image':"train_image.zip@/{}".format(im), 'caption':cap}
            json.dump(d, outfile)
            outfile.write('\n')
            
            
import json
with open('train_frcnn.json', 'w') as outfile:
    for cnt, (cap, url) in enumerate(zip(captions, urls)):
        im = "{:08d}.jpg".format(cnt)
        if (im in valids):
            d = {'image':"train_image.zip@/{}".format(im), 'caption':cap, 'frcnn':"train_frcnn.zip@/{:08d}.json".format(cnt)}
            json.dump(d, outfile)
            outfile.write('\n')