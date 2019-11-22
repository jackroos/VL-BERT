captions = []
urls = []

with open('Validation_GCC-1.1.0-Validation.tsv') as fp:
    for cnt, line in enumerate(fp):
        s = line.split('\t')
        captions.append(s[0].split(' '))
        urls.append(s[1][:-1])
        
valids = set([])
with open('val_valid.txt') as fp:
    for cnt, line in enumerate(fp):
        valids.add(line[:-1])
        
import json
with open('val.json', 'w') as outfile:
    for cnt, (cap, url) in enumerate(zip(captions, urls)):
        im = "{:08d}.jpg".format(cnt)
        if (im in valids):
            d = {'image':"val_image.zip@/{}".format(im), 'caption':cap}
            json.dump(d, outfile)
            outfile.write('\n')
            
import json
with open('val_frcnn.json', 'w') as outfile:
    for cnt, (cap, url) in enumerate(zip(captions, urls)):
        im = "{:08d}.jpg".format(cnt)
        if (im in valids):
            d = {'image':"val_image.zip@/{}".format(im), 'caption':cap, 'frcnn':"val_frcnn.zip@/{:08d}.json".format(cnt)}
            json.dump(d, outfile)
            outfile.write('\n')