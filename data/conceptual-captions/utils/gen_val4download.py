import os

captions = []
urls = []

with open('Validation_GCC-1.1.0-Validation.tsv') as fp:
    for cnt, line in enumerate(fp):
        s = line.split('\t')
        captions.append(s[0].split(' '))
        urls.append(s[1][:-1])
        
with open('val4download.txt', 'w') as fp:
    for cnt, url in enumerate(urls):
        fp.write("../val_image/{:08d}.jpg\t\"{}\"\n".format(cnt, url))

if not os.path.exists('../val_image'):
    os.makedirs('../val_image')