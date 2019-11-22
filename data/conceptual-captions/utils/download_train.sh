# use 20 threads

cat train4download.txt | xargs -n 2 -P 20 wget -nc -U 'Mozilla/5.0 (X11; Linux i686) AppleWebKit/537.17 (KHTML, like Gecko) Chrome/24.0.1312.27 Safari/537.17' --timeout=1 --waitretry=0 --tries=5 --retry-connrefused -nv -O
find ../train_image -type f -size -1c -exec rm {} \;
ls -d ../train_image/* | xargs -n 1 -P 20 python check_valid.py | tee train_size_invalid.txt
xargs rm < train_size_invalid.txt
rm train_size_invalid.txt
ls ../train_image > train_valid.txt
