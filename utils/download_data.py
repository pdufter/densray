import argparse
import urllib.request
import os
import zipfile
import random

random.seed(123)

parser = argparse.ArgumentParser(description='Prepare example data.')
parser.add_argument("--data_dir", type=str, default="", help="")

config_class = parser.parse_args()
config = config_class.__dict__


def download(filename, url, expected_bytes):
    filename, _ = urllib.request.urlretrieve(url, filename)
    statinfo = os.stat(filename)
    if statinfo.st_size == expected_bytes:
        print('Downloaded and verified {}.'.format(url))
    else:
        print(statinfo.st_size)
        raise Exception('Failed to verify {}.'.format(filename))


##############################################################################
# Download German Polarity Clues Lexicon
print("Downloading sentiment lexicon...")
download(config['data_dir'] + "germanpc.zip", "http://www.ulliwaltinger.de/sentiment/GermanPolarityClues-2012.zip", 295706)

zip_ref = zipfile.ZipFile(config['data_dir'] + "germanpc.zip", 'r')
zip_ref.extractall(config['data_dir'])
zip_ref.close()

lexicon = []
infile = open(config['data_dir'] + "GermanPolarityClues-2012/GermanPolarityClues-Negative-Lemma-21042012.tsv", "r")
for line in infile:
    lexicon.append((line.split()[0], -1.0))

infile.close()
infile = open(config['data_dir'] + "GermanPolarityClues-2012/GermanPolarityClues-Positive-Lemma-21042012.tsv", "r")
for line in infile:
    lexicon.append((line.split()[0], 1.0))

infile.close()

random.shuffle(lexicon)

lexicon_train = []
lexicon_test = []
for elem in lexicon:
    if random.random() < 0.9:
        lexicon_train.append(elem)
    else:
        lexicon_test.append(elem)

outfile = open(config['data_dir'] + "lexicon_train.txt", "w")
for elem in lexicon_train:
    outfile.write("{} {}\n".format(elem[0], elem[1]))

outfile.close()

outfile = open(config['data_dir'] + "lexicon_test.txt", "w")
for elem in lexicon_test:
    outfile.write("{} {}\n".format(elem[0], elem[1]))

outfile.close()

##############################################################################
# Download Google Analogy Dataset
print("Downloading analogies...")
download(config['data_dir'] + "questions-words.txt", "http://download.tensorflow.org/data/questions-words.txt", 603955)

##############################################################################
# Download German and English Fasttext embeddings
print("Downloading embeddings...")

download(config['data_dir'] + "cc.de.300.vec.gz", "https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.de.300.vec.gz", 1278030050)
os.system("gunzip {}".format(config['data_dir'] + "cc.de.300.vec.gz"))


download(config['data_dir'] + "cc.en.300.vec.gz", "https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.en.300.vec.gz", 1325960915)
os.system("gunzip {}".format(config['data_dir'] + "cc.en.300.vec.gz"))
