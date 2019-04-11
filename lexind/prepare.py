import argparse

from data.data import *
from utils.utils import *


###################
# ARGUMENTS
parser = argparse.ArgumentParser(description='Reduce embedding space for faster processing.')

parser.add_argument("--embeddings", type=str, default="", help="")
parser.add_argument("--embeddings_header", type=str2bool, default=True, help="")
parser.add_argument("--load_first_n", type=int, default=None, help="")

parser.add_argument("--lex_train", type=str, default="", help="")
parser.add_argument("--lex_test", type=str, default="", help="")

parser.add_argument("--store", type=str, default="", help="")
parser.add_argument("--storedict", type=str2bool, default=False, help="")


###################
# PARSE ARGUMENTS
config_class = parser.parse_args()
config = config_class.__dict__
config['logger'] = get_logger('lexind', config['store'] + ".log")

dump_dict(config, config['store'] + "," + os.path.basename(__file__) + ",config.json")
config['logger'].info("Prepare data for lexicon induction.")
###################
# PROGRAM
embeds = Embeddings(config['logger'])
embeds.load(config['embeddings'], load_first_n=config['load_first_n'], header=config['embeddings_header'])


Ltrain = Lexicon(config['logger'])
Ltrain.load(config['lex_train'], 'original')

Ltest = Lexicon(config['logger'])
Ltest.load(config['lex_test'], 'original')

# filter embeddings for faster processing
embeds.filter(set([k for k, v in Ltrain.L['original']]) | set([k for k, v in Ltest.L['original']]))

embeds.store(config['store'] + ",embeddings.txt")

if config['storedict']:
    # store binary version of train dict
    Ltrain.L['continuous'] = Ltrain.L['original']
    Ltrain.filter_words(embeds.Wset)
    Ltrain.version = 'continuous'
    Ltrain.binarise()
    Ltrain.store(config['store'] + ",ltrainbinary.txt", 'countable')
