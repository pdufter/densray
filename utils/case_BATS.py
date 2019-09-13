import argparse

from utils.utils import *
from model.model import *
from data.data import *


###################
# ARGUMENTS
parser = argparse.ArgumentParser(description='Analogy Task.')

parser.add_argument("--embeddings", type=str, default="/mounts/data/proj/sascha/corpora/Embeddings/GoogleNews-vectors-negative300.txt", help="")
parser.add_argument("--embeddings_header", type=str2bool, default=True, help="")
parser.add_argument("--load_first_n", type=int, default=100000, help="")
parser.add_argument("--analogies", type=str, default="/mounts/work/philipp/data/analogy/BATS_3.0", help="")

parser.add_argument("--store", type=str, default="", help="")


###################
# PARSE ARGUMENTS
config_class = parser.parse_args()
config = config_class.__dict__

config['logger'] = get_logger('lexind', config['store'] + ".log")

dump_dict(config, config['store'] + "," + os.path.basename(__file__) + ",config.json")
config['logger'].info("Solving Analogy task.")

###################
# DATA
embeds = Embeddings(config['logger'])
embeds.load(config['embeddings'], load_first_n=config['load_first_n'], header=config['embeddings_header'])

analogy = Bats(config['logger'])
analogy.load(config['analogies'], first_only=False)
analogy.adjust_capitalisation(embeds.W)
analogy.store(config['store'])
    