import argparse

from data.data import *
from utils.utils import *
from model.model import *


###################
# ARGUMENTS
parser = argparse.ArgumentParser(description='Run Lexicon Induction.')


parser.add_argument("--embeddings", type=str, default="", help="")
parser.add_argument("--embeddings_header", type=str2bool, default=True, help="")
parser.add_argument("--normalize_embeddings", type=str2bool, default=True, help="")

parser.add_argument("--lex_train", type=str, default="", help="")
parser.add_argument("--lex_train_version", type=str, default="", help="")
parser.add_argument("--lex_train_inconsistencies", type=str, default="delete", help="")

parser.add_argument("--lex_test", type=str, default="", help="")
parser.add_argument("--lex_test_version", type=str, default="", help="")

parser.add_argument("--allow_train_test_overlap", type=str2bool, default=False, help="")

parser.add_argument("--load_first_n", type=int, default=None, help="")
parser.add_argument("--trafo_first_n", type=int, default=80000, help="")

parser.add_argument("--store", type=str, default="", help="")

parser.add_argument("--method", type=str, default="", help="")

parser.add_argument("--densray__normalize_D", type=str2bool, default=True, help="")
parser.add_argument("--densray__weights", type=str, default=None, help="")
parser.add_argument("--densray__normalize_labels", type=str2bool, default=False, help="")

parser.add_argument("--pred_method", type=str, default="first_dimension", help="")
parser.add_argument("--dim_weights", type=str, default=None, help="")


parser.add_argument("--explore", type=str2bool, default=False, help="")


###################
# PARSE ARGUMENTS
config_class = parser.parse_args()
config = config_class.__dict__

config['logger'] = get_logger('lexind', config['store'] + ".log")
if config['dim_weights'] is not None:
    config['dim_weights'] = [float(x) for x in config['dim_weights'].split(",")]
if config['densray__weights'] is not None:
    config['densray__weights'] = [float(x) for x in config['densray__weights'].split(",")]

dump_dict(config, config['store'] + "," + config['method'] + "," + os.path.basename(__file__) + ",config.json")
config['logger'].info("Runing lexicon induction.")

###################
# DATA
config['logger'].info("Loading data.")

embeds = Embeddings(config['logger'])
embeds.load(config['embeddings'], load_first_n=config['load_first_n'], header=config['embeddings_header'])


Ltrain = Lexicon(config['logger'])
Ltrain.load(config['lex_train'], config['lex_train_version'])

if config['trafo_first_n'] is not None:
    Ltrain.filter_words(set(embeds.W[:config['trafo_first_n']]))

if config['lex_train_inconsistencies'] == 'ignore':
    pass
elif config['lex_train_inconsistencies'] == 'delete':
    Ltrain.remove_inconsistencies(remove_all=True)
elif config['lex_train_inconsistencies'] == 'keep_first':
    Ltrain.remove_inconsistencies(remove_all=False)

Ltest = Lexicon(config['logger'])
Ltest.load(config['lex_test'], config['lex_test_version'])

if not config['allow_train_test_overlap']:
    # remove potential train/test overlap
    Ltrain.filter_words(set([k for k, v in Ltrain.L[Ltrain.version]]) - set([k for k, v in Ltest.L[Ltest.version]]))

Ltrain.binarise()

# filter embeddings for faster processing
embeds.filter(set([k for k, v in Ltrain.L[Ltrain.version]]) | set([k for k, v in Ltest.L[Ltest.version]]))


if config['normalize_embeddings']:
    embeds.normalize()

###################
# MODEL
config['logger'].info("Running model.")

modeltype, model = config['method'].split(",")

if modeltype == 'densray':
    dr = DensRay(config['logger'], embeds, Ltrain)
    dr.fit(weights=config['densray__weights'], model=model, normalize_D=config['densray__normalize_D'], normalize_labels=config['densray__normalize_labels'])
    dr.store(config['store'] + "," + config['method'] + ".trafo")
    T = dr.T
elif modeltype == 'regression':
    reg = Regression(config['logger'], embeds, Ltrain)
    reg.prepare_data(model)
    reg.fit(model)
    reg.store(config['store'] + "," + config['method'] + ".trafo")
    T = reg.T
elif modeltype == 'matlab':
    dens = Densifier(config['logger'], embeds, Ltrain, Ltest)
    dens.prepare_data(config['store'])
    dens.fit_predict()


if modeltype != 'matlab':
    pred = LexIndPredictor(config['logger'], embeds, [k for k, v in Ltest.L[Ltest.version]], T)
    pred.predict(config['pred_method'], dim_weights=config['dim_weights'])
    pred.store(config['store'] + "," + config['method'] + ".predictions")
