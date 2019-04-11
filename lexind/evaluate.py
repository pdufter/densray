import argparse

import numpy as np

from data.data import *
from utils.utils import *
from evaluation.evaluation import LexIndEval

parser = argparse.ArgumentParser(description='DiffPCA.')

parser.add_argument("--lex_true", type=str, default="", help="")
parser.add_argument("--lex_pred", type=str, default="", help="")
parser.add_argument("--lex_true_version", type=str, default="", help="")
parser.add_argument("--lex_pred_version", type=str, default="", help="")
parser.add_argument("--impute", type=str, default="", help="")

parser.add_argument("--store", type=str, default="", help="")
parser.add_argument("--method", type=str, default="", help="")

###################
# PARSE ARGUMENTS
config_class = parser.parse_args()
config = config_class.__dict__
config['logger'] = get_logger('lexind', config['store'] + ".log")

dump_dict(config, config['store'] + "," + config['method'] + "," + os.path.basename(__file__) + ",config.json")
config['logger'].info("Evaluating Lexicon.")


# read dictionaries
Ltrue = Lexicon(config['logger'])
Ltrue.load(config['lex_true'], config['lex_true_version'])
n_orig = len(Ltrue.L[Ltrue.version])
Ltrue.remove_inconsistencies(remove_all=True)
# binarize and remove neutral words
Ltrue.binarise()

Lpred = Lexicon(config['logger'])
Lpred.load(config['lex_pred'], config['lex_pred_version'])
Lpred.binarise()

if config['impute'] == '':
    both = set([k for k, v in Lpred.L[Lpred.version]]) & set([k for k, v in Ltrue.L[Ltrue.version]])
    Ltrue.filter_words(both)
    Lpred.filter_words(both)
elif config['impute'] == 'median':
    # imputes the median score for missing values
    median = np.median([v for k, v in Lpred.L[Lpred.version].values()])
    for w in set([k for k, v in Ltrue.L[Ltrue.version]]) - set([k for k, v in Lpred.L[Lpred.version]]):
        Lpred.L[Lpred.version][w] = median

Ltrue.compute_ranks()
Lpred.compute_ranks()
n_final = len(Ltrue.L[Ltrue.version])
config['logger'].info("Evaluating on {} / {} words.".format(n_final, n_orig))

# determine sign
ev = LexIndEval(Ltrue.L[Ltrue.version], Lpred.L[Lpred.version])
ev.prepare()
tau, _ = ev.compute_kendalls()
if tau < 0:
    Lpred.invert()
    Lpred.binarise()
    Lpred.compute_ranks()

# run evaluation for all combinations of binarized and ranked dictionary
# note that for binary most values are considered "ties" in kendalls tau
# the true value is:
# Ltrue.L[Ltrue.version], Lpred.L[Lpred.version]
ev = LexIndEval(Ltrue.L['ranked'], Lpred.L['ranked'])
ev.prepare()
rr_tau, _ = ev.compute_kendalls()

ev = LexIndEval(Ltrue.L['countable'], Lpred.L['ranked'])
ev.prepare()
br_tau, _ = ev.compute_kendalls()

ev = LexIndEval(Ltrue.L['ranked'], Lpred.L['countable'])
ev.prepare()
rb_tau, _ = ev.compute_kendalls()

ev = LexIndEval(Ltrue.L['countable'], Lpred.L['countable'])
ev.prepare()
bb_tau, _ = ev.compute_kendalls()

outfile = store(config['store'] + "," + config['method'] + ".evaluation")
outfile.write("\t".join([config['method'], str(rr_tau), str(br_tau), str(rb_tau), str(bb_tau)]) + "\n")
outfile.close()
