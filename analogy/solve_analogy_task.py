import argparse

import numpy as np

from collections import defaultdict
from tqdm import tqdm

from utils.utils import *
from model.model import *
from data.data import *


###################
# ARGUMENTS
parser = argparse.ArgumentParser(description='Analogy Task.')

parser.add_argument("--embeddings", type=str, default="/mounts/data/proj/sascha/corpora/Embeddings/GoogleNews-vectors-negative300.txt", help="")
parser.add_argument("--embeddings_header", type=str2bool, default=True, help="")
parser.add_argument("--normalize_embeddings", type=str2bool, default=True, help="")
parser.add_argument("--load_first_n", type=int, default=20000, help="")
parser.add_argument("--analogies", type=str, default="/mounts/work/philipp/data/analogy/google/question-words.txt", help="")

parser.add_argument("--bats", type=str2bool, default=True, help="")

parser.add_argument("--method", type=str, default="regression,svm", help="")
parser.add_argument("--pred_method", type=str, default="clcomp", help="")

parser.add_argument("--store", type=str, default="", help="")

parser.add_argument("--use_proba", type=str2bool, default=False, help="")

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
if config['normalize_embeddings']:
    embeds.normalize()

if config['bats']:
    analogy = Bats(config['logger'])
    analogy.load(config['analogies'], first_only=False)
    analogy.adjust_capitalisation(embeds.W)
else:
    analogy = GoogleAnalogy(config['logger'])
    analogy.load(config['analogies'])
    analogy.get_classes()


###################
# DISTANCE ANALYSIS
config['logger'].info("Computing distance analysis.")

# compute average similarity between and within different analogy pairs
dists = defaultdict(lambda: defaultdict(list))

for chapter in analogy.tuples:
    ap = AnalogyPredictor(config['logger'], embeds, analogy.tuples[chapter])
    ranks, intersim, intrasim0, intrasim1 = ap.get_distances()

    dists[chapter]['intersim'].extend(intersim.tolist())
    dists[chapter]['intrasim0'].extend(intrasim0.tolist())
    dists[chapter]['intrasim1'].extend(intrasim1.tolist())
    if config['bats']:
        # the first character of a chapter name indicates the category
        # e.g. Inflectional, Derivational, ...
        dists[chapter[0]]['intersim'].extend(intersim.tolist())
        dists[chapter[0]]['intrasim0'].extend(intrasim0.tolist())
        dists[chapter[0]]['intrasim1'].extend(intrasim1.tolist())

    dists['OVERALL']['intersim'].extend(intersim.tolist())
    dists['OVERALL']['intrasim0'].extend(intrasim0.tolist())
    dists['OVERALL']['intrasim1'].extend(intrasim1.tolist())

# write the results
result = ""
for k in sorted(dists):
    result += k + ": {:.2f} {:.2f} {:.2f}".format(np.array(dists[k]['intersim']).mean(), np.array(dists[k]['intrasim0']).mean(), np.array(dists[k]['intrasim1']).mean()) + "\n"

outfile = open(config['store'] + ",distances.txt", 'w')
outfile.write(result)
outfile.close()




###################
# ANALOGY TASK
config['logger'].info("Solving Analogies.")
predictions = defaultdict(list)
trues = defaultdict(list)
corrects = defaultdict(list)
# for each class solve the problem and get predictions
modeltype, model = config['method'].split(",")
for chapter in analogy.tuples:
    # if chapter != "E01 [country - capital].txt":
    #    continue
    #    import ipdb; ipdb.set_trace();
    config['logger'].info(chapter)
    n_test = 2
    # filter tuples
    ap = AnalogyPredictor(config['logger'], embeds, analogy.tuples[chapter])
    rel_tuples = ap.ana

    # do cross validation
    for i in tqdm(range(0, len(rel_tuples), n_test), desc="crossvalidation", leave=False):
        tuples_train = rel_tuples[:i] + rel_tuples[i + n_test:]
        tuples_test = rel_tuples[i:i + n_test]

        ap = AnalogyPredictor(config['logger'], embeds, tuples_train)
        ap.fit_classifier(modeltype, model)
        answer = ap.predict([x[0] for x in tuples_test], method=config['pred_method'])

        assert all([x[0] == y[0] for (x, y) in zip(tuples_test, answer)]), "Inconsistencies in queries."

        predictions[chapter].extend(answer)
        trues[chapter].extend(tuples_test)
        for (x, y) in zip(answer, tuples_test):
            # ignore capitalisation for evaluation
            tmp = x[1].lower() in [z.lower() for z in y[1]]
            corrects[chapter].append(tmp)
            if config['bats']:
                corrects[chapter[0]].append(tmp)
            corrects['OVERALL'].append(tmp)

# ###################
# EVALUATION
config['logger'].info("Writing results to {}.".format(config['store']))
# write out results
result = ""
for chapter in sorted(corrects):
    result += chapter + ": {:.2f} ({})".format(sum(corrects[chapter]) / len(corrects[chapter]), len(corrects[chapter])) + "\n"

outfile = open(config['store'] + ".evaluation", 'w')
outfile.write(result)
print(result)
outfile.close()

# write out predictions and correct stuff
outfile = open(config['store'] + ",wordlevel.evaluation", 'w')
for chapter in sorted(set(analogy.tuples) & set(corrects)):
    outfile.write("\n" * 5 + chapter + "\n")
    outfile.write("true-true\tpredicted-predicted\tlabel\n")
    for true, predicted, label in zip(trues[chapter], predictions[chapter], corrects[chapter]):
        outfile.write(true[0] + "-" + true[1][0] + "\t" + predicted[0] + "-" + predicted[1] + "\t" + str(label) + "\n")
outfile.close()
    