import argparse
import numpy as np
import json
from utils.utils import *
from data.data import *
from scipy.stats import ttest_ind
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

###################
# ARGUMENTS
parser = argparse.ArgumentParser(description='Compute Debiasing with respect to certaing words.')

parser.add_argument("--result_dir", type=str, default="", help="")
parser.add_argument("--embeddings", type=str, default="", help="")
parser.add_argument("--load_first_n", type=int, default=None, help="")
parser.add_argument("--embeddings_header", type=str2bool, default=True, help="")
parser.add_argument("--normalize", type=str2bool, default=True, help="")
parser.add_argument("--trafo", type=str, default="", help="")
parser.add_argument("--professions", type=str, default="", help="")

# python -m utils.debiasing.py --embeddings /mounts/data/proj/sascha/corpora/Embeddings/GoogleNews-vectors-negative300.txt --load_first_n 80000 --trafo /mounts/work/philipp/densifier/interpretability/data_tmp/family.txt,dyadicpca.trafo.npy --professions /mounts/work/philipp/densifier/debiasing/professions.json
###################
# PARSE ARGUMENTS
config_class = parser.parse_args()
config = config_class.__dict__

config['logger'] = get_logger('debiasing', config['result_dir'] + ".log")

dump_dict(config, config['result_dir'] + ",debiasing," + os.path.basename(__file__) + ",config.json")


# read in wordspace
embeds = Embeddings(config['logger'])
embeds.load(config['embeddings'], load_first_n=config['load_first_n'], header=config['embeddings_header'])
if config['normalize']:
    embeds.normalize()
embeds.Wset = set(embeds.W)

# read in trafos
# They need to be computed before using the lexind interface.
T = np.load(open(config['trafo'], "rb"))
assert np.allclose(T.transpose().dot(T), np.eye(T.shape[0])), "T not orthogonal."

X_trafo = embeds.X.dot(T)
X_dense = X_trafo[:, :1]
X_compl = X_trafo[:, 1:]

# read professiongs
infile = open(config['professions'], 'r')
professions = []
for line in infile:
    line = line.strip()
    if line:
        professions.append(line)
infile.close()

querys = ["man", "woman"]

index_p = [embeds.W.index(x) for x in professions]
index_q = [embeds.W.index(x) for x in querys]

X_p = embeds.X[index_p, :]
X_q = embeds.X[index_q, :]

X_compl_p = X_compl[index_p, :]
X_compl_q = X_compl[index_q, :]

X_dense_p = X_dense[index_p, :]
X_dense_q = X_dense[index_q, :]

sim = cosine_similarity(X_q, X_p)
sim_compl = cosine_similarity(X_compl_q, X_compl_p)

mw = cosine_similarity(X_q)
mw_compl = cosine_similarity(X_compl_q)



def plot_bias(simthing, outfile, annotate, professions):
    fig, ax = plt.subplots()
    ax.scatter(simthing[0], simthing[1], c='b', marker='x')
    #ax.plot(np.arange(-1, 1, 0.01), np.arange(-1, 1, 0.01), color='grey')
    ax.plot([-1, 1], [-1, 1], ls="--", c=".3")
    bbox_props = dict(boxstyle="round", fc="w", ec="0.5", alpha=0.9)

    for name in annotate:
        idx = professions.index(name)
        name = name + "\n({:.2f}, {:.2f})".format(simthing[0][idx], simthing[1][idx])
        ax.annotate(name, (simthing[0][idx], simthing[1][idx]), xytext=(3* (simthing[0][idx] - 0.25) + 0.25, 3* (simthing[1][idx] - 0.25) + 0.25),
            arrowprops=dict(facecolor='black', shrink=0.01, width=1.0, headwidth=0.5), bbox=bbox_props)
    plt.xlabel("Similarity to 'man'")
    plt.ylabel("Similarity to 'woman'")
    plt.xlim((-0.25, 0.8))
    plt.ylim((-0.25, 0.8))
    plt.savefig(outfile)
    plt.clf()

# add some examples
to_plot = ["nurse", "housekeeper", "professor", "captain", "skipper"]
plot_bias(sim, config['result_dir'] + "prof_orig.png", to_plot, professions)
plot_bias(sim_compl, config['result_dir'] + "prof_debi.png", to_plot, professions)

rel_sim = sim[0] - sim[1]
rel_sim_compl = sim_compl[0] - sim_compl[1]

#plt.boxplot(rel_sim)
#plt.savefig("tmp_bp.png")
#plt.boxplot(rel_sim_compl)
#plt.savefig("tmp_bp_compl.png")

# get most extreme
rel_sim_sort = np.argsort(rel_sim)
rel_sim_compl_sort = np.argsort(rel_sim_compl)

professions = np.array(professions)

n_top = 5
tops = [i for i in rel_sim_sort[:n_top]]
bottoms = [i for i in rel_sim_sort[-n_top:]]

tops_compl = [i for i in rel_sim_compl_sort[:n_top]]
bottoms_compl = [i for i in rel_sim_compl_sort[-n_top:]]

print("Original:")
print(list(zip(professions[tops], rel_sim[tops])))
print(list(zip(professions[bottoms], rel_sim[bottoms])))


print("Debiased:")
print(list(zip(professions[tops_compl], rel_sim_compl[tops_compl])))
print(list(zip(professions[bottoms_compl], rel_sim_compl[bottoms_compl])))

# generate latex table
male = ""
female = ""
for i in range(n_top):
    female += " {} & {:.2f} & {:.2f} & ".format(professions[tops[i]], sim[0][tops[i]], sim[1][tops[i]])
    female += " {} & {:.2f} & {:.2f} \\ \n".format(professions[tops_compl[i]], sim_compl[0][tops_compl[i]], sim_compl[1][tops_compl[i]])
    male += " {} ({:.2f} & {:.2f} & ".format(professions[bottoms[i]], sim[0][bottoms[i]], sim[1][bottoms[i]])
    male += " {} ({:.2f} & {:.2f} \\ \n".format(professions[bottoms_compl[i]], sim_compl[0][bottoms_compl[i]], sim_compl[1][bottoms_compl[i]])

table = female + male


