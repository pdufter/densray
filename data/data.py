import os

import numpy as np

from tqdm import tqdm
from collections import defaultdict

from utils.utils import store


class Embeddings(object):
    """Class to load, edit and store word embeddings.

    Attr:
        X: embedding matrix
        W: list of words
        Wset: set of words
    """

    def __init__(self, log):
        """Initalize the wrapper

        Args:
            log: a logger object
        """
        self.log = log

    def load(self, path, load_first_n=None, header=True):
        """Load word embeddings in word2vec format from a txt file.

        Args:
            path: path to the embedding file
            load_first_n: int; how many lines to load
            header: bool; whether the embedding file contains a header line
        """
        self.path = path
        self.log.info("loading embeddings: {}".format(self.path))

        fin = open(self.path, 'r')

        if header:
            n, d = map(int, fin.readline().split())
        else:
            n, d = None, None

        data = {}
        count = 0
        for line in tqdm(fin):
            count += 1
            if load_first_n is not None and count > load_first_n:
                break
            tokens = line.rstrip().split(' ')
            data[tokens[0]] = list(map(float, tokens[1:]))

        self.W = list(data.keys())
        self.Wset = set(self.W)
        self.X = np.vstack(tuple([data[x] for x in self.W]))

        self.log.info("loaded {} / {} vectors with dimension {}.".format(len(self.W), n, self.X.shape[1]))

    def normalize(self):
        """Normalize the embeddings with l2 norm
        """
        self.X = (self.X.transpose() / np.linalg.norm(self.X, axis=1)).transpose()

    def filter(self, relevant):
        """Filter the embeddings to contain only words from "relevant".

        Args:
            relevant: iterable of words which should be kept
        """
        relevant = set(relevant)
        choose = []
        for word in self.W:
            if word in relevant:
                choose.append(True)
            else:
                choose.append(False)
        self.W = list(np.array(self.W)[choose])
        self.Wset = set(self.W)
        self.X = self.X[choose]

        self.log.info("filtered for {} / {} words.".format(len(relevant), len(self.W)))

    def store(self, fname):
        """Store the embedding space

        Args:
            fname: path to the file
        """
        outfile = store(fname)
        n, dim = self.X.shape
        outfile.write("{} {}\n".format(n, dim))
        for i in range(n):
            outfile.write(self.W[i])
            for k in range(dim):
                outfile.write(" {}".format(self.X[i, k]))
            outfile.write("\n")
        outfile.close()


class Lexicon(object):
    """Class to load, edit and store a lexicon.

    Attr:
        L: dictionary with different versions of the lexicon.
    """

    def __init__(self, log):
        """Initalize the lexicon

        Args:
            log: a logger object
        """
        self.log = log
        self.L = {"countable": [],
                  "ranked": [],
                  "continuous": []}

    def filter_words(self, relevant):
        """Filter the lexicon to contain only words from "relevant".

        Args:
            relevant: iterable of words which should be kept
        """
        relevant = set(relevant)
        for version in self.L:
            tmp = [(k, v) for k, v in self.L[version] if k in relevant]
            self.log.info("Filtering lexicon: {} / {} remaining.".format(len(tmp), len(self.L[version])))
            self.L[version] = tmp

    def load(self, path, version):
        """Load a lexicon from a file.

        Args:
            path: input path; one line looks like "word\sscore\n"
            version: whether the lexicon is countable, continuous or a ranking; countable has binary (integer) values, continuous float values and ranking reflects a ranking, but the actual values are irrelevant.
        """
        self.path = path
        infile = open(self.path, 'r')
        lexicon = []
        count = 0
        for i, line in enumerate(infile):
            count += 1
            line = line.replace("\n", "")
            try:
                score = line.split()[-1]
                word = line[:-len(score)].strip()
                score = float(score)
                lexicon.append((word, score))
            except:
                self.log.warning("Unexpected format in line {} from {}".format(i, self.path))
        self.log.info("loaded {} / {} lexicon entries.".format(len(lexicon), count))
        self.L[version] = lexicon
        self.version = version

    def remove_inconsistencies(self, remove_all=False):
        """Remove potential inconsistencies from the lexicon.

        Args:
            remove_all: whether to remove all instances of the inconcistency or keep one instance (the first one).
        """
        if remove_all:
            values = defaultdict(list)
            for k, v in self.L[self.version]:
                values[k].append(v)
            inconsistencies = set([k for k, v in values.items() if len(set(v)) > 1])
            self.log.info("Removed {} inconsistencies.".format(len(inconsistencies)))
            self.L[self.version] = [(k, v) for k, v in self.L[self.version] if k not in inconsistencies]
        else:
            seen = {}
            for k, v in self.L[self.version]:
                if k not in seen:
                    seen[k] = v
            self.log.info("Removed {} inconsistencies.".format(len(self.L[self.version]) - len(seen)))
            self.L[self.version] = list(seen.items())

    def binarise(self, mymap=None, neg=None, pos=None):
        """Get a binary version of the lexicon and store it in "countable"

        Args:
            mymap: map from integers to binary values
            neg: interval (e.g. [-float('inf'), 0]) which continuous scores are considered as "-1"; if None us identidy map.
            pos: same as neg for "1"; if None use median as threshold.
        """
        if self.version == 'countable':
            if mymap is None:
                mymap = {1: 1, -1: -1}
            # filter relevant words
            self.L["countable"] = [(k, v) for k, v in self.L["countable"] if v in mymap]
            self.L["countable"] = [(k, int(mymap[v])) for k, v in self.L["countable"]]
        elif self.version == 'continuous':
            if neg is None or pos is None:
                # use median
                median = np.median([v for k, v in self.L['continuous']])
                neg = [-float('inf'), median]
                pos = [median, float('inf')]
            relevant = [k for k, v in self.L["continuous"] if (neg[0] <= v < neg[1]) or (pos[0] <= v < pos[1])]
            self.filter_words(relevant)
            tmp = []
            for k, v in self.L['continuous']:
                if neg[0] <= v <= neg[1]:
                    tmp.append((k, -1))
                elif pos[0] <= float(v) <= pos[1]:
                    tmp.append((k, 1))
            self.L['countable'] = tmp

    def compute_ranks(self):
        """Get a ranked version of the lexicon and store it in "ranked"
        """
        # check number of ties
        n_ties = len(self.L[self.version]) - len(set([v for k, v in self.L[self.version]]))
        self.log.info("Computing ranks. No. of ties: {} / {}".format(n_ties, len(self.L[self.version])))
        tmp = sorted(self.L[self.version], key=lambda x: x[1])
        self.L['ranked'] = [(k, i) for i, (k, _) in enumerate(tmp)]

    def store(self, fname, version=None):
        """Store the lexicon.

        Args:
            fname: path where to store
            version: if given, just store the specific version of the lexicon.
        """
        if version is None:
            for version in self.L:
                outfile = store(fname + "_" + version + ".txt")
                for k, v in self.L[version]:
                    outfile.write("{} {}\n".format(k, v))
                outfile.close()
        else:
            outfile = store(fname)
            for k, v in self.L[version]:
                outfile.write("{} {}\n".format(k, v))
            outfile.close()

    def normalize(self):
        """Min-Max Normalize the continuous lexicon.
        """
        score_min = min([v for k, v in self.L['continuous']])
        score_max = max([v for k, v in self.L['continuous']])
        self.L['continuous'] = [(k, (v - score_min) / (score_max - score_min))for k, v in self.L['continuous']]

    def invert(self):
        """Invert the continuous lexicon.
        """
        self.L['continuous'] = [(k, -v) for k, v in self.L['continuous']]


class GoogleAnalogy(object):
    """Class to load the Google Analogy Dataset.
    """

    def __init__(self, log):
        """Initalize the wrapper

        Args:
            log: a logger object
        """
        self.log = log

    def load(self, path):
        """Load the Analogy Dataset

        Args:
            path: path to the dataset
        """
        self.path = path
        infile = open(self.path, 'r')
        chapter = None
        self.tuples = defaultdict(set)
        for line in infile:
            line = line.replace("\n", "")
            if line[0] == ":":
                chapter = line.split()[1].strip()
                continue
            a1, b1, a2, b2 = line.split()
            self.tuples[chapter] = self.tuples[chapter] | set([(a1, (b1,))])
            self.tuples[chapter] = self.tuples[chapter] | set([(a2, (b2,))])
        for chapter in self.tuples:
            self.tuples[chapter] = list(self.tuples[chapter])

    def get_classes(self, first_only=True):
        """Out of the analogy pairs create a lexion. 

        We consider all left/right elements of the pair to be in one class, respectively.

        Args:
            first_only: whether to consider all right elements or just the first one
        """
        self.L = defaultdict(list)
        for chapter in self.tuples:
            for x in self.tuples[chapter]:
                self.L[chapter].append((x[0], 1))
                if first_only:
                    self.L[chapter].append((x[1][0], -1))
                else:
                    for b in x[1]:
                        self.L[chapter].append((b, -1))


class SingleBats(object):
    """Class to load one chapter of the BATS Dataset
    """

    def __init__(self, log):
        """Initalize the wrapper

        Args:
            log: a logger object
        """
        self.log = log

    def load(self, path):
        """Load the BATS Dataset

        Args:
            path: path to the dataset
        """
        self.path = path
        infile = open(self.path, 'r')
        self.tuples = []
        for line in infile:
            line = line.replace("\n", "")
            part1, part2 = line.split()
            w1 = part1
            w2 = tuple(part2.split("/"))
            self.tuples.append((w1, w2))
        infile.close()

    def get_classes(self, first_only=True):
        """Out of the analogy pairs create a lexion.

        We consider all left/right elements of the pair to be in one class, respectively.

        Args:
            first_only: whether to consider all right elements or just the first one
        """
        self.L = []
        for x in self.tuples:
            self.L.append((x[0], 1))
            if first_only:
                self.L.append((x[1][0], -1))
            else:
                for b in x[1]:
                    self.L.append((b, -1))


class Bats(object):
    """Class to unify all chapter of BATS in one class.

    This is the equivalent of GoogleAnalogy for BATS.
    """

    def __init__(self, log):
        """Initalize the wrapper

        Args:
            log: a logger object
        """
        self.log = log
        self.chapter_paths = [
                     "1_Inflectional_morphology/I02 [noun - plural_irreg].txt",
                     "1_Inflectional_morphology/I07 [verb_inf - Ved].txt",
                     "1_Inflectional_morphology/I10 [verb_3pSg - Ved].txt",
                     "1_Inflectional_morphology/I01 [noun - plural_reg].txt",
                     "1_Inflectional_morphology/I08 [verb_Ving - 3pSg].txt",
                     "1_Inflectional_morphology/I03 [adj - comparative].txt",
                     "1_Inflectional_morphology/I04 [adj - superlative].txt",
                     "1_Inflectional_morphology/I06 [verb_inf - Ving].txt",
                     "1_Inflectional_morphology/I05 [verb_inf - 3pSg].txt",
                     "1_Inflectional_morphology/I09 [verb_Ving - Ved].txt",
                     "2_Derivational_morphology/D02 [un+adj_reg].txt",
                     "2_Derivational_morphology/D10 [verb+ment_irreg].txt",
                     "2_Derivational_morphology/D06 [re+verb_reg].txt",
                     "2_Derivational_morphology/D07 [verb+able_reg].txt",
                     "2_Derivational_morphology/D03 [adj+ly_reg].txt",
                     "2_Derivational_morphology/D05 [adj+ness_reg].txt",
                     "2_Derivational_morphology/D09 [verb+tion_irreg].txt",
                     "2_Derivational_morphology/D01 [noun+less_reg].txt",
                     "2_Derivational_morphology/D04 [over+adj_reg].txt",
                     "2_Derivational_morphology/D08 [verb+er_irreg].txt",
                     "3_Encyclopedic_semantics/E01 [country - capital].txt",
                     "3_Encyclopedic_semantics/E06 [animal - young].txt",
                     "3_Encyclopedic_semantics/E04 [name - nationality].txt",
                     "3_Encyclopedic_semantics/E03 [UK_city - county].txt",
                     "3_Encyclopedic_semantics/E02 [country - language].txt",
                     "3_Encyclopedic_semantics/E08 [animal - shelter].txt",
                     "3_Encyclopedic_semantics/E09 [things - color].txt",
                     "3_Encyclopedic_semantics/E07 [animal - sound].txt",
                     "3_Encyclopedic_semantics/E10 [male - female].txt",
                     "3_Encyclopedic_semantics/E05 [name - occupation].txt",
                     "4_Lexicographic_semantics/L06 [meronyms - part].txt",
                     "4_Lexicographic_semantics/L05 [meronyms - member].txt",
                     "4_Lexicographic_semantics/L01 [hypernyms - animals].txt",
                     "4_Lexicographic_semantics/L07 [synonyms - intensity].txt",
                     "4_Lexicographic_semantics/L03 [hyponyms - misc].txt",
                     "4_Lexicographic_semantics/L04 [meronyms - substance].txt",
                     "4_Lexicographic_semantics/L10 [antonyms - binary].txt",
                     "4_Lexicographic_semantics/L08 [synonyms - exact].txt",
                     "4_Lexicographic_semantics/L09 [antonyms - gradable].txt",
                     "4_Lexicographic_semantics/L02 [hypernyms - misc].txt"]

    def load(self, path, first_only=True):
        """Load the BATS Dataset

        Args:
            path: path to the dataset
            first_only: whether to consider all right elements or just the first one
        """
        self.path = path
        self.tuples = dict()
        self.L = dict()
        for chapter_path in self.chapter_paths:
            chapter = chapter_path.split("/")[-1]
            # if chapter[0] == "D" or chapter[0] == "L" or chapter[0] == "I":
            #     continue
            bats = SingleBats(self.log)
            bats.load(os.path.join(self.path, chapter_path))
            bats.get_classes(first_only=first_only)
            self.tuples[chapter] = bats.tuples
            self.L[chapter] = bats.L

    def adjust_capitalisation(self, wordlist):
        """Adjust the capitalisation

        BATS is lowercased, but most wordspaces are cased.
        Lowercase vector of "washington" is of low quality.
        Thus convert the case of BATS to fit the word embeddings.

        Args:
            path: path to the dataset
            first_only: whether to consider all right elements or just the first one
        """
        w = wordlist
        wset = set(w)
        wlower = [x.lower() for x in wordlist]
        wlowerset = set(wlower)
        bw2truew = {}
        for chapter in self.tuples:
            for tup in self.tuples[chapter]:
                tmp_words = set([tup[0]] + list(tup[1]))
                for word in tmp_words:
                    if word in wset:
                        bw2truew[word] = word
                    elif word in wlowerset:
                        idx = wlower.index(word)
                        bw2truew[word] = w[idx]
                    else:
                        bw2truew[word] = word
            # transform the data
            self.tuples[chapter] = [(bw2truew[w1], tuple([bw2truew[x] for x in w2])) for w1, w2 in self.tuples[chapter]]
            self.L[chapter] = [(bw2truew[k], v) for k, v in self.L[chapter]]
