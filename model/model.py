import os

import numpy as np

from sklearn.svm import SVR, SVC
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.cross_decomposition import CCA
from scipy.linalg import null_space
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict

from utils.utils import store
from data.data import Lexicon


class Densifier(object):
    """Python wrapper of Densifier Matlab Implementation

    To obtain the Matlab implementaiton, contact the authors of
    https://arxiv.org/pdf/1602.07572.pdf
    """

    def __init__(self, log, Embeddings, Ltrain, Ltest):
        """Initalize the wrapper

        Args:
            log: a logger object
            Embeddings: a embedding object
            Ltrain: lexicon object with training dictionary
            Ltest: lexicon object with test dictionary
        """
        self.log = log
        self.embed = Embeddings
        self.Ltrain = Ltrain
        self.Ltest = Ltest
        self.matlab_path = "/mounts/Users/cisintern/philipp/Dokumente/FeatureCat/"
        assert os.path.exists(self.matlab_path), "No Matlab implementation for Densifer found. Please clone git@github.com:pdufter/FeatureCat.git and specify the corresponding directory in this class."

    def prepare_data(self, path):
        """Stores the training data on disk to make it accessible to matlab

        Args:
            path: path where to store the data
        """
        self.path = path
        self.embedding_path = self.path + ",matlab,matlab,embeddings.txt"
        self.ltrain_path = self.path + ",matlab,matlab,trainlexicon.txt"
        self.ltest_path = self.path + ",matlab,matlab,testlexicon.txt"
        self.embed.store(self.embedding_path)
        self.Ltrain.store(self.ltrain_path, version='countable')
        self.Ltest.store(self.ltest_path, version=self.Ltest.version)
        self.line_count = self.embed.X.shape[0] + 1
        self.outfilename = self.path + ",matlab,matlab.predictions"

    def fit_predict(self):
        """Calls the matlabl implementation through the command line

        The output of matlab is store on disk.
        """
        self.log.info("Calling Densifier through commandline.")
        command = """matlab -r 'addpath(\"{}\");FeatureCat(\"{}\", {}, \"{}\", \"{}\", \"{}\");quit;' """.format(self.matlab_path, self.embedding_path, self.line_count, self.ltrain_path, self.ltest_path, self.outfilename)
        os.system(command)


class DensRay(object):
    """Implements the DensRay method.

    There are two basid models: "binary" and "continuous".
    For binary input lexicons both are equivalent.
    See the paper for more details.
    """

    def __init__(self, log, Embeddings, Lexicon):
        """Initialize DensRay

        Args:
            log: logger object
            Embeddings: embedding object
            Lexicon: the lexicon which is used to fit DensRay
        """
        self.log = log
        self.embed = Embeddings
        self.lexic = Lexicon

    def fit(self, weights=None, model='binary', normalize_D=True, normalize_labels=True):
        """Fit DensRay

        Args:
            weights: only for binary model; how to weight the two
                summands; if none: apply dynamic weighting. Example input: [1.0, 1.0]
            model: 'binary' or 'continuous'; which model version of Densray to use
            normalize_D: bool whether to normalize the difference vectors with l2 norm
            normalize_labels: bool whether to normalize the predicted labels.
        """
        if model == 'binary':
            self.prepare_data_binary()
            self.computeA_binary_part1(normalize_D=normalize_D)
            self.computeA_binary_part2(weights=weights)
        elif model == 'continuous':
            self.prepare_data_continuous()
            self.computeA_continuous(normalize_D=normalize_D, normalize_labels=normalize_labels)
        else:
            raise NotImplementedError
        self.compute_trafo()

    def prepare_data_binary(self):
        """Data preparation function for the binary model

        It selects the relevant vectors from the embedding space.
        """
        Lrel = [(k, v) for k, v in self.lexic.L['countable'] if k in self.embed.Wset]
        values = set([v for k, v in Lrel])
        assert len(values) == 2
        v1, v2 = values
        indexpos = []
        indexneg = []
        for k, v in Lrel:
            if v == v1:
                indexpos.append(self.embed.W.index(k))
            else:
                indexneg.append(self.embed.W.index(k))
        self.Xpos = self.embed.X[indexpos, :]
        self.Xneg = self.embed.X[indexneg, :]
        self.npos = self.Xpos.shape[0]
        self.nneg = self.Xneg.shape[0]

    def prepare_data_continuous(self):
        """Data preparation function for the continuous model

        It selects the relevant vectors from the embedding space.
        """
        if len(self.lexic.L['continuous']) == 0:
            self.log.warning("No continuous labels available, using countable labels instead.")
            self.lexic.L['continuous'] = self.lexic.L['countable']
        self.Wrel = [k for k, v in self.lexic.L['continuous'] if k in self.embed.Wset]
        self.scoresrel = np.array([v for k, v in self.lexic.L['continuous'] if k in self.Wrel])
        self.Xrel = self.embed.X[[self.embed.W.index(x) for x in self.Wrel]]

    @staticmethod
    def outer_product_sub_binary(v, M, normD):
        """Helper function to compute the sum of outer products

        While it is not very readable, it is more efficient than
        a brute force implementation.
        """
        D = v.transpose() - M
        if normD:
            norm = np.linalg.norm(D, axis=1)
            D[norm == 0.0] = 0.0
            norm[norm == 0.0] = 1.0
            D = D / norm[:, np.newaxis]
        return D.transpose().dot(D)

    @staticmethod
    def outer_product_sub_continuous(v, M, i, gammas, normD):
        """Helper function to compute the sum of outer products

        While it is not very readable, it is more efficient than
        a brute force implementation.
        """
        D = v.transpose() - M
        gamma = gammas[i] * gammas
        if normD:
            norm = np.linalg.norm(D, axis=1)
            D[norm == 0.0] = 0.0
            norm[norm == 0.0] = 1.0
            D = D / norm[:, np.newaxis]
        D1 = gamma[:, np.newaxis] * D
        return D1.transpose().dot(D)

    def store(self, fname):
        """Stores the transformation in npy format.

        Args:
            fname: path where to store the transformation.
        """
        np.save(fname, self.T)

    def computeA_binary_part1(self, normalize_D=True):
        """First part of computing the matrix A.

        Args:
            normalize_D: bool whether to normalize the difference vectors with l2 norm.

        Todo:
            can be made more efficient (dot product is symmetric and we compute both directions here)
        """
        dim = self.Xpos.shape[1]
        self.A_equal = np.zeros((dim, dim))
        self.A_unequal = np.zeros((dim, dim))
        for ipos in tqdm(range(self.npos), desc="compute matrix part1", leave=False):
            v = self.Xpos[ipos:ipos + 1, :].transpose()
            self.A_equal += self.outer_product_sub_binary(v, self.Xpos, normalize_D)
            self.A_unequal += self.outer_product_sub_binary(v, self.Xneg, normalize_D)
        for ineg in tqdm(range(self.nneg), desc="compute matrix part2", leave=False):
            v = self.Xneg[ineg:ineg + 1, :].transpose()
            self.A_equal += self.outer_product_sub_binary(v, self.Xneg, normalize_D)
            self.A_unequal += self.outer_product_sub_binary(v, self.Xpos, normalize_D)

    def computeA_binary_part2(self, weights=None):
        """Second part of computing the matrix A.

        Args:
            weights: only for binary model; how to weight the two 
                summands; if none: apply dynamic weighting. Example input: [1.0, 1.0]
        """
        if weights is None:
            weights = [1 / (2 * self.npos * self.nneg), 1 / (self.npos**2 + self.nneg**2)]
        # normalize matrices for numerical reasons
        # note that this does not change the eigenvectors
        n1 = self.A_unequal.max()
        n2 = self.A_equal.max()
        weights = [weights[0] / max(n1, n2), weights[1] / max(n1, n2)]
        self.A = weights[0] * self.A_unequal - weights[1] * self.A_equal

    def computeA_continuous(self, normalize_D=True, normalize_labels=True):
        """Compute the matrix A for the continuous case.

        Args:
            normalize_D: normalize_D: bool whether to normalize the difference vectors with l2 norm.
            normalize_labels: bool whether to normalize the predicted labels.

        Todo:
            can be made more efficient (dot product is symmetric and we compute both directions here)
        """
        dim = self.Xrel.shape[1]
        self.A = np.zeros((dim, dim))
        gammas = self.scoresrel
        if normalize_labels:
            gammas = (gammas - gammas.mean()) / gammas.std()
        for i, w in tqdm(enumerate(self.Wrel), desc="compute matrix", leave=False):
            v = self.Xrel[i:i + 1, :].transpose()
            self.A += self.outer_product_sub_continuous(v, self.Xrel, i, gammas, normalize_D)
        self.A = - self.A / self.A.max()

    def compute_trafo(self):
        """Given A, this function computes the actual Transformation.

        It essentially just does an eigenvector decomposition.
        """
        # note that (eigvecs(A) = eigvecs (A'A))
        # when using eigh the are always real
        self.eigvals, self.eigvecs = np.linalg.eigh(self.A)
        # need to sort the eigenvalues
        idx = self.eigvals.argsort()[::-1]
        self.eigvals, self.eigvecs = self.eigvals[idx], self.eigvecs[:, idx]
        self.T = self.eigvecs
        assert np.allclose(self.T.transpose().dot(self.T), np.eye(self.T.shape[0])), "self.T not orthonormal."


class Regression(object):
    """Implements a regression based method of obtaining interpretable dimensions.

    Different models from sklearn are available.
    The word "regression" is not really appropriate, as one can also apply SVMs.
    All models: ["logistic", "svm", "linear", "svr", "cca"]
    """

    def __init__(self, log, Embeddings, Lexicon):
        """Initialize the Regression

        Args:
            log: logger object
            Embeddings: embedding object
            Lexicon: the lexicon which is used to fit the model
        """
        self.log = log
        self.embed = Embeddings
        self.lexic = Lexicon

    def prepare_data(self, model):
        """Prepare the data (i.e. select vectors and create labels)

        Args:
            model: string; a value in ["logistic", "svm", "linear", "svr", "cca"]
        """
        if model in ["logistic", "svm"]:
            version = 'countable'
        elif model in ["linear", "svr", "cca"]:
            version = self.lexic.version
        else:
            raise ValueError("Model unknown.")
        idxs = []
        ys = []
        words = []
        for k, v in self.lexic.L[version]:
            if k in self.embed.Wset:
                idxs.append(self.embed.W.index(k))
                ys.append(v)
                words.append(k)
        self.Wrel = words
        self.Xrel = self.embed.X[idxs, :]
        self.Yrel = np.array(ys)
        if model == 'logistic':
            self.Yrel[self.Yrel == -1] = 0
        if model == 'svm':
            self.Yrel[self.Yrel == 0] = -1

    def fit(self, model):
        """Fits the model and creates a (random) orthogonal transformation.

        Args:
            model: string; a value in ["logistic", "svm", "linear", "svr", "cca"]
        """
        if model == 'linear':
            self.mod = LinearRegression()
        elif model == 'logistic':
            self.mod = LogisticRegression()
        elif model == 'svr':
            self.mod = SVR(kernel='linear')
        elif model == 'svm':
            self.mod = SVC(C=1.0, kernel='linear')
        elif model == 'cca':
            self.mod = CCA(n_components=1, scale=True, max_iter=500, tol=1e-06, copy=True)
            self.mod.intercept_ = 0.0
        self.mod.fit(self.Xrel, self.Yrel)
        # now compute T with a random orthogonal basis
        # todo potential bug: what to do with the intercept_?
        w0 = self.mod.coef_  # + self.mod.intercept_
        if len(w0.shape) < 2:
            w0 = w0.reshape(1, -1)
        w0 = w0 / np.linalg.norm(w0)
        Wcompl = null_space(w0)
        self.T = np.hstack((w0.transpose(), Wcompl))
        assert np.allclose(self.T.transpose().dot(self.T), np.eye(self.T.shape[0])), "self.T not orthonormal."

    def store(self, fname):
        """Stores the transformation in npy format.

        Args:
            fname: path where to store the transformation.
        """
        np.save(fname, self.T)


class LexIndPredictor(object):
    """Given an interpretable word space and queries, predict lexical scores (e.g. for sentiment.
    """

    def __init__(self, log, embeddings, queries, T):
        """Initialize the predictor.

        Args:
            log: logger object
            embeddings: word embedding object
            queries: list of strings with queries
            T: np.array; linear transformation
        """
        self.log = log
        self.embeds = embeddings
        self.queries = queries
        self.T = T

    def predict(self, method, dim_weights=None):
        """Predict scores for the query words.

        Args:
            method: string; either "first_dimension", "first_n_dimensions"
            dim_weights: only available if method == "first_n_dimensions"; how to weight the scores in the first n dimensions.
        """
        X_trafo = self.embeds.X.dot(self.T)
        self.predictions = []
        for k in self.queries:
            if k not in self.embeds.Wset:
                continue
            if method == 'first_dimension':
                score = X_trafo[self.embeds.W.index(k), 0]
            elif method == 'first_n_dimensions':
                n = len(dim_weights)
                score = 0.0
                for i in range(n):
                    score += dim_weights[i] * X_trafo[self.embeds.W.index(k), i]
            self.predictions.append((k, score))

    def store(self, fname):
        """Stores the predictions in a text file.

        Args:
            fname: path where to store the predictions.
        """
        outfile = store(fname)
        for k, v in self.predictions:
            outfile.write("{} {}\n".format(k, v))
        outfile.close()


class AnalogyPredictor(object):
    """Given an interpretable word space and queries, solve the word analogy task.
    """

    def __init__(self, log, embeddings, analogies):
        """Initialize the predictor.

        Args:
            log: logger object
            embeddings: word embedding object
            analogies: list of analogy pairs on which to train the predictor
        """
        self.log = log
        self.embeds = embeddings
        self.ana = self.filter_analogies(analogies)
        self.get_lexicon()

    def get_lexicon(self):
        """Prepare the training data.

        Converts the analogy pairs into a lexicon
        to enable training of an SVM, DensRay, etc.
        """
        self.lex = Lexicon(self.log)
        self.lex.version = 'countable'
        self.lex.L['countable'] = []
        for tup in self.ana:
            self.lex.L['countable'].append((tup[0], 1))
            self.lex.L['countable'].append((tup[1][0], -1))

    def fit_classifier(self, modeltype, model):
        """Fit the chosen model.

        Args:
            modeltype: "regression" or "densray"
            model: for "densray" one in ["binary", "continuous"]; for "regression" one in ["logistic", "svm", "linear", "svr", "cca"]
        """
        if modeltype == "regression":
            trafo = Regression(self.log, self.embeds, self.lex)
            trafo.prepare_data(model)
            trafo.fit(model)
            T = trafo.T
            if model == "logistic":
                self.scores_prob = trafo.mod.predict_proba(self.embeds.X)[:, 0]
            else:
                self.scores_prob = None
        elif modeltype == "densray":
            trafo = DensRay(self.log, self.embeds, self.lex)
            trafo.fit(weights=None, model=model, normalize_D=True, normalize_labels=False)
            T = trafo.T
            self.scores_prob = None

        X_trafo = self.embeds.X.dot(T)
        scores = X_trafo[:, 0]
        self.X_compl = X_trafo[:, 1:]
        # potentially invert scores
        sc0 = scores[self.embeds.W.index(self.ana[0][0])]
        sc1 = scores[self.embeds.W.index(self.ana[0][1][0])]
        if sc0 > sc1:
            scores = -scores
        # normalize
        self.scores = (scores - scores.min()) / (scores.max() - scores.min())

    def predict(self, queries, method=None, use_compl=None, use_prob=None):
        """Given queries, complete the analogies.

        Args:
            queries: list of strings with the query words
            method: "clcomp" or "lrcos"; this just steers meaningful combinations of the remaining boolean parameters
            use_compl: whether to consider distance in the complementary embedding space
            use_prob: whether to use the probability score (only availbale for logistic regression)

        Returns:
            list of tuples [(query1, prediction1), (query2, prediction2), ...]
        """
        if method == 'clcomp':
            use_compl = True
            use_prob = False
        elif method == "lrcos":
            use_compl = False
            use_prob = True
        elif method is None:
            assert use_compl is not None and use_prob is not None, "Please provide a method to predict analogies."

        pred = defaultdict(lambda: "<NONE>")
        rel_queries = [x for x in queries if x in self.embeds.Wset]
        queries_index = [self.embeds.W.index(x) for x in rel_queries]

        if len(rel_queries) != 0:
            if use_compl:
                queries_vec = self.X_compl[queries_index, :]
                sim = cosine_similarity(queries_vec, self.X_compl)
            else:
                queries_vec = self.embeds.X[queries_index, :]
                sim = cosine_similarity(queries_vec, self.embeds.X)

            if use_prob:
                assert self.scores_prob is not None, "prob scores only available with logistics regression."
                scores = self.scores_prob
            else:
                scores = self.scores

            for i, query in enumerate(rel_queries):
                # multiply both scores and get word with highest score
                new_scores = np.multiply(scores, sim[i, :])
                new_scores_sorted = new_scores.argsort()
                w_candidate = query
                r = 0
                while w_candidate.lower() == query.lower():
                    r += 1
                    candidate = new_scores_sorted[-r]
                    w_candidate = self.embeds.W[candidate]
                pred[query] = w_candidate

        return [(q, pred[q]) for q in queries]

    def is_answerable(self, pair):
        """Decides whether a analogy pair is answerable.

        A pair is answerable if both member of the analogy are contained in the wordspace

        Args:
            pair: analogy pair in the BATS format (a, (a'1, a'2, ...))

        Returns:
            bool: true if pair is answerable, else false
        """
        return pair[0] in self.embeds.Wset and pair[1][0] in self.embeds.Wset

    def filter_analogies(self, analogies):
        """Filters a list of analogies for answerability.

        Args:
            analogies: list of analogy pairs in the BATS format

        Returns:
            filtered list of analogy pairs in the BATS format
        """
        return [pair for pair in analogies if self.is_answerable(pair)]

    def get_distances(self):
        """Provides some analysis for the training analogies.

        Returns:
            ranks: list of ranks (rank 3 means the answer to the analog is the third nearest neighbour to the query)
            intersim: list of similarities between analogy pairs
            intrasim0: list of similarities within the left class of analogy pairs
            intrasim1: list of similarities within the right class of analogy pairs
        """
        q0 = [self.embeds.W.index(x[0]) for x in self.ana]
        q1 = [self.embeds.W.index(x[1][0]) for x in self.ana]

        v0 = self.embeds.X[q0, :]
        v1 = self.embeds.X[q1, :]

        intersim = np.diag(cosine_similarity(v0, v1))
        intrasim0 = cosine_similarity(v0, v0)
        intrasim1 = cosine_similarity(v1, v1)
        # compute ranks between query and answer
        rel_sim = cosine_similarity(v0, self.embeds.X)
        consider_max_n = 200
        mynns = rel_sim.argsort(axis=1)[:, -consider_max_n:]
        ranks = []
        for i in range(mynns.shape[0]):
            if q1[i] not in mynns[i, :]:
                ranks.append(consider_max_n)
            else:
                ranks.append(list(mynns[i, :])[::-1].index(q1[i]))
        # todo return lists and not np.arrays
        return ranks, intersim, intrasim0.flatten(), intrasim1.flatten()
