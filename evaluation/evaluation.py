import scipy.stats as stats
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


class LexIndEval(object):
    """Class to evaluate Lexicon Induction.
    """

    def __init__(self, gt, pred):
        """Initalize the wrapper

        Args:
            gt: ground truth lexicon [(word1, score2), (word2, score2), ...]
            pred: predicted score in same format
        """
        self.gt = gt
        self.pred = pred

    def prepare(self):
        """For the overlap of both lexicons reorder the lexicons.
        """
        assert set([k for k, v in self.gt]) == set([k for k, v in self.pred]), "Predictions and Groundtruth inconsistent."

        gt_sorted = sorted(self.gt, key=lambda x: x[0])
        pred_sorted = sorted(self.pred, key=lambda x: x[0])

        compare = []
        self.words = []
        for gtitem, preditem in zip(gt_sorted, pred_sorted):
            assert gtitem[0] == preditem[0]
            compare.append((gtitem[1], preditem[1]))
            self.words.append(gtitem[0])
        self.y_pred, self.y_true = zip(*compare)

    def compute_kendalls(self):
        """Computes kendalls tau

            Returns:
                tau: float; kendalls tau
                p_value: float;
        """
        tau, p_value = stats.kendalltau(self.y_pred, self.y_true)
        return tau, p_value

    def compute_classic(self):
        """Computes accuracy, precision, etc. for binary prediction only.

            Returns:
                ac: accuracy
                pr: precision
                re: recall
                f1: f1-score
        """
        ac = accuracy_score(self.y_true, self.y_pred)
        pr = precision_score(self.y_true, self.y_pred)
        re = recall_score(self.y_true, self.y_pred)
        f1 = f1_score(self.y_true, self.y_pred)
        return ac, pr, re, f1

    def view_errors(self):
        """Prints all errors.
        """
        for i, word in enumerate(self.words):
            if self.y_pred[i] != self.y_true[i]:
                print(word, self.y_true[i], self.y_pred[i])
