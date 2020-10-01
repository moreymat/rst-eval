"""This utility script trains a classifier for nuclearity of RST edges.

Given the path to a nuclearity dataset, it trains a classifier and
evaluates it.
"""

import argparse
import codecs
from collections import defaultdict
import copy
import itertools
import os
import sys

from sklearn.datasets import load_svmlight_file, load_svmlight_files
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder

from educe.rst_dt.annotation import NUC_N, NUC_S


# 2017-12-06 non-dummy nuc_clf
# DIRTY load the feature vectors of all candidate edges in the TEST
# set
feat_vecs = dict()
dset_folder = os.path.join(
    '..', 'rst-eval-data', 'TMP/syn_pred_coarse'
)
dset_test = os.path.join(dset_folder, 'TEST.relations.sparse')
# we use the original svmlight files whose label is the relation
# class (which we actually don't need here)
# FIXME read n_features from .vocab
X_test, y_lbl_test = load_svmlight_file(dset_test, n_features=46731,
                                        zero_based=False)
# build mapping from doc_name, src_idx, tgt_idx to line number
# in X_test
pairs = dset_test + '.pairings'
pair_map = defaultdict(lambda: defaultdict(dict))
with codecs.open(pairs, mode='rb', encoding='utf-8') as f_pairs:
    for i, line in enumerate(f_pairs):
        src_id, tgt_id = line.strip().split('\t')
        src_idx = (0 if src_id == 'ROOT'
                   else int(src_id.rsplit('_', 1)[1]))
        doc_name, tgt_idx = tgt_id.rsplit('_', 1)
        tgt_idx = int(tgt_idx)
        # print(line)
        # print(doc_name, src_idx, tgt_idx)
        pair_map[doc_name][src_idx][tgt_idx] = i
# end DIRTY


class RightBinaryNuclearityClassifier(object):
    """Predict the nuclearity of right-oriented dependencies (binary).

    The nuclearity of ordinary, right-oriented dependencies can be
    either `NUC_S` or `NUC_N` (NS or NN relations).
    Right-oriented dependencies from the fake root have nuclearity
    `NUC_R` by convention ; Left-oriented dependencies have nuclearity
    `NUC_S`.

    Parameters
    ----------
    bin_clf : sklearn classifier
        Binary classifier for right dependencies: NN vs NS.
    model_split : str, one of {'none', 'sent', 'sent-para'}
        Distinct models for subsets of instances.
    """

    def __init__(self, bin_clf=LogisticRegression(penalty='l1', solver='liblinear', n_jobs=2), model_split='none'):
        """Init"""
        self.model_split = model_split
        if model_split == 'none':
            self.bin_clf = bin_clf
        elif model_split == 'sent':
            self.bin_clf_intra = copy.deepcopy(bin_clf)
            self.bin_clf_inter = copy.deepcopy(bin_clf)
        else:
            raise ValueError("model_split?")

    def fit(self, X, y):
        """Fit.

        FIXME X is currently expected to be a (flat) list of candidate
        edges instead of a list of RstDepTrees.
        """
        if self.model_split == 'none':
            self.bin_clf = self.bin_clf.fit(X, y)
            if True:  # verbose
                scores = cross_val_score(self.bin_clf, X, y, cv=10)
                print(scores)
                print("Accuracy: %0.2f (+/- %0.2f)" % (
                    scores.mean(), scores.std() * 2))
        elif self.model_split == 'sent':
            assert len(X) == 2  # intra, inter
            assert len(y) == 2  # intra, inter
            # * intra
            self.bin_clf_intra = self.bin_clf_intra.fit(X[0], y[0])
            if True:  # verbose
                scores = cross_val_score(self.bin_clf_intra, X[0], y[0], cv=10)
                print(scores)
                print("Accuracy: %0.2f (+/- %0.2f)" % (
                    scores.mean(), scores.std() * 2))
            # * inter
            self.bin_clf_inter = self.bin_clf_inter.fit(X[1], y[1])
            if True:  # verbose
                scores = cross_val_score(self.bin_clf_inter, X[1], y[1], cv=10)
                print(scores)
                print("Accuracy: %0.2f (+/- %0.2f)" % (
                    scores.mean(), scores.std() * 2))

        return self

    def predict(self, X):
        """Predict nuclearity of edges in RstDepTrees X from the TEST set.

        Parameters
        ----------
        X : list of RstDepTree
            D-trees ; the feature vectors of all edges are already
            available from the global context.
        """
        y = []
        for dtree in X:
            doc_name = dtree.origin.doc
            yi = []
            for i, head in enumerate(dtree.heads):
                if i == 0:
                    # fake root !? maybe we shouldn't write anything
                    # here ;
                    # FIXME check how to be consistent throughout educe and
                    # eval code
                    yi.append(NUC_N)
                elif i < head:
                    # left edge: SN
                    yi.append(NUC_S)
                elif head == 0:
                    # FIXME NUC_R for edges from the root?
                    yi.append(NUC_N)
                else:
                    # right edge: NN or NS?
                    line_idx = pair_map[doc_name][head][i]
                    # X_test[line_idx,:] is a matrix with 1 row
                    Xi = X_test[line_idx,:]
                    if self.model_split == 'none':
                        try:
                            y_pred = self.bin_clf.predict(Xi)
                        except ValueError:
                            print(Xi)
                            raise
                    elif self.model_split == 'sent':
                        # same_sentence_intra_{right,left}: 269, 303
                        # our vocab is 1-based but sklearn converts it to
                        # 0-based ;
                        # check it's not a left dep
                        assert Xi[0, 302] == 0
                        #
                        if Xi[0, 268] == 1:
                            sel_clf = self.bin_clf_intra
                        else:
                            sel_clf = self.bin_clf_inter
                        #
                        try:
                            y_pred = sel_clf.predict(Xi)
                        except ValueError:
                            print(Xi)
                            raise
                    # append prediction
                    if y_pred == 1:
                        yi.append(NUC_N)
                    elif y_pred == 2:
                        yi.append(NUC_S)
                    else:
                        raise ValueError("Weird prediction: {}".format(
                            y_pred))

            y.append(yi)
        return y


if __name__ == "__main__":
    model_split = 'sent'  # {'none', 'sent'}
    # eval on intra- and inter-sent
    # * intra
    dset_folder_intra = os.path.join(
        os.path.expanduser('~'),
        'melodi/rst/irit-rst-dt/TMP/syn_pred_coarse_NUC_intrasent'
    )
    dset_train_intra = os.path.join(dset_folder_intra, 'TRAINING.relations.sparse')
    dset_test_intra = os.path.join(dset_folder_intra, 'TEST.relations.sparse')
    X_train_intra, y_train_intra, X_test_intra, y_test_intra = load_svmlight_files(
        (dset_train_intra, dset_test_intra),
        n_features=46731,
        zero_based=False
    )
    # * inter
    dset_folder_inter = os.path.join(
        os.path.expanduser('~'),
        'melodi/rst/irit-rst-dt/TMP/syn_pred_coarse_NUC_intersent'
    )
    dset_train_inter = os.path.join(dset_folder_inter, 'TRAINING.relations.sparse')
    dset_test_inter = os.path.join(dset_folder_inter, 'TEST.relations.sparse')
    X_train_inter, y_train_inter, X_test_inter, y_test_inter = load_svmlight_files(
        (dset_train_inter, dset_test_inter),
        n_features=46731,
        zero_based=False
    )
    #
    if model_split == 'none':
        # import the nuclearity TRAIN and TEST sets
        dset_folder = os.path.join(
            os.path.expanduser('~'),
            'melodi/rst/irit-rst-dt/TMP/syn_pred_coarse_NUC'
        )
        dset_train = os.path.join(dset_folder, 'TRAINING.relations.sparse')
        dset_test = os.path.join(dset_folder, 'TEST.relations.sparse')

        X_train, y_train, X_test, y_test = load_svmlight_files(
            (dset_train, dset_test),
            n_features=46731,
            zero_based=False
        )
        nuc_clf = LogisticRegressionCV(penalty='l1', solver='liblinear',
                                       n_jobs=3)
        # train nuclearity classifier, cross-validate performance on train
        scores = cross_val_score(nuc_clf, X_train, y_train, cv=10)
        print(scores)
        print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
        # fit a 
        nuc_clf = nuc_clf.fit(X_train, y_train)
        print(nuc_clf.score(X_test, y_test))
        print('separate eval on intra then inter')
        print(nuc_clf.score(X_test_intra, y_test_intra))
        print(nuc_clf.score(X_test_inter, y_test_inter))
    elif model_split == 'sent':
        # fit distinct classifiers for intra- and inter-sentential
        # * intra: train nuclearity classifier, cross-validate performance on train
        nuc_clf_intra = LogisticRegressionCV(penalty='l1', solver='liblinear',
                                             n_jobs=3)
        scores_intra = cross_val_score(nuc_clf_intra, X_train_intra, y_train_intra,
                                       cv=10)
        print(scores_intra)
        print("Accuracy: %0.2f (+/- %0.2f)" % (
            scores_intra.mean(), scores_intra.std() * 2))
        #
        nuc_clf_intra = nuc_clf_intra.fit(X_train_intra, y_train_intra)
        print(nuc_clf_intra.score(X_test_intra, y_test_intra))
        # * inter: train nuclearity classifier, cross-validate performance on train
        nuc_clf_inter = LogisticRegressionCV(penalty='l1', solver='liblinear',
                                             n_jobs=3)
        scores_inter = cross_val_score(nuc_clf_inter, X_train_inter, y_train_inter,
                                       cv=10)
        print(scores_inter)
        print("Accuracy: %0.2f (+/- %0.2f)" % (
            scores_inter.mean(), scores_inter.std() * 2))
        #
        nuc_clf_inter = nuc_clf_inter.fit(X_train_inter, y_train_inter)
        print(nuc_clf_inter.score(X_test_inter, y_test_inter))
