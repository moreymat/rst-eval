"""This utility script trains a (re)labeller for RST edges.

Given the path to a relation labelling dataset, it trains a classifier
and evaluates it.
"""

import argparse
import codecs
from collections import defaultdict
import copy
import os

from sklearn.datasets import load_svmlight_file, load_svmlight_files
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.model_selection import cross_val_score

from educe.rst_dt.deptree import _ROOT_HEAD, _ROOT_LABEL


# build mapping from int to label (reverse label encoding)
dset_rel_folder = os.path.join(
    '..', 'rst-eval-data', 'TMP/syn_pred_coarse_REL'
)
dset_rel_train = os.path.join(dset_rel_folder, 'TRAINING.relations.sparse')
dset_rel_test = os.path.join(dset_rel_folder, 'TEST.relations.sparse')

with codecs.open(dset_rel_train, mode='rb', encoding='utf-8') as f_train:
    header = f_train.readline()
    header_prefix = '# labels: '
    assert header.startswith(header_prefix)
    # DEBUG? explicit cast from unicode to str
    labels = [str(lbl) for lbl in header[len(header_prefix):].split()]
    int2lbl = dict(enumerate(labels, start=1))
    lbl2int = {lbl: i for i, lbl in int2lbl.items()}
    # unrelated = lbl2int["UNRELATED"]
    # root = lbl2int["ROOT"]

# 2017-12-14 relation (re)labeller
# DIRTY load the feature vector for all *candidate* edges in the TEST
# set (for predict())
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


if False:
    # load the relation TRAIN and TEST sets
    X_rel_train, y_rel_train, X_rel_test, y_rel_test = load_svmlight_files(
        (dset_rel_train, dset_rel_test),
        zero_based=False
    )
    rel_clf = LogisticRegressionCV(penalty='l1', solver='liblinear',
                                   n_jobs=3)
    # train relation classifier, cross-validate performance on train
    scores = cross_val_score(rel_clf, X_rel_train, y_rel_train, cv=10)
    print(scores)
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
    # fit a 
    rel_clf = rel_clf.fit(X_rel_train, y_rel_train)
    print(rel_clf.score(X_rel_test, y_rel_test))


class RelationRelabeller(object):
    """Predict the coarse-grained RST relation of dependencies.

    Dependencies headed by the fake root node are labelled "ROOT" by
    convention.

    Parameters
    ----------
    mul_clf : sklearn classifier
        Multi-class classifier for RST (coarse-grained) relations.
    """

    def __init__(self, mul_clf=LogisticRegression(penalty='l1', solver='liblinear', n_jobs=3), model_split='none'):
        """Init"""
        self.model_split = model_split
        if model_split == 'none':
            self.mul_clf = mul_clf
        elif model_split == 'sent':
            self.mul_clf_intra = copy.deepcopy(mul_clf)
            self.mul_clf_inter = copy.deepcopy(mul_clf)
        else:
            raise ValueError("model_split?")

    def fit(self, X, y):
        """Fit.

        FIXME X is currently expected to be a (flat) list of candidate
        edges instead of a list of RstDepTrees.
        """
        if self.model_split == 'none':
            self.mul_clf = self.mul_clf.fit(X, y)
            if True:  # verbose
                scores = cross_val_score(self.mul_clf, X, y, cv=10)
                print(scores)
                print("Accuracy: %0.2f (+/- %0.2f)" % (
                    scores.mean(), scores.std() * 2))
        elif self.model_split == 'sent':
            assert len(X) == 2  # intra, inter
            assert len(y) == 2  # intra, inter
            # * intra
            self.mul_clf_intra = self.mul_clf_intra.fit(X[0], y[0])
            if True:  # verbose
                scores = cross_val_score(self.mul_clf_intra, X[0], y[0], cv=10)
                print(scores)
                print("Accuracy: %0.2f (+/- %0.2f)" % (
                    scores.mean(), scores.std() * 2))
            # * inter
            self.mul_clf_inter = self.mul_clf_inter.fit(X[1], y[1])
            if True:  # verbose
                scores = cross_val_score(self.mul_clf_inter, X[1], y[1], cv=10)
                print(scores)
                print("Accuracy: %0.2f (+/- %0.2f)" % (
                    scores.mean(), scores.std() * 2))

        return self

    def predict(self, X):
        """Predict relation of edges in RstDepTrees X from the TEST set.
        """
        y = []
        for dtree in X:
            doc_name = dtree.origin.doc
            yi = []
            for i, (head, rel) in enumerate(zip(dtree.heads, dtree.labels)):
                if i == 0:
                    # fake root !? maybe we shouldn't write anything
                    # here ;
                    # FIXME check how to be consistent throughout educe and
                    # eval code
                    # yi.append(_ROOT_LABEL)
                    yi.append(None)
                elif head == 0:
                    # TODO check the expected value (consistency)
                    yi.append(_ROOT_LABEL)
                else:
                    # regular edge
                    line_idx = pair_map[doc_name][head][i]
                    # X_test[line_idx,:] is a matrix with 1 row
                    Xi = X_test[line_idx,:]
                    if self.model_split == 'none':
                        try:
                            y_pred = self.mul_clf.predict(Xi)
                        except ValueError:
                            print(Xi)
                            raise
                    elif self.model_split == 'sent':
                        # same_sentence_intra_{right,left}: 269, 303
                        # our vocab is 1-based but sklearn converts it to
                        # 0-based ;
                        # same_para_* : 103, 158, 234, 314
                        if ((Xi[0, 268] == 1 or Xi[0, 302] == 1) and
                            (Xi[0, 102] == 1 or Xi[0, 157] == 1 or
                             Xi[0, 233] == 1 or Xi[0, 313] == 1)):
                            sel_clf = self.mul_clf_intra
                        else:
                            sel_clf = self.mul_clf_inter
                        #
                        try:
                            y_pred = sel_clf.predict(Xi)
                        except ValueError:
                            print(Xi)
                            raise
                    # append prediction
                    try:
                        yi.append(int2lbl[int(y_pred[0])])
                        if False and rel != int2lbl[int(y_pred[0])]:
                            print(doc_name, head, i,
                                  rel, int2lbl[int(y_pred[0])])  # DEBUG
                    except KeyError:
                        raise ValueError("Weird prediction: {}".format(
                            y_pred))
            y.append(yi)
        return y
