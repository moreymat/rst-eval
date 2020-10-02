"""This module evaluates the output of discourse parsers.

Included are dependency and constituency tree metrics.
"""

import argparse
import codecs
import itertools
import os

from sklearn.datasets import load_svmlight_files
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV

from educe.rst_dt.annotation import _binarize, SimpleRSTTree
from educe.rst_dt.corpus import (RstRelationConverter,
                                 Reader as RstReader)
from educe.rst_dt.dep2con import (DummyNuclearityClassifier,
                                  InsideOutAttachmentRanker)
from educe.rst_dt.deptree import RstDepTree
from educe.rst_dt.metrics.rst_parseval import (rst_parseval_detailed_report,
                                               rst_parseval_compact_report,
                                               rst_parseval_report,
                                               rst_parseval_similarity)
#
from attelo.metrics.deptree import (compute_uas_las,
                                    dep_compact_report,
                                    dep_similarity)

# local to this package
from braud_coling import (load_braud_coling_ctrees,
                                load_braud_coling_dtrees)
from braud_eacl import (load_braud_eacl_ctrees,
                              load_braud_eacl_dtrees)
from codra import load_codra_ctrees, load_codra_dtrees
from feng import load_feng_ctrees, load_feng_dtrees
from gcrf_tree_format import load_gcrf_ctrees, load_gcrf_dtrees
from hayashi_cons import (load_hayashi_hilda_ctrees,
                                load_hayashi_hilda_dtrees)
from hayashi_deps import (load_hayashi_dep_dtrees,
                                load_hayashi_dep_ctrees)
from ji import load_ji_ctrees, load_ji_dtrees
from li_qi import load_li_qi_ctrees, load_li_qi_dtrees
from li_sujian import (DEFAULT_FILE as LI_SUJIAN_OUT_FILE,
                             load_li_sujian_dep_ctrees,
                             load_li_sujian_dep_dtrees)
from ours import (load_deptrees_from_attelo_output,
                        load_attelo_ctrees,
                        load_attelo_dtrees)
from surdeanu import load_surdeanu_ctrees, load_surdeanu_dtrees
from rs22 import load_rs22_ctrees, load_rs22_dtrees  # 2020-10-02 WIP
# 2017-12-12 nuc_clf WIP
from train_nuc_classifier import RightBinaryNuclearityClassifier
from train_rel_relabeller import RelationRelabeller
# end WIP nuc_clf

# 2020-10-01 FIXME silently fails later if the path is wrong
DATA_DIR = os.path.join('..', 'rst-eval-data', 'rst-dt')
# RST corpus
CORPUS_DIR = os.path.join(DATA_DIR, 'RSTtrees-WSJ-main-1.01/')
CD_TRAIN = os.path.join(CORPUS_DIR, 'TRAINING')
CD_TEST = os.path.join(CORPUS_DIR, 'TEST')
DOUBLE_DIR = os.path.join(DATA_DIR, 'RSTtrees-WSJ-double-1.0')
# relation converter (fine- to coarse-grained labels)
RELMAP_FILE = os.path.join('educe', 'rst_dt',
                           'rst_112to18.txt')
REL_CONV_BASE = RstRelationConverter(RELMAP_FILE)
REL_CONV = REL_CONV_BASE.convert_tree
REL_CONV_DTREE = REL_CONV_BASE.convert_dtree


#
# EVALUATIONS
#

# * syntax: pred vs gold
# old-style .edu_input: whole test set
EDUS_FILE = os.path.join('..', 'rst-eval-data',
                         'TMP/syn_gold_coarse',
                         'TEST.relations.sparse.edu_input')

# new style .edu_input: one file per doc in test set
# was: TMP/latest/data..., replaced latest with 2016-09-30T1701 but
# might be wrong (or it might have no consequence here)
EDUS_FILE_PAT = "../rst-eval-data/TMP/2016-09-30T1701/data/TEST/{}.relations.edu-pairs.sparse.edu_input"

# outputs of parsers
EISNER_OUT_SYN_PRED = os.path.join(
    '../rst-eval-data', 'TMP/syn_pred_coarse',  # lbl
    'scratch-current/combined',
    'output.maxent-iheads-global-AD.L-jnt-eisner')

# 2016-09-14 "tree" transform, predicted syntax
EISNER_OUT_TREE_SYN_PRED = os.path.join(
    '../rst-eval-data', 'TMP/2016-09-12T0825',  # lbl
    'scratch-current/combined',
    'output.maxent-iheads-global-AD.L-jnt-eisner')

EISNER_OUT_TREE_SYN_PRED_SU = os.path.join(
    '../rst-eval-data', 'TMP/2016-09-12T0825',  # lbl
    'scratch-current/combined',
    'output.maxent-iheads-global-AD.L-jnt_su-eisner')
# end 2016-09-14


EISNER_OUT_SYN_PRED_SU = os.path.join(
    '../rst-eval-data', 'TMP/latest',  # lbl
    'scratch-current/combined',
    'output.maxent-AD.L-jnt_su-eisner')

EISNER_OUT_SYN_GOLD = os.path.join(
    '../rst-eval-data', 'TMP/syn_gold_coarse',  # lbl
    'scratch-current/combined',
    'output.maxent-iheads-global-AD.L-jnt-eisner')

# replicated parsers
REPLICATION_DIR = '../rst-eval-data/replication'
# output of Joty's parser CODRA
CODRA_OUT_DIR = os.path.join(
    REPLICATION_DIR, 'joty/Doc-level'
)
# output of Ji's parser DPLP
# JI_OUT_DIR = os.path.join('/home/mmorey/melodi/rst/replication/ji_eisenstein', 'DPLP/data/docs/test/')
JI_OUT_DIR = os.path.join(REPLICATION_DIR, 'ji_eisenstein',
                          'official_output/outputs/')
# Feng's parsers
FENG_DIR = os.path.join(REPLICATION_DIR, 'feng_hirst/')
FENG1_OUT_DIR = os.path.join(FENG_DIR, 'phil', 'tmp')
FENG2_OUT_DIR = os.path.join(FENG_DIR, 'gCRF_dist/texts/results/test_batch_gold_seg')
# Li Qi's parser
LI_QI_OUT_DIR = os.path.join(REPLICATION_DIR, 'li_qi/result')
# Hayashi's HILDA
HAYASHI_OUT_DIR = os.path.join(REPLICATION_DIR, 'hayashi/SIGDIAL')
HAYASHI_HILDA_OUT_DIR = os.path.join(HAYASHI_OUT_DIR, 'auto_parse/cons/HILDA')
HAYASHI_MST_OUT_DIR = os.path.join(HAYASHI_OUT_DIR, 'auto_parse/dep/li')
# Braud
BRAUD_COLING_OUT_DIR = os.path.join(REPLICATION_DIR, 'braud/coling16/pred_trees')
BRAUD_EACL_MONO = os.path.join(REPLICATION_DIR, 'braud/eacl16/best-en-mono/test_it8_beam16')
BRAUD_EACL_CROSS_DEV = os.path.join(REPLICATION_DIR, 'braud/eacl16/best-en-cross+dev/test_it10_beam32')
# Surdeanu
SURDEANU_LOG_FILE = os.path.join(REPLICATION_DIR, 'surdeanu/output/log')
# Li Sujian dep parser
# imported, see above
# Wang, Li and Wang at ACL 2017
WLW17_OUT_DIR = os.path.join(REPLICATION_DIR,
                             'wang/rst-dt/RSTtrees-WSJ-main-1.0/TEST')
# (Alexeeva et al. forthcoming)
RS22_LOG_FILE = os.path.join(REPLICATION_DIR, 'rs22/trees_rs22.txt')

# level of detail for parseval
STRINGENT = False
# additional dependency metrics
INCLUDE_LS = False
EVAL_NUC_RANK = True
# hyperparams
NUC_STRATEGY = 'unamb_else_most_frequent'
NUC_CONSTANT = None  # only useful for NUC_STRATEGY='constant'
RNK_STRATEGY = 'sdist-edist-rl'
RNK_PRIORITY_SU = True
# known 'authors'
AUTHORS = [
    'gold',  # RST-main
    'silver',  # RST-double
    'JCN15_1S1S', 'FH14_gSVM', 'FH14_gCRF', 'JE14',
    'LLC16', 'HHN16_HILDA', 'HHN16_MST',
    'BPS16', 'BCS17_mono',
    'BCS17_cross',
    'SHV15_D',
    'WLW17',  # Wang, Li and Wang, ACL17
    'RS22',  # Alexeeva et al forthcoming?
    'li_sujian',
    'ours-chain', 'ours-tree', 'ours-tree-su'
]


def setup_dtree_postprocessor(nary_enc='chain', order='strict',
                              nuc_strategy=NUC_STRATEGY,
                              nuc_constant=NUC_CONSTANT,
                              rnk_strategy=RNK_STRATEGY,
                              rnk_prioritize_same_unit=RNK_PRIORITY_SU):
    """Setup the nuclearity and rank classifiers to flesh out dtrees."""
    # load train section of the RST corpus, fit (currently dummy) classifiers
    # for nuclearity and rank
    reader_train = RstReader(CD_TRAIN)
    corpus_train = reader_train.slurp()
    # gold RST trees
    ctree_true = dict()  # ctrees
    dtree_true = dict()  # dtrees from the original ctrees ('tree' transform)

    for doc_id, ct_true in sorted(corpus_train.items()):
        doc_name = doc_id.doc
        # flavours of ctree
        ct_true = REL_CONV(ct_true)  # map fine to coarse relations
        ctree_true[doc_name] = ct_true
        # flavours of dtree
        dt_true = RstDepTree.from_rst_tree(ct_true, nary_enc=nary_enc)
        dtree_true[doc_name] = dt_true
    # fit classifiers for nuclearity and rank (DIRTY)
    # NB: both are (dummily) fit on weakly ordered dtrees
    X_train = []
    y_nuc_train = []
    y_rnk_train = []
    for doc_name, dt in sorted(dtree_true.items()):
        # print(dt.__dict__)
        # raise ValueError('wip wip nuc_clf')
        X_train.append(dt)
        y_nuc_train.append(dt.nucs)
        y_rnk_train.append(dt.ranks)
    # 2017-12-14 WIP relation relabeller
    if False:
        model_split = 'sent'  # {'none', 'sent'}
        if model_split == 'none':
            dset_folder = os.path.join(
                os.path.expanduser('~'),
                'melodi/rst/irit-rst-dt/TMP/syn_pred_coarse_REL'
            )
            dset_rel_train = os.path.join(dset_folder, 'TRAINING.relations.sparse')
            dset_rel_test = os.path.join(dset_folder, 'TEST.relations.sparse')
            # FIXME read n_features from .vocab
            X_rel_train, y_rel_train, X_rel_test, y_rel_test = load_svmlight_files(
                (dset_rel_train, dset_rel_test),
                n_features=46731,
                zero_based=False
            )
        elif model_split == 'sent':
            # * intra
            dset_folder_intra = os.path.join(
                os.path.expanduser('~'),
                'melodi/rst/irit-rst-dt/TMP/syn_pred_coarse_REL_intrasent'
            )
            dset_train_intra = os.path.join(dset_folder_intra, 'TRAINING.relations.sparse')
            dset_test_intra = os.path.join(dset_folder_intra, 'TEST.relations.sparse')
            # FIXME read n_features from .vocab
            X_rel_train_intra, y_rel_train_intra, X_rel_test_intra, y_rel_test_intra = load_svmlight_files(
                (dset_train_intra, dset_test_intra),
                n_features=46731,
                zero_based=False
            )
            # * inter
            dset_folder_inter = os.path.join(
                os.path.expanduser('~'),
                'melodi/rst/irit-rst-dt/TMP/syn_pred_coarse_REL_intersent'
            )
            dset_train_inter = os.path.join(dset_folder_inter, 'TRAINING.relations.sparse')
            dset_test_inter = os.path.join(dset_folder_inter, 'TEST.relations.sparse')
            # FIXME read n_features from .vocab
            X_rel_train_inter, y_rel_train_inter, X_rel_test_inter, y_rel_test_inter = load_svmlight_files(
                (dset_train_inter, dset_test_inter),
                n_features=46731,
                zero_based=False
            )
            # put together intra and inter
            X_rel_train = (X_rel_train_intra, X_rel_train_inter)
            y_rel_train = (y_rel_train_intra, y_rel_train_inter)
            # TODO the same for {X,y}_rel_test ?
        else:
            raise ValueError("what model_split?")
        # common call
        mul_clf = LogisticRegressionCV(Cs=10,  # defaults to 10,
                                       penalty='l1', solver='liblinear',
                                       n_jobs=3)
        rel_clf = RelationRelabeller(mul_clf=mul_clf, model_split=model_split)
        rel_clf = rel_clf.fit(X_rel_train, y_rel_train)
    else:
        rel_clf = None
    # end 2017-12-14 relations relabeller
    # nuclearity clf
    if True:
        # TODO see whether intra/inter-sentential would be good
        # for the dummy nuc clf
        nuc_clf = DummyNuclearityClassifier(strategy=nuc_strategy,
                                            constant=nuc_constant)
        nuc_clf.fit(X_train, y_nuc_train)
    else:
        # 2017-12-12 WIP nuc_clf
        # shiny new nuc_clf ; still very hacky
        # import the nuclearity TRAIN and TEST sets generated from
        # the svmlight feature vectors (ahem)
        model_split = 'sent'
        #
        if model_split == 'none':
            dset_folder = os.path.join(
                os.path.expanduser('~'),
                'melodi/rst/irit-rst-dt/TMP/syn_pred_coarse_NUC'
            )
            dset_train = os.path.join(dset_folder, 'TRAINING.relations.sparse')
            dset_test = os.path.join(dset_folder, 'TEST.relations.sparse')
            # FIXME read n_features from .vocab
            X_nuc_train, y_nuc_train, X_nuc_test, y_nuc_test = load_svmlight_files(
                (dset_train, dset_test),
                n_features=46731,
                zero_based=False
            )
        elif model_split == 'sent':
            # * intra
            dset_folder_intra = os.path.join(
                os.path.expanduser('~'),
                'melodi/rst/irit-rst-dt/TMP/syn_pred_coarse_NUC_intrasent'
            )
            dset_train_intra = os.path.join(dset_folder_intra, 'TRAINING.relations.sparse')
            dset_test_intra = os.path.join(dset_folder_intra, 'TEST.relations.sparse')
            # FIXME read n_features from .vocab
            X_nuc_train_intra, y_nuc_train_intra, X_nuc_test_intra, y_nuc_test_intra = load_svmlight_files(
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
            # FIXME read n_features from .vocab
            X_nuc_train_inter, y_nuc_train_inter, X_nuc_test_inter, y_nuc_test_inter = load_svmlight_files(
                (dset_train_inter, dset_test_inter),
                n_features=46731,
                zero_based=False
            )
            # put together intra and inter
            X_nuc_train = (X_nuc_train_intra, X_nuc_train_inter)
            y_nuc_train = (y_nuc_train_intra, y_nuc_train_inter)
            # TODO the same for {X,y}_nuc_test ?
        else:
            raise ValueError("what model_split?")
        bin_clf = LogisticRegressionCV(Cs=10,  # defaults to 10
                                       penalty='l1', solver='liblinear',
                                       n_jobs=3)
        nuc_clf = RightBinaryNuclearityClassifier(bin_clf=bin_clf,
                                                  model_split=model_split)
        nuc_clf = nuc_clf.fit(X_nuc_train, y_nuc_train)
        # end WIP nuc_clf
    # rank clf
    rnk_clf = InsideOutAttachmentRanker(
        strategy=rnk_strategy, prioritize_same_unit=rnk_prioritize_same_unit,
        order=order)
    rnk_clf.fit(X_train, y_rnk_train)
    return nuc_clf, rnk_clf, rel_clf


# FIXME:
# * [ ] create summary table with one system per row, one metric per column,
#   keep only the f-score (because for binary trees with manual segmentation
#   precision = recall = f-score).
def main():
    """Run the eval"""
    parser = argparse.ArgumentParser(
        description="Evaluate parsers' output against a given reference")
    # predictions
    parser.add_argument('authors_pred', nargs='+',
                        choices=AUTHORS,
                        help="Author(s) of the predictions")
    # reference
    parser.add_argument('--author_true', default='gold',
                        choices=AUTHORS + ['each'],  # NEW generate sim matrix
                        help="Author of the reference")
    # * ctree/dtree eval: the value of binarize_true determines the values
    # of nary_enc_true and order_true (the latter is yet unused)
    parser.add_argument('--binarize_true', default='none',
                        choices=['none', 'right', 'right_mixed', 'left'],
                        help=("Binarization method for the reference ctree"
                              "in the eval ; defaults to 'none' for no "
                              "binarization"))
    parser.add_argument('--simple_rsttree', action='store_true',
                        help="Binarize ctree and move relations up")
    # * non-standard evals
    parser.add_argument('--per_doc', action='store_true',
                        help="Doc-averaged scores (cf. Ji's eval)")
    parser.add_argument('--eval_li_dep', action='store_true',
                        help=("Evaluate as in the dep parser of Li et al. "
                              "2014: all relations are NS, spiders map to "
                              "left-heavy branching, three trivial spans "))
    # * display options
    parser.add_argument('--digits', type=int, default=3,
                        help='Precision (number of digits) of scores')
    parser.add_argument('--percent', action='store_true',
                        help='Scores are displayed as percentages (ex: 57.9)')
    parser.add_argument('--detailed', type=int, default=0,
                        help='Level of detail for evaluations')
    parser.add_argument('--out_fmt', default='text',
                        choices=['text', 'latex'],
                        help='Output format')
    #
    args = parser.parse_args()
    author_true = args.author_true
    authors_pred = args.authors_pred
    binarize_true = args.binarize_true
    simple_rsttree = args.simple_rsttree
    # display
    digits = args.digits
    percent = args.percent
    if percent:
        if digits < 3:
            raise ValueError('--percent requires --digits >= 3')
    # level of detail for evals
    detailed = args.detailed
    out_fmt = args.out_fmt

    # "per_doc = True" computes p, r, f as in DPLP: compute scores per doc
    # then average over docs
    # it should be False, except for comparison with the DPLP paper
    per_doc = args.per_doc
    # "eval_li_dep = True" replaces the original nuclearity and order with
    # heuristically determined values for _pred but also _true, and adds
    # three trivial spans
    eval_li_dep = args.eval_li_dep
    # nary_enc_true is used ; order_true currently is not (implicit in
    # nary_enc_true)
    if binarize_true in ('right', 'right_mixed'):
        nary_enc_true = 'chain'
        order_true = 'strict'
    elif binarize_true == 'left':
        nary_enc_true = 'tree'
        order_true = 'strict'
    else:  # 'none' for no binarization of the reference tree
        nary_enc_true = 'tree'
        order_true = 'weak'

    # 0. setup the postprocessors to flesh out unordered dtrees into ordered
    # ones with nuclearity
    # * tie the order with the encoding for n-ary nodes
    nuc_clf_chain, rnk_clf_chain, rel_clf_chain = setup_dtree_postprocessor(
        nary_enc='chain', order='strict')
    # FIXME explicit differenciation between (heuristic) classifiers for
    # the "chain" vs "tree" transforms (2 parameters: nary_enc, order) ;
    # nuc_clf, rnk_clf, rel_clf might contain implicit assumptions
    # tied to the "chain" transform, might not be optimal for "tree"
    nuc_clf_tree, rnk_clf_tree, rel_clf_tree = setup_dtree_postprocessor(
        nary_enc='tree', order='weak')

    # the eval compares parses for the test section of the RST corpus
    reader_test = RstReader(CD_TEST)
    corpus_test = reader_test.slurp()
    doc_edus_test = {k.doc: ct_true.leaves() for k, ct_true
                     in corpus_test.items()}

    # reference: author_true can be any of the authors_pred (defaults to gold)
    ctree_true = dict()  # ctrees
    dtree_true = dict()  # dtrees from the original ctrees ('tree' transform)
    for doc_id, ct_true in sorted(corpus_test.items()):
        doc_name = doc_id.doc
        # original reference ctree, with coarse labels
        ct_true = REL_CONV(ct_true)  # map fine to coarse relations
        if binarize_true != "none":
            # binarize ctree if required
            ct_true = _binarize(ct_true, branching=binarize_true)
        ctree_true[doc_name] = ct_true
        # corresponding dtree
        dt_true = RstDepTree.from_rst_tree(ct_true, nary_enc=nary_enc_true)
        dtree_true[doc_name] = dt_true
    # sorted doc_names, because braud_eacl put all predictions in one file
    sorted_doc_names = sorted(dtree_true.keys())

    c_preds = []  # predictions: [(parser_name, dict(doc_name, ct_pred))]
    d_preds = []  # predictions: [(parser_name, dict(doc_name, dt_pred))]

    for author_pred in authors_pred:
        # braud coling 2016
        if author_pred == 'BPS16':
            ctree_pred = load_braud_coling_ctrees(BRAUD_COLING_OUT_DIR,
                                                  REL_CONV)
            c_preds.append(
                ('BPS16', ctree_pred)
            )
            d_preds.append(
                ('BPS16', load_braud_coling_dtrees(
                    BRAUD_COLING_OUT_DIR, REL_CONV, nary_enc='chain',
                    ctree_pred=ctree_pred))
            )
        # braud eacl 2017 - mono
        if author_pred == 'BCS17_mono':
            ctree_pred = load_braud_eacl_ctrees(BRAUD_EACL_MONO, REL_CONV,
                                                sorted_doc_names)
            c_preds.append(
                ('BCS17_mono', ctree_pred)
            )
            d_preds.append(
                ('BCS17_mono', load_braud_eacl_dtrees(
                    BRAUD_EACL_MONO, REL_CONV, sorted_doc_names,
                    nary_enc='chain', ctree_pred=ctree_pred))
            )
        # braud eacl 2017 - cross+dev
        if author_pred == 'BCS17_cross':
            ctree_pred = load_braud_eacl_ctrees(BRAUD_EACL_CROSS_DEV,
                                                REL_CONV, sorted_doc_names)
            c_preds.append(
                ('BCS17_cross', ctree_pred)
            )
            d_preds.append(
                ('BCS17_cross', load_braud_eacl_dtrees(
                    BRAUD_EACL_CROSS_DEV, REL_CONV, sorted_doc_names,
                    nary_enc='chain', ctree_pred=ctree_pred))
            )

        if author_pred == 'HHN16_HILDA':
            ctree_pred = load_hayashi_hilda_ctrees(HAYASHI_HILDA_OUT_DIR,
                                                   REL_CONV)
            c_preds.append(
                ('HHN16_HILDA', ctree_pred)
            )
            d_preds.append(
                ('HHN16_HILDA', load_hayashi_hilda_dtrees(
                    HAYASHI_HILDA_OUT_DIR, REL_CONV, nary_enc='chain',
                    ctree_pred=ctree_pred))
            )

        if author_pred == 'HHN16_MST':
            # paper: {nary_enc_pred='chain', order='strict'}
            dtree_pred = load_hayashi_dep_dtrees(
                HAYASHI_MST_OUT_DIR, REL_CONV_DTREE, doc_edus_test,
                EDUS_FILE_PAT, nuc_clf_chain, rnk_clf_chain)
            c_preds.append(
                ('HHN16_MST', load_hayashi_dep_ctrees(
                    HAYASHI_MST_OUT_DIR, REL_CONV_DTREE, doc_edus_test,
                    EDUS_FILE_PAT, nuc_clf_chain, rnk_clf_chain,
                    dtree_pred=dtree_pred))
            )
            d_preds.append(
                ('HHN16_MST', dtree_pred)
            )

        if author_pred == 'LLC16':
            ctree_pred = load_li_qi_ctrees(LI_QI_OUT_DIR, REL_CONV)
            c_preds.append(
                ('LLC16', ctree_pred)
            )
            d_preds.append(
                ('LLC16', load_li_qi_dtrees(LI_QI_OUT_DIR, REL_CONV,
                                            nary_enc='chain',
                                            ctree_pred=ctree_pred))
            )

        if author_pred == 'li_sujian':
            # FIXME load d-trees once, pass dtree_pred to the c-loader ;
            # paper says 'chain' transform, but it might be worth
            # checking
            c_preds.append(
                ('li_sujian', load_li_sujian_dep_ctrees(
                    LI_SUJIAN_OUT_FILE, REL_CONV_DTREE, EDUS_FILE_PAT,
                    nuc_clf_chain, rnk_clf_chain))
            )
            d_preds.append(
                ('li_sujian', load_li_sujian_dep_dtrees(
                    LI_SUJIAN_OUT_FILE, REL_CONV_DTREE, EDUS_FILE_PAT,
                    nuc_clf_chain, rnk_clf_chain))
            )

        if author_pred == 'FH14_gSVM':
            # FIXME load c-trees once, pass ctree_pred to the d-loader
            c_preds.append(
                ('FH14_gSVM', load_feng_ctrees(FENG1_OUT_DIR, REL_CONV))
            )
            d_preds.append(
                ('FH14_gSVM', load_feng_dtrees(FENG1_OUT_DIR, REL_CONV,
                                               nary_enc='chain'))
            )

        if author_pred == 'FH14_gCRF':
            ctree_pred = load_gcrf_ctrees(FENG2_OUT_DIR, REL_CONV)
            c_preds.append(
                ('FH14_gCRF', ctree_pred)
            )
            d_preds.append(
                ('FH14_gCRF', load_gcrf_dtrees(FENG2_OUT_DIR, REL_CONV,
                                               nary_enc='chain',
                                               ctree_pred=ctree_pred))
            )

        if author_pred == 'JCN15_1S1S':
            # CODRA outputs RST ctrees ; eval_codra_output maps them to RST dtrees
            ctree_pred = load_codra_ctrees(CODRA_OUT_DIR, REL_CONV)
            c_preds.append(
                ('JCN15_1S1S', ctree_pred)
            )
            d_preds.append(
                ('JCN15_1S1S', load_codra_dtrees(CODRA_OUT_DIR, REL_CONV,
                                                 nary_enc='chain',
                                                 ctree_pred=ctree_pred))
            )
            # joty-{chain,tree} would be the same except nary_enc='tree' ;
            # the nary_enc does not matter because codra outputs binary ctrees,
            # hence both encodings result in (the same) strictly ordered dtrees

        if author_pred == 'JE14':
            # DPLP outputs RST ctrees in the form of lists of spans;
            # load_ji_dtrees maps them to RST dtrees
            ctree_pred = load_ji_ctrees(JI_OUT_DIR, REL_CONV, doc_edus_test)
            c_preds.append(
                ('JE14', ctree_pred)
            )
            d_preds.append(
                ('JE14', load_ji_dtrees(JI_OUT_DIR, REL_CONV, doc_edus_test,
                                        nary_enc='chain',
                                        ctree_pred=ctree_pred))
            )
            # ji-{chain,tree} would be the same except nary_enc='tree' ;
            # the nary_enc does not matter because DPLP outputs binary ctrees,
            # hence both encodings result in (the same) strictly ordered dtrees

        if author_pred == 'WLW17':
            # WLW17 outputs RST ctrees in the form of lists of spans, just
            # like JE14 ;
            # load_ji_dtrees maps them to RST dtrees
            c_preds.append(
                ('WLW17', load_ji_ctrees(
                    WLW17_OUT_DIR, REL_CONV))
            )
            d_preds.append(
                ('WLW17', load_ji_dtrees(
                    WLW17_OUT_DIR, REL_CONV, nary_enc='chain'))
            )
            # the nary_enc does not matter because WLW17 outputs binary ctrees,
            # hence both encodings result in (the same) strictly ordered dtrees

        if author_pred == 'SHV15_D':
            ctree_pred = load_surdeanu_ctrees(SURDEANU_LOG_FILE, REL_CONV)
            c_preds.append(
                ('SHV15_D', ctree_pred)
            )
            d_preds.append(
                ('SHV15_D', load_surdeanu_dtrees(
                    SURDEANU_LOG_FILE, REL_CONV, nary_enc='chain',
                    ctree_pred=ctree_pred))
            )
        # WIP 2020-10-02
        if author_pred == 'RS22':
            ctree_pred = load_rs22_ctrees(RS22_LOG_FILE, REL_CONV)
            c_preds.append(
                ('RS22', ctree_pred)
            )
            d_preds.append(
                ('RS22', load_rs22_dtrees(
                    RS22_LOG_FILE, REL_CONV, nary_enc='chain',
                    ctree_pred=ctree_pred))
            )

        if author_pred == 'ours-chain':
            # Eisner, predicted syntax, chain
            dtree_pred = load_attelo_dtrees(
                EISNER_OUT_SYN_PRED, EDUS_FILE,
                rel_clf_chain, nuc_clf_chain, rnk_clf_chain,
                doc_edus=doc_edus_test)
            c_preds.append(
                ('ours-chain', load_attelo_ctrees(
                    EISNER_OUT_SYN_PRED, EDUS_FILE,
                    rel_clf_chain, nuc_clf_chain, rnk_clf_chain,
                    doc_edus=doc_edus_test,
                    dtree_pred=dtree_pred))
            )
            d_preds.append(
                ('ours-chain', dtree_pred)
            )

        if author_pred == 'ours-tree':
            # Eisner, predicted syntax, tree + same-unit
            dtree_pred = load_attelo_dtrees(
                EISNER_OUT_TREE_SYN_PRED, EDUS_FILE,
                rel_clf_tree, nuc_clf_tree, rnk_clf_tree,
                doc_edus=doc_edus_test)
            c_preds.append(
                ('ours-tree', load_attelo_ctrees(
                    EISNER_OUT_TREE_SYN_PRED, EDUS_FILE,
                    rel_clf_tree, nuc_clf_tree, rnk_clf_tree,
                    doc_edus=doc_edus_test,
                    dtree_pred=dtree_pred))
            )
            d_preds.append(
                ('ours-tree', dtree_pred)
            )
        if author_pred == 'ours-tree-su':
            # Eisner, predicted syntax, tree + same-unit
            dtree_pred = load_attelo_dtrees(
                EISNER_OUT_TREE_SYN_PRED_SU, EDUS_FILE, 
                rel_clf_tree, nuc_clf_tree, rnk_clf_tree,
                doc_edus=doc_edus_test)
            c_preds.append(
                ('ours-tree-su', load_attelo_ctrees(
                    EISNER_OUT_TREE_SYN_PRED_SU, EDUS_FILE,
                    rel_clf_tree, nuc_clf_tree, rnk_clf_tree,
                    doc_edus=doc_edus_test,
                    dtree_pred=dtree_pred))
            )
            d_preds.append(
                ('ours-tree-su', dtree_pred)
            )
        # 2017-05-17 enable "gold" as parser, should give perfect scores
        if author_pred == 'gold':
            c_preds.append(
                ('gold', ctree_true)
            )
            d_preds.append(
                ('gold', dtree_true)
            )

        if False:  # FIXME repair (or forget) these
            print('Eisner, predicted syntax + same-unit')
            load_deptrees_from_attelo_output(
                ctree_true, dtree_true,
                EISNER_OUT_SYN_PRED_SU, EDUS_FILE,
                rel_clf_chain, nuc_clf_chain, rnk_clf_chain)
            print('======================')

            print('Eisner, gold syntax')
            load_deptrees_from_attelo_output(
                ctree_true, dtree_true,
                EISNER_OUT_SYN_GOLD, EDUS_FILE,
                rel_clf_chain, nuc_clf_chain, rnk_clf_chain)
            print('======================')

    # dependency eval
    dep_metrics = ["U"]
    if EVAL_NUC_RANK:
        dep_metrics += ['O', 'N', 'O+N']
    dep_metrics += ["R"]
    if INCLUDE_LS:
        dep_metrics += ["tag_R"]
    if EVAL_NUC_RANK:
        dep_metrics += ["R+N", "F"]  # 2017-11-29 disable "R+O"

    # _true
    doc_names = sorted(dtree_true.keys())
    labelset_true = set(itertools.chain.from_iterable(
        x.labels for x in dtree_true.values()))
    labelset_true.add("span")  # RST-DT v.1.0 has an error in wsj_1189 7-9
    # 2017-05-17 any author can be used as reference
    if author_true != 'each':
        parser_true = author_true
        print(dep_compact_report(parser_true, d_preds, dep_metrics,
                                 doc_names, labelset_true,
                                 digits=digits,
                                 percent=percent,
                                 out_format=out_fmt))
    else:
        print(dep_similarity(d_preds, doc_names, labelset_true,
                             dep_metric='U', digits=digits, percent=percent,
                             out_format=out_fmt))
        # raise ValueError("Sim matrix on dependencies not implemented yet")

    # constituency eval
    ctree_type = 'SimpleRST' if simple_rsttree else 'RST'

    doc_names = sorted(ctree_true.keys())

    if False:  # back when 'gold' was the only possible ref
        ctree_true_list = [ctree_true[doc_name] for doc_name in doc_names]
        if simple_rsttree:
            ctree_true_list = [SimpleRSTTree.from_rst_tree(x)
                               for x in ctree_true_list]
        # WIP print SimpleRSTTrees
        if not os.path.exists('gold'):
            os.makedirs('gold')
        for doc_name, ct in zip(doc_names, ctree_true_list):
            with codecs.open('gold/' + ct.origin.doc, mode='w',
                             encoding='utf-8') as f:
                print(ct, file=f)

    # sort the predictions of each parser, so they match the order of
    # documents and reference trees in _true
    ctree_preds = [(parser_name,
                    [ctree_pred[doc_name] for doc_name in doc_names])
                   for parser_name, ctree_pred in c_preds]
    if simple_rsttree:
        ctree_preds = [(parser_name,
                        [SimpleRSTTree.from_rst_tree(x)
                         for x in ctree_pred_list])
                       for parser_name, ctree_pred_list in ctree_preds]

    # 2017-05-17 allow any parser to be ref
    # generate report
    if detailed == 0:
        # 2017-05-17 WIP similarity matrix: author_true='each': restrict
        # to the S metric only, so as to display a sim. matrix
        if author_true == 'each':
            metric_type = 'S'
            print(rst_parseval_similarity(ctree_preds,
                                          ctree_type=ctree_type,
                                          metric_type=metric_type,
                                          digits=digits,
                                          percent=percent,
                                          print_support=False,
                                          per_doc=per_doc,
                                          add_trivial_spans=eval_li_dep,
                                          stringent=STRINGENT,
                                          out_format=out_fmt))
        else:
            metric_types = [
                'S', 'N', 'R', 'F',
                'S+H', 'N+H', 'R+H', 'F+H',
                # 'S+K', 'N+K', 'R+K', 'F+K',
                # 'S+HH', 'N+HH', 'R+HH', 'F+HH',
                # 'S+K+HH', 'N+K+HH', 'R+K+HH', 'F+K+HH',
                # 'S+H+K+HH', 'N+H+K+HH', 'R+H+K+HH', 'F+H+K+HH',
            ]
            # compact report, f1-scores only
            print(rst_parseval_compact_report(author_true, ctree_preds,
                                              ctree_type=ctree_type,
                                              metric_types=metric_types,
                                              digits=digits,
                                              percent=percent,
                                              print_support=False,
                                              per_doc=per_doc,
                                              add_trivial_spans=eval_li_dep,
                                              stringent=STRINGENT))  # 2020 ,
                                              # out_format=out_fmt))  # 2020
    else:
        parsers_true = [author_true] if author_true != 'each' else authors_pred
        for parser_true in parsers_true:
            # standard reports: 1 table per parser, 1 line per metric,
            # cols = [p, r, f1, support_true, support_pred]
            # FIXME
            ctree_true_list = []
            for parser_name, ctree_pred in c_preds:
                if parser_name == parser_true:
                    ctree_true_list = [ctree_pred[doc_name] for doc_name in doc_names]
                    break
            # end FIXME

            for parser_name, ctree_pred_list in ctree_preds:
                # WIP print SimpleRSTTrees
                if not os.path.exists(parser_name):
                    os.makedirs(parser_name)
                for doc_name, ct in zip(doc_names, ctree_pred_list):
                    with codecs.open(parser_name + '/' + doc_name, mode='w',
                                     encoding='utf-8') as f:
                        print(ct, file=f)

                # compute and print PARSEVAL scores
                print(parser_name)
                # metric_types=None includes the variants with head:
                # S+H, N+H, R+H, F+H
                print(rst_parseval_report(ctree_true_list, ctree_pred_list,
                                          ctree_type=ctree_type,
                                          metric_types=None,
                                          digits=digits,
                                          percent=percent,
                                          per_doc=per_doc,
                                          add_trivial_spans=eval_li_dep,
                                          stringent=STRINGENT))
                # detailed report on R
                if detailed >= 2:
                    print(rst_parseval_detailed_report(
                        ctree_true_list, ctree_pred_list, ctree_type=ctree_type,
                        metric_type='R'))
                # end FIXME

    # 2017-04-11 compute agreement between human annotators, on DOUBLE
    if 'silver' in authors_pred:
        # 'silver' can be meaningfully compared to 'gold' only (too few
        # documents otherwise)
        if author_true != 'gold':
            raise NotImplementedError('Not yet')

        # read the annotation we'll consider as "silver"
        reader_dbl = RstReader(DOUBLE_DIR)
        corpus_dbl_pred = {k.doc: v for k, v in reader_dbl.slurp().items()}
        docs_dbl = sorted(k for k in corpus_dbl_pred.keys())
        # collect the "true" annotation for the docs in double, from train
        # and test
        # (test has already been read at the beginning of this script)
        corpus_test_dbl = {k.doc: v for k, v in corpus_test.items()
                           if k.doc in docs_dbl}
        # read the docs from train that are in double
        reader_train = RstReader(CD_TRAIN)
        corpus_train = reader_train.slurp()
        corpus_train_dbl = {k.doc: v for k, v in corpus_train.items()
                            if k.doc in docs_dbl}
        # assemble the "true" version of the double subset
        corpus_dbl_true = dict(corpus_test_dbl.items() +
                               corpus_train_dbl.items())
        assert (sorted(corpus_dbl_true.keys()) ==
                sorted(corpus_dbl_pred.keys()))
        # extra check?
        if False:
            for doc_name in docs_dbl:
                leaf_spans_true = [x.text_span() for x
                                   in corpus_dbl_true[doc_name].leaves()]
                leaf_spans_pred = [x.text_span() for x
                                   in corpus_dbl_pred[doc_name].leaves()]
                if (leaf_spans_true != leaf_spans_pred):
                    print(doc_name, 'EEEE')
                    print('true - pred',
                          set(leaf_spans_true) - set(leaf_spans_pred))
                    print('pred - true',
                          set(leaf_spans_pred) - set(leaf_spans_true))
                else:
                    print(doc_name, 'ok')
        # end extra check

        # 48 docs in train,
        # 5 docs in test: ['wsj_0627.out', 'wsj_0684.out', 'wsj_1129.out',
        # 'wsj_1365.out', 'wsj_1387.out']
        # create parallel lists of ctrees for _true and _pred, mapped to
        # coarse rels and binarized
        # _pred:
        # * ctree
        ctree_dbl_pred = [corpus_dbl_pred[doc_name] for doc_name in docs_dbl]
        ctree_dbl_pred = [REL_CONV(x) for x in ctree_dbl_pred]
        if binarize_true != 'none':  # maybe not?
            ctree_dbl_pred = [_binarize(x, branching=binarize_true)
                              for x in ctree_dbl_pred]
        # * dtree (as dict from doc_name to dtree !?)
        dtree_dbl_pred = {doc_name: RstDepTree.from_rst_tree(
            ct, nary_enc=nary_enc_true)
                          for doc_name, ct in zip(docs_dbl, ctree_dbl_pred)}
        # * simple_rsttree (?)
        if simple_rsttree:
            ctree_dbl_pred = [SimpleRSTTree.from_rst_tree(x)
                              for x in ctree_dbl_pred]
        # _true:
        ctree_dbl_true = [corpus_dbl_true[doc_name] for doc_name in docs_dbl]
        ctree_dbl_true = [REL_CONV(x) for x in ctree_dbl_true]
        if binarize_true != 'none':
            ctree_dbl_true = [_binarize(x, branching=binarize_true)
                              for x in ctree_dbl_true]
        # * dtree (as dict from doc_name to dtree !?)
        dtree_dbl_true = {doc_name: RstDepTree.from_rst_tree(
            ct, nary_enc=nary_enc_true)
                          for doc_name, ct in zip(docs_dbl, ctree_dbl_true)}
        if simple_rsttree:
            ctree_dbl_true = [SimpleRSTTree.from_rst_tree(x)
                              for x in ctree_dbl_true]
        # generate report
        # * ctree eval
        ctree_dbl_preds = [('silver', ctree_dbl_pred),
                           ('gold', ctree_dbl_true)]
        print(rst_parseval_compact_report(author_true, ctree_dbl_preds,
                                          ctree_type=ctree_type,
                                          span_type='chars',
                                          metric_types=['S', 'N', 'R', 'F'],
                                          digits=digits,
                                          percent=percent,
                                          per_doc=per_doc,
                                          add_trivial_spans=eval_li_dep,
                                          stringent=STRINGENT))
        # * dtree eval
        if False:
            # TODO cope with differences in segmentation
            dtree_dbl_preds = [('silver', dtree_dbl_pred),
                               ('gold', dtree_dbl_true)]
            print(dep_compact_report(author_true, dtree_dbl_preds,
                                     dep_metrics, docs_dbl,
                                     labelset_true,
                                     digits=digits,
                                     percent=percent))
    # end 2017-04-11 agreement between human annotators


if __name__ == '__main__':
    main()
