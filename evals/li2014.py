"""Evaluation procedure used in the parser of (Li et al. 2014).

This is a reimplementation of this evaluation procedure.
"""

from educe.rst_dt.metrics.rst_parseval import (rst_parseval_report,
                                               rst_parseval_detailed_report)



# FIXME legacy code brutally dumped here, broken
def twisted_eval_li2014(data_true, data_pred):
    """Run Parseval on transformed gold trees, as in (Li et al., 2014).

    This applies a deterministic transform to the gold constituency tree
    that basically re-orders attachments of a head EDU.
    """
    # 1. ctrees_true -> dtrees_true or dtrees_twis (if the procedure
    # is fishy)
    # 2. dtrees_[true|twis] -> ctrees_twis
    # RESUME HERE
    # hint: ctrees_twis contain only NS nuclearity (...)

    # TODO check exact conformance with the code of their parser:
    # how rank and nuclearity are determined
    data_true['rst_ctrees'] = []
    for dt_true in data_true['rst_dtrees']:
        # FIXME map EDUs to sentences
        dt_true.sent_idx = [edu_id2sent_idx[e.identifier()]
                            for e in dt_true.edus]
        # TODO check that 'lllrrr' effectively corresponds to the strategy
        # they apply
        chn_bin_srtree_true = deptree_to_simple_rst_tree(
            dt_true, MULTINUC_LBLS, strategy='lllrrr')
        chn_bin_rtree_true = SimpleRSTTree.to_binary_rst_tree(
            chn_bin_srtree_true)
        bin_rtree_true = chn_bin_rtree_true
        data_true['rst_ctrees'].append(bin_rtree_true)
# end FIXME


# FIXME currently broken, need to declare and fit classifiers for nuc and rank
# (nuc_classifier and rank_classifier)
# TODO move to ?
def eval_distortion_gold(corpus, nuc_strategy, rank_strategy,
                         prioritize_same_unit):
    """Load an RstDepTree from the output of attelo.

    Parameters
    ----------
    corpus: string
        Path to the gold corpus to be evaluated
    nuc_strategy: string
        Strategy to predict nuclearity
    rank_strategy: string
        Strategy to predict attachment ranking
    """
    # print parameters
    print('corpus: {}\tnuc_strategy: {}\trank_strategy: {}'.format(
        corpus, nuc_strategy, rank_strategy))

    gold_orig = dict()
    gold_twis = dict()

    # FIXME: find ways to read the right (not necessarily TEST) section
    # and only the required documents
    rst_reader = RstReader(corpus)
    rst_corpus = rst_reader.slurp()
    for doc_id, rtree_ref in sorted(rst_corpus.items()):
        doc_name = doc_id.doc

        # original gold
        # convert labels to coarse
        coarse_rtree_ref = REL_CONV(rtree_ref)
        # convert to binary tree
        bin_rtree_ref = _binarize(coarse_rtree_ref)
        gold_orig[doc_name] = bin_rtree_ref

        # distorted gold: forget nuclearity and order of attachment
        # convert to RstDepTree via SimpleRSTTree
        bin_srtree_ref = SimpleRSTTree.from_rst_tree(coarse_rtree_ref)
        dt_ref = RstDepTree.from_simple_rst_tree(bin_srtree_ref)
        # FIXME replace gold nuclearity and rank with predicted ones,
        # using the given heuristics
        # dt_ref.nucs = nuc_classifier.predict([dt_ref])[0]
        # dt_ref.ranks = rank_classifier.predict([dt_ref])[0]
        # end FIXME
        # regenerate a binary RST tree
        chn_bin_srtree_ref = deptree_to_simple_rst_tree(dt_ref)
        chn_bin_rtree_ref = SimpleRSTTree.to_binary_rst_tree(
            chn_bin_srtree_ref)
        gold_twis[doc_name] = chn_bin_rtree_ref

    print(rst_parseval_report(gold_orig, gold_twis,
                              metric_types=[x[0] for x in LBL_FNS],
                              digits=4))
    # detailed report on S+N+R
    print(rst_parseval_detailed_report(ctree_true, ctree_pred,
                                       metric_type='S+R'))


def comparative_distortion_on_gold():
    """Evaluate the impact of forgetting nuclearity and rank in the gold.

    Quantify the distortion and loss when forgetting nuclearity and rank
    in the gold and replacing them with deterministically-determined
    values.

    Possible configurations are the cross-product of strategies to
    heuristically determine rank and nuclearity.
    """
    gold_corpus = CD_TRAIN  # CD_TEST
    nuc_strats = ["most_frequent_by_rel",
                  "unamb_else_most_frequent"]
    rank_strats = ['lllrrr',
                   'rrrlll',
                   'lrlrlr',
                   'rlrlrl']
    prioritize_same_units = [True, False]
    for nuc_strat in nuc_strats:
        for rank_strat in rank_strats:
            eval_distortion_gold(gold_corpus, nuc_strat, rank_strat)
