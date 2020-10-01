"""Evaluate our parsers.

"""

from __future__ import print_function

from collections import defaultdict

import numpy as np

from educe.annotation import Span as EduceSpan
from educe.rst_dt.annotation import (EDU as EduceEDU, SimpleRSTTree)
from educe.rst_dt.corpus import mk_key
from educe.rst_dt.dep2con import (deptree_to_simple_rst_tree,
                                  deptree_to_rst_tree)
from educe.rst_dt.deptree import RstDepTree, RstDtException
from educe.rst_dt.document_plus import align_edus_with_paragraphs
#
from attelo.io import load_edus
from attelo.table import UNRELATED  # for load_attelo_output_file


# move to attelo.datasets.attelo_out_format
def load_attelo_output_file(output_file):
    """Load edges from an attelo output file.

    An attelo output file typically contains edges from several
    documents. This function indexes edges by the name of their
    document.

    Parameters
    ----------
    output_file: string
        Path to the attelo output file

    Returns
    -------
    edges_pred: dict(string, [(string, string, string)])
        Predicted edges for each document, indexed by doc name

    Notes
    -----
    See `attelo.io.load_predictions` that is almost equivalent to this
    function. They are expected to converge some day into a better,
    obvious in retrospect, function.
    """
    edges_pred = defaultdict(list)
    with open(output_file) as f:
        for line in f:
            src_id, tgt_id, lbl = line.strip().split('\t')
            if lbl != UNRELATED:
                # dirty hack: get doc name from EDU id
                # e.g. (EDU id = wsj_0601_1) => (doc id = wsj_0601)
                doc_name = tgt_id.rsplit('_', 1)[0]
                edges_pred[doc_name].append((src_id, tgt_id, lbl))

    return edges_pred


def load_attelo_dtrees(output_file, edus_file, rel_clf, nuc_clf, rnk_clf,
                       doc_edus=None):
    """Load RST dtrees from attelo output files.

    Parameters
    ----------
    output_file: string
        Path to the file that contains attelo's output
    edus_file: string
        Path to the file that describes EDUs.
    doc_edus : dict(str, list(EDU)), optional
        Mapping from doc_name to the list of its EDUs (read from the
        corpus). If None, each EDU is re-created using information in
        the `.edu_input` file, otherwise EDUs are created but their text
        is taken from `doc_edus`.
        FIXME avoid creating "new" EDUs altogether if `doc_edus` is not
        None.

    Returns
    -------
    TODO
    """
    dtree_pred = dict()  # predicted dtrees
    # * setup...
    # load EDUs as they are known to attelo (sigh): rebuild educe EDUs
    # from their attelo description and group them by doc_name
    educe_edus = defaultdict(list)
    edu2sent_idx = defaultdict(dict)
    gid2num = dict()
    att_edus = load_edus(edus_file)
    for att_edu in att_edus:
        # doc name
        doc_name = att_edu.grouping
        # EDU info
        edu_num = int(att_edu.id.rsplit('_', 1)[1])
        edu_span = EduceSpan(att_edu.start, att_edu.end)
        if doc_edus is not None:
            edu_text = doc_edus[doc_name][edu_num - 1].raw_text
        else:
            edu_text = att_edu.text
        educe_edus[doc_name].append(EduceEDU(edu_num, edu_span, edu_text))
        # map global id of EDU to num of EDU inside doc
        gid2num[att_edu.id] = edu_num
        # map EDU to sentence
        sent_idx = int(att_edu.subgrouping.split('_sent')[1])
        edu2sent_idx[doc_name][edu_num] = sent_idx
    # sort EDUs by num
    educe_edus = {doc_name: sorted(edus, key=lambda e: e.num)
                  for doc_name, edus in educe_edus.items()}
    # rebuild educe-style edu2sent ; prepend 0 for the fake root
    doc_name2edu2sent = {doc_name: ([0] +
                                    [edu2sent_idx[doc_name][e.num]
                                     for e in doc_educe_edus])
                         for doc_name, doc_educe_edus in educe_edus.items()}

    # load predicted edges, on these EDUs, into RstDepTrees
    edges_pred = load_attelo_output_file(output_file)
    for doc_name, es_pred in sorted(edges_pred.items()):
        # get educe EDUs
        doc_educe_edus = educe_edus[doc_name]
        # create pred dtree
        dt_pred = RstDepTree(doc_educe_edus)
        for src_id, tgt_id, lbl in es_pred:
            if src_id == 'ROOT':
                if lbl == 'ROOT':
                    dt_pred.set_root(gid2num[tgt_id])
                else:
                    raise ValueError('Weird root label: {}'.format(lbl))
            else:
                dt_pred.add_dependency(gid2num[src_id], gid2num[tgt_id], lbl)
        dt_pred.origin = mk_key(doc_name)
        # 2017-12-14 relabel relations
        if rel_clf is not None:
            dt_pred.labels = rel_clf.predict([dt_pred])[0]
        # end relabel relations
        # add nuclearity: heuristic baseline WIP or true classifier
        dt_pred.nucs = nuc_clf.predict([dt_pred])[0]
        # add rank: heuristic baseline, needs edu2sent
        edu2sent = doc_name2edu2sent[doc_name]
        dt_pred.sent_idx = edu2sent  # DIRTY
        dt_pred.ranks = rnk_clf.predict([dt_pred])[0]
        # store
        dtree_pred[doc_name] = dt_pred

    return dtree_pred


def load_attelo_ctrees(output_file, edus_file, rel_clf, nuc_clf, rnk_clf,
                       doc_edus=None, dtree_pred=None):
    """Load RST ctrees from attelo output files.

    Parameters
    ----------
    output_file: string
        Path to the file that contains attelo's output
    edus_file: string
        Path to the file that describes EDUs.
    nuc_clf: NuclearityClassifier
        Classifier to predict nuclearity
    rnk_clf: RankClassifier
        Classifier to predict attachment ranking
    doc_edus : dict(str, list(EDU)), optional
        Mapping from doc_name to the list of its EDUs (read from the
        corpus). If None, each EDU is re-created using information in
        the `.edu_input` file, otherwise EDUs are created but their text
        is taken from `doc_edus`.
        FIXME avoid creating "new" EDUs altogether if `doc_edus` is not
        None.
    dtree_pred : dict(str, RstDepTree), optional
        RST d-trees, indexed by doc_name. If d-trees are provided this
        way, `out_dir` is ignored.

    Returns
    -------
    TODO
    """
    if dtree_pred is None:
        # load RST dtrees, with heuristics for nuc and rank
        dtree_pred = load_attelo_dtrees(output_file, edus_file,
                                        rel_clf, nuc_clf, rnk_clf,
                                        doc_edus=doc_edus)
    # convert to RST ctrees
    ctree_pred = dict()
    for doc_name, dt_pred in dtree_pred.items():
        try:
            rtree_pred = deptree_to_rst_tree(dt_pred)
            ctree_pred[doc_name] = rtree_pred
        except RstDtException as rst_e:
            print(rst_e)
            if False:
                print('\n'.join('{}: {}'.format(edu.text_span(), edu)
                                for edu in educe_edus[doc_name]))
            # raise

    return ctree_pred


def load_deptrees_from_attelo_output(ctree_true, dtree_true,
                                     output_file, edus_file,
                                     nuc_clf, rnk_clf):
    """Load an RstDepTree from the output of attelo.

    Parameters
    ----------
    ctree_true: dict(str, RSTTree)
        Ground truth RST ctree.
    dtree_true: dict(str, RstDepTree)
        Ground truth RST (ordered) dtree.
    skpd_docs: set(string)
        Names of documents that should be skipped to compute scores

    Returns
    -------
    skipped_docs: set(string)
        Names of documents that have been skipped to compute scores
    """
    # USE TO INCORPORATE CONSTITUENCY LOSS INTO STRUCTURED CLASSIFIERS
    # load predicted trees
    # end USE TO INCORPORATE CONSTITUENCY LOSS INTO STRUCTURED CLASSIFIERS
