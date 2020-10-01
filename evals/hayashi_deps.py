"""Load dependencies output by Hayashi et al.'s parsers.

This module enables to process files in auto_parse/{dep/li,cons/trans_li}.
"""

from __future__ import absolute_import, print_function

import os
from glob import glob

from educe.learning.edu_input_format import load_edu_input_file
from educe.rst_dt.deptree import RstDepTree, RstDtException
from educe.rst_dt.dep2con import deptree_to_rst_tree


def _load_hayashi_dep_file(f, edus):
    """Do load.

    Parameters
    ----------
    f: File
        dep file, open
    edus: list of EDU
        True EDUs in this document.

    Returns
    -------
    dt: RstDepTree
        Predicted dtree
    """
    dt = RstDepTree(edus=edus, origin=None, nary_enc='chain')  # FIXME origin
    for line in f:
        line = line.strip()
        if not line:
            continue
        dep_idx, gov_idx, lbl = line.split()
        dep_idx = int(dep_idx)
        gov_idx = int(gov_idx)
        dt.add_dependency(gov_idx, dep_idx, label=lbl)
    return dt


def load_hayashi_dep_file(fname, edus):
    """Load a file.

    Parameters
    ----------
    fname: str
        Path to the file

    Returns
    -------
    dt: RstDepTree
        Dependency tree corresponding to the content of this file.
    """
    with open(fname) as f:
        return _load_hayashi_dep_file(f, edus)


def load_hayashi_dep_files(out_dir, doc_edus):
    """Load dep files output by one of Hayashi et al.'s parser.

    Parameters
    ----------
    out_dir: str
        Path to the folder containing the .dis files.
    doc_edus : dict(str, list(EDU))
        Mapping from doc_name to the list of its EDUs (read from the
        corpus).
    """
    dtrees = dict()
    for fname in glob(os.path.join(out_dir, '*.dis')):
        doc_name = os.path.splitext(os.path.basename(fname))[0]
        edus = doc_edus[doc_name]
        dtrees[doc_name] = load_hayashi_dep_file(fname, edus)
    return dtrees


def load_hayashi_dep_dtrees(out_dir, rel_conv, doc_edus, edus_file_pat,
                            nuc_clf, rnk_clf):
    """Load the dtrees output by one of Hayashi et al.'s dep parsers.

    Parameters
    ----------
    out_dir : str
        Path to the folder containing .dis files.
    rel_conv : RstRelationConverter
        Converter for relation labels (fine- to coarse-grained, plus
        normalization).
    doc_edus : dict(str, list(EDU))
        Mapping from doc_name to the list of its EDUs (read from the
        corpus).
    edus_file_pat : str
        Pattern for the .edu_input files.
    nuc_clf : NuclearityClassifier
        Nuclearity classifier
    rnk_clf : RankClassifier
        Rank classifier

    Returns
    -------
    dtree_pred: dict(str, RstDepTree)
        RST dtree for each document.
    """
    dtree_pred = dict()

    dtrees = load_hayashi_dep_files(out_dir, doc_edus)
    for doc_name, dt_pred in dtrees.items():
        if rel_conv is not None:
            dt_pred = rel_conv(dt_pred)
        # normalize names of classes of RST relations:
        # "root" is "ROOT" in my coarse labelset (TODO: make it consistent)
        dt_pred.labels = ['ROOT' if x == 'root' else x
                          for x in dt_pred.labels]
        # end normalize
        # WIP add nuclearity and rank
        edus_data = load_edu_input_file(edus_file_pat.format(doc_name),
                                        edu_type='rst-dt')
        edu2sent = edus_data['edu2sent']
        dt_pred.sent_idx = [0] + edu2sent  # 0 for fake root ; DIRTY
        dt_pred.nucs = nuc_clf.predict([dt_pred])[0]
        dt_pred.ranks = rnk_clf.predict([dt_pred])[0]
        # end WIP
        dtree_pred[doc_name] = dt_pred
        
    return dtree_pred


def load_hayashi_dep_ctrees(out_dir, rel_conv, doc_edus, edus_file_pat,
                            nuc_clf, rnk_clf, dtree_pred=None):
    """Load the ctrees for the dtrees output by one of Hayashi et al.'s
    dep parsers.

    Parameters
    ----------
    out_dir : str
        Path to the folder containing .dis files.
    rel_conv : RstRelationConverter
        Converter for relation labels (fine- to coarse-grained, plus
        normalization).
    doc_edus : dict(str, list(EDU))
        Mapping from doc_name to the list of its EDUs (read from the
        corpus).
    edus_file_pat : str
        Pattern for the .edu_input files.
    nuc_clf : NuclearityClassifier
        Nuclearity classifier
    rnk_clf : RankClassifier
        Rank classifier
    dtree_pred : dict(str, RstDepTree), optional
        RST d-trees, indexed by doc_name. If d-trees are provided this
        way, `out_dir` is ignored.

    Returns
    -------
    ctree_pred: dict(str, RSTTree)
        RST ctree for each document.
    """
    ctree_pred = dict()
    if dtree_pred is None:
        dtree_pred = load_hayashi_dep_dtrees(out_dir, rel_conv, doc_edus,
                                             edus_file_pat,
                                             nuc_clf, rnk_clf)
    for doc_name, dt_pred in dtree_pred.items():
        try:
            ct_pred = deptree_to_rst_tree(dt_pred)
        except RstDtException:
            print(doc_name)
            raise
        else:
            ctree_pred[doc_name] = ct_pred

    return ctree_pred
