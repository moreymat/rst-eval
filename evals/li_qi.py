"""Load the output of the parser from (Li et al. 2016).

This is 99% a copy/paste from our own evals/joty.py.
I really, really need to come up with a better API and refactor accordingly.
"""

import codecs
import glob
import itertools
import os

from educe.rst_dt.parse import parse_rst_dt_tree
from educe.rst_dt.deptree import RstDepTree


def load_li_qi_output_files(root_dir):
    """Load ctrees output by Li Qi's parser on the TEST section of the RST-DT.

    Parameters
    ----------
    root_dir: string
        Path to the main folder containing the parser's output

    Returns
    -------
    data: dict
        Dictionary that should be akin to a sklearn Bunch, with
        interesting keys 'filenames', 'doc_names' and 'rst_ctrees'.

    Notes
    -----
    To ensure compatibility with the rest of the code base, doc_names
    are automatically added the ".out" extension. This would not work
    for fileX documents, but they are absent from the TEST section of
    the RST-WSJ treebank.
    """
    # map output filename to doc filename:
    # here, remove prefix "parsed_"
    # ex of filename: parsed_wsj_0602.out
    out_filenames = sorted(glob.glob(os.path.join(root_dir, 'parsed_*')))
    doc_names = [os.path.basename(out_fn).split('_', 1)[1]
                 for out_fn in out_filenames]
    # load the RST trees
    rst_ctrees = []
    for out_fn in out_filenames:
        with codecs.open(out_fn, 'r', 'utf-8') as f:
            # TODO(?) add support for and use RSTContext
            rst_ctree = parse_rst_dt_tree(f.read(), None)
            rst_ctrees.append(rst_ctree)

    data = dict(filenames=out_filenames,
                doc_names=doc_names,
                rst_ctrees=rst_ctrees)
    return data


def load_li_qi_ctrees(out_dir, rel_conv):
    """Load the ctrees output by Li Qi's parser as .dis files.

    This currently runs on the document-level files (.doc_dis).

    Parameters
    ----------
    out_dir: str
        Path to the base directory containing the output files.

    Returns
    -------
    ctree_pred: dict(str, RSTTree)
        RST ctree for each document.
    """
    # load predicted trees
    data_pred = load_li_qi_output_files(out_dir)
    doc_names_pred = data_pred['doc_names']
    rst_ctrees_pred = data_pred['rst_ctrees']
    # map doc_name to ctree (RSTTree)
    ctree_pred = dict()
    for doc_name, ct_pred in zip(doc_names_pred, rst_ctrees_pred):
        # ctree
        # replace fine-grained labels with coarse-grained labels :
        # the files we have already contain the coarse labels, except their
        # initial letter is capitalized, except for same-unit and span,
        # whereas ours are not
        if rel_conv is not None:
            ct_pred = rel_conv(ct_pred)
        ctree_pred[doc_name] = ct_pred

    return ctree_pred


def load_li_qi_dtrees(out_dir, rel_conv, nary_enc='chain', ctree_pred=None):
    """Get the dtrees that correspond to the ctrees output by Li Qi's parser.

    Parameters
    ----------
    out_dir: str
        Path to the base directory containing the output files.
    nary_enc: one of {'chain', 'tree'}
        Encoding for n-ary nodes.
    ctree_pred : dict(str, RSTTree), optional
        RST c-trees, indexed by doc_name. If c-trees are provided this
        way, `out_dir` is ignored.

    Returns
    -------
    dtree_pred: dict(str, RstDepTree)
        RST dtree for each document.
    """
    if ctree_pred is None:
        # load predicted trees
        data_pred = load_li_qi_output_files(out_dir)
        # filenames = data_pred['filenames']
        doc_names_pred = data_pred['doc_names']
        rst_ctrees_pred = data_pred['rst_ctrees']
        ctree_pred = {doc_name: ct_pred for doc_name, ct_pred
                      in zip(doc_names_pred, rst_ctrees_pred)}
    # build a dict from doc_name to ordered dtree (RstDepTree)
    dtree_pred = dict()
    for doc_name, ct_pred in ctree_pred.items():
        # constituency tree
        # replace fine-grained labels with coarse-grained labels ;
        # the files we have already contain the coarse labels, except their
        # initial letter is capitalized whereas ours are not
        if rel_conv is not None:
            ct_pred = rel_conv(ct_pred)
        # convert to an ordered dependency tree ;
        # * 'tree' produces a weakly-ordered dtree strictly equivalent
        # to the original ctree,
        # * 'chain' produces a strictly-ordered dtree for which strict
        # equivalence is not preserved
        dt_pred = RstDepTree.from_rst_tree(ct_pred, nary_enc=nary_enc)
        dtree_pred[doc_name] = dt_pred

    return dtree_pred

