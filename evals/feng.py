"""Load the output of the RST parser from (Feng and Hirst, 2014).

This is 99% a copy/paste from evals/joty.py .
I need to come up with a better API and refactor accordingly.
"""

import codecs
import glob
import itertools
import os

from nltk import Tree

from educe.rst_dt.deptree import RstDepTree
from educe.rst_dt.parse import parse_rst_dt_tree


def load_feng_output_files(root_dir):
    """Load ctrees output by Feng & Hirst's parser on the TEST section of
    RST-WSJ.

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
    # find all files with the right extension
    file_ext = '.txt.dis'
    pathname = os.path.join(root_dir, '*{}'.format(file_ext))
    # filenames are sorted by name to avoid having to realign data
    # loaded with different functions
    filenames = sorted(glob.glob(pathname))  # glob.glob() returns a list

    # find corresponding doc names
    doc_names = [os.path.basename(filename).rsplit('.', 2)[0] + '.out'
                 for filename in filenames]

    # load the RST trees
    rst_ctrees = []
    for filename in filenames:
        with codecs.open(filename, 'r', 'utf-8') as f:
            # TODO (?) add support for and use RSTContext
            rst_ctree = parse_rst_dt_tree(f.read(), None)
            rst_ctrees.append(rst_ctree)

    data = dict(filenames=filenames,
                doc_names=doc_names,
                rst_ctrees=rst_ctrees)

    return data


def load_feng_ctrees(out_dir, rel_conv):
    """Load the ctrees output by Feng's parser as .dis files.

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
    data_pred = load_feng_output_files(out_dir)
    # filenames = data_pred['filenames']
    doc_names_pred = data_pred['doc_names']
    rst_ctrees_pred = data_pred['rst_ctrees']

    # build a dict from doc_name to ctree (RSTTree)
    ctree_pred = dict()  # constituency trees
    for doc_name, ct_pred in zip(doc_names_pred, rst_ctrees_pred):
        # constituency tree
        # replace fine-grained labels with coarse-grained labels ;
        # the files we have already contain the coarse labels, except their
        # initial letter is capitalized whereas ours are not
        if rel_conv is not None:
            ct_pred = rel_conv(ct_pred)
        # "normalize" names of classes of RST relations:
        # "textual-organization" => "textual"
        for pos in ct_pred.treepositions():
            t = ct_pred[pos]
            if isinstance(t, Tree):
                node = t.label()
                if node.rel == 'textual-organization':
                    node.rel = 'textual'
        # end normalize
        ctree_pred[doc_name] = ct_pred

    return ctree_pred


def load_feng_dtrees(out_dir, rel_conv, nary_enc='chain'):
    """Get the dtrees that correspond to the ctrees output by Feng's parser.

    Parameters
    ----------
    out_dir: str
        Path to the base directory containing the output files.
    nary_enc: one of {'chain', 'tree'}
        Encoding for n-ary nodes.

    Returns
    -------
    dtree_pred: dict(str, RstDepTree)
        RST dtree for each document.
    """
    # load predicted c-trees
    ctree_pred = load_feng_ctrees(out_dir, rel_conv)

    # build a dict from doc_name to ordered dtree (RstDepTree)
    dtree_pred = dict()
    for doc_name, ct_pred in ctree_pred.items():
        # convert to an ordered dependency tree ;
        # * 'tree' produces a weakly-ordered dtree strictly equivalent
        # to the original ctree,
        # * 'chain' produces a strictly-ordered dtree for which strict
        # equivalence is not preserved
        dt_pred = RstDepTree.from_rst_tree(ct_pred, nary_enc=nary_enc)
        dtree_pred[doc_name] = dt_pred

    return dtree_pred
