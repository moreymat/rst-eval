"""Load RST c-trees output by Hayashi et al.'s reimplementation of HILDA.

"""

from collections import namedtuple
import codecs
import glob
import itertools
import os

from nltk import Tree

from educe.annotation import Span
from educe.rst_dt.annotation import EDU, Node, RSTTree
from educe.rst_dt.deptree import RstDepTree


node_struct = namedtuple('node_struct', ['nuc', 'rel', 'span'])

def read_node(s):
    """Helper applied when reading a node"""
    nuc, rel = s.split(':') if s != 'Root' else (s, '---')
    res = node_struct(nuc=nuc, rel=rel, span=(0, 0))
    return res


leaf_struct = namedtuple('leaf_struct', ['edu_id', 'sent_id', 'para_id'])

def read_leaf(s):
    """Helper applied when reading a leaf"""
    edu_id, sent_id, para_id = s[4:].split('_')  # ex: leaf1_1_1
    res = leaf_struct(edu_id=edu_id, sent_id=sent_id,
                      para_id=para_id)
    return res

def propagate_spans(t):
    """Propagate spans bottom-up in our custom NLTK tree."""
    dft_span = Span(0, 0)  # default text span
    dft_text = ''

    lbl = t.label()
    if all(isinstance(kid, Tree) for kid in t):
        new_kids = [propagate_spans(kid) for kid in t]
        edu_start = new_kids[0].label().edu_span[0]
        edu_end = new_kids[-1].label().edu_span[1]
    else:
        # pre-terminal
        assert len(t) == 1
        kid = t[0]
        new_kid = EDU(int(kid.edu_id), dft_span, dft_text)
        new_kids = [new_kid]
        edu_start = new_kid.num
        edu_end = new_kid.num
    new_lbl = Node(lbl.nuc, (edu_start, edu_end), dft_span, lbl.rel)
    new_tree = RSTTree(new_lbl, new_kids)
    return new_tree


def load_hayashi_con_files(root_dir):
    """Load the ctrees output by Hayashi et al.'s reimplementation of HILDA.

    The RST ctrees are supposedly document-level RST trees, with classes of
    relations.

    Parameters
    ----------
    out_dir: str
        Path to the base directory containing the output files.

    Returns
    -------
    data: dict
        Dictionary that should be akin to a sklearn Bunch, with
        interesting keys 'filenames', 'doc_names' and 'rst_ctrees'.
    """
    # map output filename to doc filename
    # ex of filename: wsj_0602.out.dis
    out_filenames = sorted(glob.glob(os.path.join(root_dir, '*.dis')))
    doc_names = [os.path.basename(out_fn).rsplit('.', 1)[0]
                 for out_fn in out_filenames]
    # load the RST trees
    rst_ctrees = []
    for out_fn in out_filenames:
        with codecs.open(out_fn, 'r', 'utf-8') as f:
            tree_str = f.read()
            tree_raw = Tree.fromstring(tree_str, read_node=read_node,
                                       read_leaf=read_leaf)
            # TODO(?) add support for and use RSTContext
            rst_ctree = propagate_spans(tree_raw)
            rst_ctrees.append(rst_ctree)

    data = dict(filenames=out_filenames,
                doc_names=doc_names,
                rst_ctrees=rst_ctrees)
    return data


def load_hayashi_hilda_ctrees(out_dir, rel_conv):
    """Load the ctrees output by Hayashi et al.'s HILDA.

    Parameters
    ----------
    out_dir: str
        Path to the folder containing .dis files.
    rel_conv: RstRelationConverter
        Converter for relation labels (fine- to coarse-grained, plus
        normalization).

    Returns
    -------
    ctree_pred: dict(str, RSTTree)
        RST ctree for each document.
    """
    # load predicted ctrees
    data_pred = load_hayashi_con_files(out_dir)
    doc_names_pred = data_pred['doc_names']
    rst_ctrees_pred = data_pred['rst_ctrees']

    # build a dict from doc_name to RST ctree
    ctree_pred = dict()
    for doc_name, ct_pred in zip(doc_names_pred, rst_ctrees_pred):
        if rel_conv is not None:
            ct_pred = rel_conv(ct_pred)
        ctree_pred[doc_name] = ct_pred
    return ctree_pred


def load_hayashi_hilda_dtrees(out_dir, rel_conv, nary_enc='chain',
                              ctree_pred=None):
    """Load the dtrees for the ctrees output by Hayashi et al.'s HILDA.

    Parameters
    ----------
    out_dir: str
        Path to the folder containing .dis files.
    rel_conv: RstRelationConverter
        Converter for relation labels (fine- to coarse-grained, plus
        normalization).
    ctree_pred : dict(str, RSTTree), optional
        RST c-trees, indexed by doc_name. If c-trees are provided this
        way, `out_dir` is ignored.

    Returns
    -------
    dtree_pred: dict(str, RstDepTree)
        RST dtree for each document.
    """
    if ctree_pred is None:
        # load predicted ctrees
        ctree_pred = load_hayashi_hilda_ctrees(out_dir, rel_conv)
    # convert to dtrees
    dtree_pred = dict()
    for doc_name, ct_pred in ctree_pred.items():
        dt_pred = RstDepTree.from_rst_tree(ct_pred, nary_enc=nary_enc)
        dtree_pred[doc_name] = dt_pred

    return dtree_pred
