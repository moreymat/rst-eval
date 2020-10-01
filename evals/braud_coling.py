"""Read the output of Braud et al.'s COLING parser.

"""

from __future__ import absolute_import, print_function

import codecs
from glob import glob
import itertools
import os

from nltk import Tree

from educe.annotation import Span
from educe.rst_dt.annotation import EDU, Node, SimpleRSTTree
from educe.rst_dt.deptree import RstDepTree


# map *.mrg.pred files to the original doc names
MRG_TO_RST = {
    '12.mrg.pred': 'wsj_0644.out',  # 4
    '4.mrg.pred': 'wsj_1129.out',  # 5
    '26.mrg.pred': 'wsj_1197.out',  # 6
    '24.mrg.pred': 'wsj_1113.out',  # 8
    '14.mrg.pred': 'wsj_0684.out',  # 10
    '32.mrg.pred': 'wsj_1354.out',  # 11
    '18.mrg.pred': 'wsj_1183.out',  # 12
    '29.mrg.pred': 'wsj_1346.out',  # 15
    '28.mrg.pred': 'wsj_1169.out',  # 17
    '37.mrg.pred': 'wsj_0667.out',  # 17
    '19.mrg.pred': 'wsj_0607.out', # 19
    '7.mrg.pred': 'wsj_0654.out', # 19
    '16.mrg.pred': 'wsj_1325.out',  # 21
    '25.mrg.pred': 'wsj_2375.out',  # 22
    '31.mrg.pred': 'wsj_1380.out',  # 23
    '1.mrg.pred': 'wsj_0623.out',  # 25
    '15.mrg.pred': 'wsj_2373.out',  # 31
    '30.mrg.pred': 'wsj_2336.out',  # 31
    '3.mrg.pred': 'wsj_1365.out',  # 39
    '34.mrg.pred': 'wsj_1148.out',  # 43
    '11.mrg.pred': 'wsj_1306.out',  # 47
    '10.mrg.pred': 'wsj_2354.out',  # 52
    '35.mrg.pred': 'wsj_1126.out',  # 55
    '0.mrg.pred': 'wsj_2385.out',  # 60
    '2.mrg.pred': 'wsj_0632.out',  # 62
    '20.mrg.pred': 'wsj_0602.out',  # 69
    '27.mrg.pred': 'wsj_0627.out',  # 69
    '13.mrg.pred': 'wsj_1189.out',  # 91
    '6.mrg.pred': 'wsj_0616.out',  # 92
    '36.mrg.pred': 'wsj_1307.out',  # 98
    '33.mrg.pred': 'wsj_1142.out',  # 106
    '9.mrg.pred': 'wsj_0655.out',  # 110
    '21.mrg.pred': 'wsj_2386.out',  # 127
    '23.mrg.pred': 'wsj_0689.out',  # 132
    '8.mrg.pred': 'wsj_1387.out',  # 134
    '17.mrg.pred': 'wsj_1331.out',  # 158
    '22.mrg.pred': 'wsj_1376.out',  # 202
    '5.mrg.pred': 'wsj_1146.out',  # 304
}


def tree_to_simple_rsttree(tree):
    """Build a SimpleRSTTree from a NLTK Tree"""
    origin = None  # or is it?
    if not tree:
        # no kid: EDU (+pre-terminal)
        num = int(tree.label())
        span = Span(num, num)  # FIXME
        text = ''  # FIXME
        edu = EDU(num, span, text, context=None, origin=origin)
        # pre-terminal
        edu_span = (num, num)
        nuc = "leaf"
        rel = "leaf"
        node = Node(nuc, edu_span, span, rel, context=None)
        return SimpleRSTTree(node, [edu], origin=origin)

    # internal node
    new_kids = [tree_to_simple_rsttree(kid) for kid in tree]
    # node
    nuc, rel = tree.label().split('-', 1)
    # map to our coarse rel names
    if rel == 'Textual-organization':
        rel = 'Textual'
    # end map
    edu_beg = (new_kids[0].num if isinstance(new_kids[0], EDU)
               else new_kids[0].label().edu_span[0])
    edu_end = (new_kids[-1].num if isinstance(new_kids[-1], EDU)
               else new_kids[-1].label().edu_span[1])
    edu_span = (edu_beg, edu_end)
    char_beg = (new_kids[0].num if isinstance(new_kids[0], EDU)
                  else new_kids[0].label().span.char_start)
    char_end = (new_kids[-1].num if isinstance(new_kids[-1], EDU)
                else new_kids[-1].label().span.char_end)
    span = Span(char_beg, char_end)
    new_node = Node(nuc, edu_span, span, rel, context=None)
    new_tree = SimpleRSTTree(new_node, new_kids, origin=origin)
    return new_tree


def _load_braud_coling_file(f):
    """Do load file"""
    tree = Tree.fromstring(f.read().strip())
    simple_ctree = tree_to_simple_rsttree(tree)
    return simple_ctree
        

def load_braud_coling_file(fpath):
    """Load a file."""
    with codecs.open(fpath, 'rb', 'utf-8') as f:
        return _load_braud_coling_file(f)


def load_braud_coling_ctrees(out_dir, rel_conv):
    """Load the ctrees output by Braud et al.'s parser

    Parameters
    ----------
    out_dir : str
        Path to the output directory.

    rel_conv : TODO
        Relation converter

    Returns
    -------
    ctree_pred : dict(str, RSTTree)
        RST c-tree for each document.
    """
    ctree_pred = dict()
    for fpath in sorted(glob(os.path.join(out_dir, '*.mrg.pred'))):
        fname = os.path.basename(fpath)
        doc_name = MRG_TO_RST.get(fname, fname)
        sct_pred = load_braud_coling_file(fpath)
        # convert to regular RSTTree
        ct_pred = SimpleRSTTree.to_binary_rst_tree(sct_pred)
        # convert relation labels
        ct_pred = rel_conv(ct_pred)
        # TODO check ct_true: assert that mrg.gold == .out.dis
        ctree_pred[doc_name] = ct_pred
    return ctree_pred


def load_braud_coling_dtrees(out_dir, rel_conv, nary_enc='chain',
                             ctree_pred=None):
    """Do load dtrees.

    Parameters
    ----------
    ctree_pred : dict(str, RSTTree), optional
        RST c-trees, indexed by doc_name. If c-trees are provided this
        way, `out_dir` is ignored.
    """
    dtree_pred = dict()
    if ctree_pred is None:
        ctree_pred = load_braud_coling_ctrees(out_dir, rel_conv)
    for doc_name, ct_pred in ctree_pred.items():
        dt_pred = RstDepTree.from_rst_tree(ct_pred)
        dtree_pred[doc_name] = dt_pred
    return dtree_pred
