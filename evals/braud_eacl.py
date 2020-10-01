"""Read the output of Braud et al.'s EACL parsers.

"""

from __future__ import absolute_import, print_function

import codecs
import itertools
from glob import glob
import os

from nltk import Tree

from educe.annotation import Span
from educe.rst_dt.annotation import EDU, Node, SimpleRSTTree
from educe.rst_dt.deptree import RstDepTree


def tree_to_simple_rsttree(tree, edu_num=1):
    """Build a SimpleRSTTree from a NLTK Tree.

    Parameters
    ----------
    edu_num : int, defaults to 1
        Number of the next EDU
    """
    origin = None

    if tree.label() == 'EDU':
        # EDU (+pre-terminal)
        num = edu_num
        span = Span(num, num)
        # 'EDU <joint_text>'
        text = tree[0]
        edu = EDU(num, span, text, context=None, origin=origin)
        # pre-terminal
        edu_span = (num, num)
        nuc = "leaf"
        rel = "leaf"
        node = Node(nuc, edu_span, span, rel, context=None)
        return SimpleRSTTree(node, [edu], origin=origin)

    new_kids = []
    for kid in tree:
        new_kid = tree_to_simple_rsttree(kid, edu_num=edu_num)
        edu_num = new_kid.label().edu_span[1] + 1
        new_kids.append(new_kid)

    # ROOT
    if tree.label() == 'ROOT':
        assert len(new_kids) == 1
        return new_kids[0]

    # internal node
    # label: 'NNTextualorganization'
    nuc = tree.label()[:2]
    rel = tree.label()[2:]
    # map to our coarse rel names
    rel_map = {
        'MannerMeans': 'manner-means',
        'Sameunit': 'same-unit',
        'TopicChange': 'topic-change',
        'TopicComment': 'topic-comment',
    }        
    rel = rel_map.get(rel, rel)
    # end map

    # same as in braud_coling
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


def _load_braud_eacl_file(f):
    """Do load SimpleRSTTrees from f"""
    sctrees = []
    for line in f:
        tree = Tree.fromstring(line.strip())
        sctree = tree_to_simple_rsttree(tree)
        sctrees.append(sctree)
    return sctrees


def load_braud_eacl_file(fpath):
    """Load SimpleRSTTrees from a file"""
    with codecs.open(fpath, 'rb', 'utf-8') as f:
        return _load_braud_eacl_file(f)


def load_braud_eacl_ctrees(fpath, rel_conv, doc_names):
    """Load the ctrees output by Braud et al.'s parser

    Parameters
    ----------
    fpath : str
        Path to the output file.

    rel_conv : TODO
        Relation converter.

    Returns
    -------
    ctree_pred : dict(str, RSTTree)
        RST c-tree for each document.
    """
    ctree_pred = dict()
    sctree_pred = load_braud_eacl_file(fpath)
    for doc_name, sct_pred in zip(doc_names, sctree_pred):
        ct_pred = SimpleRSTTree.to_binary_rst_tree(sct_pred)
        ct_pred = rel_conv(ct_pred)
        ctree_pred[doc_name] = ct_pred
    return ctree_pred


def load_braud_eacl_dtrees(fpath, rel_conv, doc_names, nary_enc='chain',
                           ctree_pred=None):
    """Do load dtrees

    Parameters
    ----------
    ctree_pred : dict(str, RSTTree), optional
        RST c-trees, indexed by doc_name. If c-trees are provided this
        way, `out_dir` is ignored.
    """
    dtree_pred = dict()
    if ctree_pred is None:
        ctree_pred = load_braud_eacl_ctrees(fpath, rel_conv, doc_names)
    for doc_name, ct_pred in ctree_pred.items():
        dt_pred = RstDepTree.from_rst_tree(ct_pred)
        dtree_pred[doc_name] = dt_pred
    return dtree_pred
