"""Module to load .tree files, output by Feng's gCRF parser.

The .tree files contain binary constituency trees as bracketed strings.
They differ from the .dis files in that the relation label and
nuclearity are written on the top node instead of the daughter nodes,
plus edu spans are not explicitly written at each node.
"""

from __future__ import absolute_import, print_function
import codecs
from glob import glob
import os
import re

from nltk.tree import Tree

from educe.rst_dt.annotation import EDU, Node, SimpleRSTTree, Span
from educe.rst_dt.deptree import RstDepTree


TXT_RE = r"(?P<prefix>.+)_!(?P<text>.+)!_(?P<suffix>.+)"
TXT_PATTERN = re.compile(TXT_RE, flags=re.DOTALL)


def reduce_preterminal(terminals, txt_offset, edu_offset):
    """Create a pre-terminal from a list of terminals.

    Parameters
    ----------
    terminals: list of str
        List of terminals

    Returns
    -------
    sct: SimpleRSTTree
        Pre-terminal.
    """
    edu_num = edu_offset
    edu_txt = ' '.join(terminals)
    assert edu_txt.startswith('_!') and edu_txt.endswith('!_')
    edu_txt = edu_txt[2:-2]  # shave off _! and !_
    edu_txt_span = Span(txt_offset,
                        txt_offset + len(edu_txt))
    edu = EDU(edu_num, edu_txt_span, edu_txt,
              context=None,
              origin=None)
    # "pre-terminal"
    pre_node = Node('leaf', (edu_num, edu_num), edu_txt_span,
                    'leaf', context=None)
    sct = SimpleRSTTree(pre_node, [edu])
    return sct


def nltk_to_simple(node, txt_offset=0, edu_offset=1):
    """Convert an NLTK Tree to a SimpleRSTTree.

    Parameters
    ----------
    node: Tree
        Current tree node.
    txt_offset: int, defaults to 0
        Current text offset.
    edu_offset: int, defaults to 1
        Current EDU id offset.

    Returns
    -------
    sct: SimpleRSTTree
        Corresponding SimpleRSTTree.
    """
    cur_txt_offset = txt_offset
    cur_edu_offset = edu_offset

    # first, recurse: convert kids
    new_kids = []
    for kid in node:
        if isinstance(kid, Tree):
            # convert gCRF .tree subtree to SimpleRSTTree
            new_kid = nltk_to_simple(kid, txt_offset=cur_txt_offset,
                                     edu_offset=cur_edu_offset)
            # update current offsets
            cur_txt_offset = new_kid.label().span.char_end + 1
            cur_edu_offset = new_kid.label().edu_span[1] + 1
            new_kids.append(new_kid)
        else:
            # kid is a terminal
            # first, restore parentheses in the text
            kid = kid.replace('-LRB-', '(').replace('-RRB-', ')')
            #
            if not new_kids or isinstance(new_kids[-1], SimpleRSTTree):
                new_kids.append([])
            new_kids[-1].append(kid)
            if kid.endswith('!_'):
                new_kid = reduce_preterminal(
                    new_kids[-1], cur_txt_offset, cur_edu_offset)
                new_kids[-1] = new_kid
                # update current offsets
                # * txt_offset: + 1 for whitespace or newline
                cur_txt_offset = new_kid.label().span.char_end + 1
                # * edu_offset: + 1 for next EDU
                cur_edu_offset = new_kid.label().edu_span[1] + 1
    # check that all have been converted
    assert all(isinstance(x, SimpleRSTTree) for x in new_kids)

    # we can now compute the label ; the edu_span depends on the
    # recursive calls
    lbl = node.label()
    rel, nuc = lbl.split('[', 1)  # nuc = "N][S]"
    nuc = nuc[0] + nuc[3]
    edu_span = (new_kids[0].label().edu_span[0],
                new_kids[-1].label().edu_span[1])
    txt_span = Span(new_kids[0].label().span.char_start,
                    new_kids[-1].label().span.char_end)
    new_lbl = Node(nuc, edu_span, txt_span, rel)
    return SimpleRSTTree(new_lbl, new_kids)
    

def _load_gcrf_tree_file(f):
    """Do load"""
    # replace parentheses in text to avoid confusion with parentheses
    # denoting the bracketed tree structure
    lines = []
    for line in f:
        # replace non-breaking spaces... damn python 2
        if u"\u00a0" in line:
            line = line.replace(u"\u00a0", u" ")
        #
        m = TXT_PATTERN.match(line)
        if m is not None:
            new_line = (m.group('prefix')
                        + '_!'
                        + (m.group('text')
                           .replace('(', '-LRB-')
                           .replace(')', '-RRB-'))
                        + '!_'
                        + m.group('suffix'))
            line = new_line
        lines.append(line)
    ct_str = ''.join(lines)
    ct = Tree.fromstring(ct_str)
    sct = nltk_to_simple(ct)
    return sct


def load_gcrf_tree_file(fname):
    """Load a gCRF tree file.

    Parameters
    ----------
    fname: str
        Path to the file to be loaded.

    Returns
    -------
    ct: SimpleRSTTree
        Binary constituency tree with relation label and nuclearity
        moved one up.
    """
    with codecs.open(fname, encoding='utf-8') as f:
        ct = _load_gcrf_tree_file(f)
    return ct


def load_gcrf_ctrees(out_dir, rel_conv):
    """Load the ctrees output by gCRF as .tree files.

    Parameters
    ----------
    out_dir: str
        Path to the base directory containing the output files.

    Returns
    -------
    ctree_pred: dict(str, RSTTree)
        RST ctree for each document.
    """
    ctree_pred = dict()
    for f_tree in glob(os.path.join(out_dir, '*.tree')):
        doc_name = os.path.splitext(os.path.basename(f_tree))[0]
        sct_pred = load_gcrf_tree_file(f_tree)
        ct_pred = SimpleRSTTree.to_binary_rst_tree(sct_pred)
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


def load_gcrf_dtrees(out_dir, rel_conv, nary_enc='chain', ctree_pred=None):
    """Get the dtrees that correspond to the ctrees output by gCRF.

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
        ctree_pred = load_gcrf_ctrees(out_dir, rel_conv)
    dtree_pred = dict()
    for doc_name, ct_pred in ctree_pred.items():
        dt_pred = RstDepTree.from_rst_tree(ct_pred, nary_enc=nary_enc)
        dtree_pred[doc_name] = dt_pred
    return dtree_pred
