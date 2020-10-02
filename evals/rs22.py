"""Load RST trees output by Alexeeva et al.'s forthcoming parser.

This format differs slightly from that of Surdeanu's parser :
- Each predicted tree is prefixed with the document name (ex:
'Document::wsj_0632.out::(elaboration:NS
[...] 
)'
- ?
"""

import re

from nltk import Tree

from educe.annotation import Span
from educe.corpus import FileId
from educe.rst_dt.annotation import EDU, Node, SimpleRSTTree
from educe.rst_dt.deptree import RstDepTree


# timestamped line
TS_LINE = r"\d\d:\d\d:\d\d.\d\d\d \[run-main-0\].*"
TS_RE = re.compile(TS_LINE)


def tree_to_simple_rsttree(tree, edu_num=1):
    """Build a SimpleRSTTree from an NLTK Tree (formatted a la Surdeanu).

    Parameters
    ----------
    tree : nltk.Tree
        Tree

    edu_num : int, defaults to 1
        Number of the next EDU

    Returns
    -------
    sct : SimpleRSTTree
        The corresponding SimpleRSTTree.
    """
    origin = None

    if tree.label() == 'TEXT':
        # EDU (+pre-terminal)
        num = edu_num
        span = Span(num, num)
        # 'TEXT <text>'
        text = '__'.join(tree)
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

    # internal node
    # (modified) label: 'elaboration:NS' or 'joint' (no explicit nuc: NN)
    if tree.label()[-3] == ':':
        rel = tree.label()[:-3]
        nuc = tree.label()[-2:]
    else:
        rel = tree.label()
        nuc = 'NN'
    # map to our coarse rel names
    # TODO?
    # end map
    # same as in braud_coling and braud_eacl
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


def _load_rs22_ctrees(log_file, rel_conv):
    """Do load"""
    doc_names = []
    nltk_ctrees = []
    ctree_pred = dict()  # result

    ctree_cur = []  # lines for the current c-tree
    state_cur = 0  # current state (finite state machine for dummies)
    for line in log_file:
        # DIRTY replace non-breaking spaces output by CoreNLP, as in
        # educe.rst_dt.learning.doc_vectorizer
        # if isinstance(line, unicode):  # python 2
        if isinstance(line, str):  # python 3
            line2 = line.replace(u'\xa0', u' ')
            # line = line2.encode('utf-8')  # python 2
        # end replace

        if line.startswith('Document::'):
            # new doc tree
            if ctree_cur:
                # parse the preceding predicted c-tree (accumulated lines)
                nltk_ct_pred = Tree.fromstring(''.join(ctree_cur))
                nltk_ctrees.append(nltk_ct_pred)
                # reset accumulator
                ctree_cur = []
            # beginning of predicted tree for doc:
            # 'Document::wsj_0632.out::(elaboration:NS'
            _, doc_name, tree_beg = line.split('::')
            doc_names.append(doc_name)
            # replace "_" with "-" in relations, see below
            tree_beg = (tree_beg
                        .replace("topic_change", "topic-change")
                        .replace("manner_means", "manner-means")
                        .replace("textual_organization", "textual")
                        .replace("topic_comment", "topic-comment")
                        .replace("same_unit", "same-unit")
                        .replace("None", "textual")  # FIXME confirm with authors?
            )           
            # end replace
            ctree_cur.append(tree_beg)
        elif line.strip() == '':
            # do nothing (unless we want to parse the preceding predicted c-tree here?)
            continue
        else:
            # accumulate lines for the next predicted c-tree
            # we immediately replace "_" with "-" in relations, plus "textual" ;
            # remember that if neither :NS nor :SN, then it is implicitly ":NN"
            line = (line
                    .replace("topic_change", "topic-change")
                    .replace("manner_means", "manner-means")
                    .replace("textual_organization", "textual")
                    .replace("topic_comment", "topic-comment")
                    .replace("same_unit", "same-unit")
                    .replace("None", "textual")  # FIXME confirm with authors?
            )
            ctree_cur.append(line)
    else:
        # parse the last predicted c-tree (accumulated lines)
        nltk_ct_pred = Tree.fromstring(''.join(ctree_cur))
        nltk_ctrees.append(nltk_ct_pred)
        # reset accumulator
        ctree_cur = []
    # FIXME we got two predicted ctrees for each doc, with gold then predicted EDUs
    # filter to keep only ctrees with gold EDUs, i.e. at even indices
    # nltk_ctrees = nltk_ctrees[::2]
    #
    # for each doc, create an RSTTree from the NLTK tree
    for doc_name, nltk_ct_pred in zip(doc_names, nltk_ctrees):
        # the c-tree read corresponds to a SimpleRstTree
        sct_pred = tree_to_simple_rsttree(nltk_ct_pred)
        ct_pred = SimpleRSTTree.to_binary_rst_tree(sct_pred)
        ct_pred = rel_conv(ct_pred)
        ctree_pred[doc_name] = ct_pred
    return ctree_pred


def load_rs22_ctrees(log_file, rel_conv):
    """Load c-trees output by Alexeeva's parser.

    Parameters
    ----------
    log_file : str
        Path to the log file with the document names followed by the
        reference and predicted c-trees.

    rel_conv : RstRelationConverter
        Converter to map fine-grained relation labels to classes.

    Returns
    -------
    ctree_pred : dict(str, RSTTree)
        Predicted c-tree for each doc.
    """
    with open(log_file, mode='r', encoding='utf-8') as f:
        return _load_rs22_ctrees(f, rel_conv)


def load_rs22_dtrees(log_file, rel_conv, nary_enc='chain',
                     ctree_pred=None):
    """Get the dtrees for the ctrees output by Alexeeva's parser.

    Parameters
    ----------
    log_file: str
        Path to the log file with the output.
    rel_conv: TODO
        Relation converter, from fine- to coarse-grained labels.
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
    dtree_pred = dict()
    if ctree_pred is None:
        ctree_pred = load_rs22_ctrees(log_file, rel_conv)
    for doc_name, ct_pred in ctree_pred.items():
        dtree_pred[doc_name] = RstDepTree.from_rst_tree(
            ct_pred, nary_enc=nary_enc)
    # set reference to the document in the RstDepTree (required by
    # dump_disdep_files)
    for doc_name, dt_pred in dtree_pred.items():
        dt_pred.origin = FileId(doc_name, None, None, None)

    return dtree_pred
