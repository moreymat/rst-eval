"""Load the output of Ji's DPLP parser.

"""

from __future__ import absolute_import, print_function

from collections import defaultdict
from glob import glob
import os

from educe.annotation import Span
from educe.corpus import FileId
from educe.rst_dt.annotation import Node, RSTTree
from educe.rst_dt.deptree import RstDepTree


def load_ji_ctrees(ji_out_dir, rel_conv, doc_edus):
    """Load the ctrees output by DPLP as .brackets files.

    Parameters
    ----------
    ji_out_dir : str
        Path to the base directory containing the output files.
    rel_conv : RstRelationConverter?
        Relation converter.
    doc_edus : dict(str, list(EDU))
        Mapping from doc_name to the list of its EDUs (read from the
        corpus).

    Returns
    -------
    ctree_pred: dict(str, RSTTree)
        RST ctree for each document.
    """
    # FIXME? get the text of EDUs from the .merge files?
    # * for each doc, load the predicted spans from the .brackets
    ctree_pred = dict()
    files_pred = os.path.join(ji_out_dir, '*.brackets')
    for f_pred in sorted(glob(files_pred)):
        doc_name = os.path.splitext(os.path.basename(f_pred))[0]
        edus = {i: e for i, e in enumerate(doc_edus[doc_name], start=1)}
        origin = FileId(doc_name, None, None, None)
        # read spans
        spans_pred = defaultdict(list)  # predicted spans by length
        with open(f_pred) as f:
            for line in f:
                # FIXME use a standard module: ast? pickle?
                # * drop surrounding brackets + opening bracket of edu span
                line = line.strip()[2:-1]
                edu_span, nuc_rel = line.split('), ')
                edu_span = tuple(int(x) for x in edu_span.split(', '))
                nuc, rel = nuc_rel.split(', ')
                # * remove quotes around nuc and rel
                nuc = nuc[1:-1]
                rel = rel[1:-1]
                #
                edu_span_len = edu_span[1] - edu_span[0]
                spans_pred[edu_span_len].append((edu_span, nuc, rel))
        # bottom-up construction of the RST ctree
        # left_border -> list of RST ctree fragments, sorted by len
        tree_frags = defaultdict(list)
        for span_len, spans in sorted(spans_pred.items()):
            for edu_span, nuc, rel in spans:
                children = []
                edu_beg, edu_end = edu_span
                if edu_beg == edu_end:
                    # pre-terminal
                    txt_span = edus[edu_beg].span
                    # one child: leaf node: EDU
                    leaf = edus[edu_beg]
                    children.append(leaf)
                else:
                    # internal node
                    # * get the children (subtrees)
                    edu_cur = edu_beg
                    while edu_cur <= edu_end:
                        kid_nxt = tree_frags[edu_cur][-1]
                        children.append(kid_nxt)
                        edu_cur = kid_nxt.label().edu_span[1] + 1
                    # compute properties of this node
                    txt_span = Span(children[0].label().span.char_start,
                                    children[-1].label().span.char_end)
                # build node and RSTTree fragment
                node = Node(nuc, edu_span, txt_span, rel,
                            context=None)  # TODO context?
                tree_frags[edu_beg].append(
                    RSTTree(node, children, origin=origin))
        # build the top node
        edu_nums = sorted(edus.keys())
        edu_span = (edu_nums[0], edu_nums[-1])
        children = []
        edu_beg, edu_end = edu_span
        edu_cur = edu_beg
        while edu_cur <= edu_end:
            kid_nxt = tree_frags[edu_cur][-1]
            children.append(kid_nxt)
            edu_cur = kid_nxt.label().edu_span[1] + 1
        txt_span = Span(children[0].label().span.char_start,
                        children[-1].label().span.char_end)
        node = Node(nuc, edu_span, txt_span, 'Root', context=None)
        tree_frags[edu_beg].append(
            RSTTree(node, children, origin=origin))
        # now we should have a spanning ctree
        ct_pred = tree_frags[1][-1]
        assert ct_pred.label().edu_span == (sorted(edus.keys())[0],
                                            sorted(edus.keys())[-1])
        # convert relation labels
        if rel_conv is not None:
            ct_pred = rel_conv(ct_pred)
            # normalize names of classes of RST relations:
            # "same_unit" => "same-unit"
            # "topic" => "topic-change" or "topic-comment"?
            for pos in ct_pred.treepositions():
                t = ct_pred[pos]
                if isinstance(t, RSTTree):
                    node = t.label()
                    # replace "same_unit" with "same-unit"
                    if node.rel == 'same_unit':  # DPLP v. 1
                        node.rel = 'same-unit'
                    elif node.rel == 'topic':  # DPLP v. 1
                        # either "topic-comment" or "topic-change" ;
                        # I expect the parser to find "topic-comment" to
                        # be easier but apparently it has no consequence
                        # on the current output I reproduced
                        node.rel = 'topic-comment'
                    elif node.rel == 'sameunit':  # Ji's output
                        node.rel = 'same-unit'
                    elif node.rel == 'topicchange':  # Ji's output
                        node.rel = 'topic-change'
                    elif node.rel == 'topiccomment':  # Ji's output
                        node.rel = 'topic-comment'
                    elif node.rel == 'textual-organization':  # WLW17 output
                        # we use 'textual' as the coarse label ;
                        # JE14 outputs textualorganization which is the
                        # fine label in our taxonomy, hence is mapped to
                        # textual beforehand
                        node.rel = 'textual'
            # end normalize
        # store the resulting RSTTree
        ctree_pred[doc_name] = ct_pred

    return ctree_pred


def load_ji_dtrees(ji_out_dir, rel_conv, doc_edus, nary_enc='chain',
                   ctree_pred=None):
    """Get the dtrees that correspond to the ctrees output by DPLP.

    Parameters
    ----------
    ji_out_dir: str
        Path to the base directory containing the output files.
    rel_conv: TODO
        Relation converter, from fine- to coarse-grained labels.
    nary_enc: one of {'chain', 'tree'}
        Encoding for n-ary nodes.
    doc_edus : dict(str, list(EDU))
        Mapping from doc_name to the list of its EDUs (read from the
        corpus).
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
        ctree_pred = load_ji_ctrees(ji_out_dir, rel_conv, doc_edus)
    for doc_name, ct_pred in ctree_pred.items():
        dtree_pred[doc_name] = RstDepTree.from_rst_tree(
            ct_pred, nary_enc=nary_enc)
    # set reference to the document in the RstDepTree (required by
    # dump_disdep_files)
    for doc_name, dt_pred in dtree_pred.items():
        dt_pred.origin = FileId(doc_name, None, None, None)

    return dtree_pred

