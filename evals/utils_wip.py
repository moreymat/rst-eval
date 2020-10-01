"""Various utility functions that are WIP.

These functions are expected to move to educe or attelo when they
are mature.
"""

from __future__ import print_function

import os
import sys

from educe.rst_dt.annotation import RSTTree
from educe.rst_dt.corpus import Reader as RstReader
from educe.rst_dt.dep2con import deptree_to_simple_rst_tree
from educe.rst_dt.deptree import RstDepTree, RstDtException
#
from evals.ours import load_attelo_output_file


# RST corpus
CORPUS_DIR = os.path.abspath(os.path.join(
    os.path.dirname(os.path.realpath(__file__)),
    '..', 'corpus',
    'RSTtrees-WSJ-main-1.0/'))
CD_TRAIN = os.path.join(CORPUS_DIR, 'TRAINING')
CD_TEST = os.path.join(CORPUS_DIR, 'TEST')

# move to educe.rst_dt.datasets.rst_dis_format
STR_ROOT = '{nuc} (span {edu_span})'
STR_NODE = '{nuc} (span {edu_span}) (rel2par {rel})'
STR_LEAF = '{nuc} (leaf {edu_num}) (rel2par {rel}) (text _!{edu_txt}_!)'


def _str_node(tree):
    """String for the top node of an RSTTree

    Parameters
    ----------
    tree: educe.rst_dt.annotation.RSTTree
        The tree whose top node we want to print
    """
    node = tree.label()
    # get fields
    nuc = node.nuclearity
    edu_span = node.edu_span
    rel = node.rel
    # leaf (in reality, we are at the pre-terminal)
    if len(tree) == 1:
        # get text from the real leaf (EDU)
        txt = tree[0].text()
        node_str = STR_LEAF.format(nuc=nuc, edu_num=edu_span[0],
                                   rel=rel, edu_txt=txt)
    # internal node
    else:
        edu_span_str = '{} {}'.format(str(edu_span[0]), str(edu_span[1]))
        node_str = STR_NODE.format(nuc=nuc, edu_span=edu_span_str,
                                   rel=rel)

    return node_str


def tree_str_gen(tree):
    """Return a generator of strings, one per tree node"""
    # init tree stack with the whole tree, nesting level 0
    tree_stack = [(tree, 0)]

    while tree_stack:
        tree, lvl = tree_stack.pop()
        yield '{lw}{node_str}'.format(lw='  ' * lvl,
                                      node_str=_str_node(tree))
        tree_stack.extend(reversed([(subtree, lvl + 1) for subtree in tree
                                    if isinstance(subtree, RSTTree)]))
    # RESUME HERE: add opening (easy) and closing (trickier) parentheses
    # TODO do not print relation (None) for ROOT


def _dump_rst_dis_file(out_file, ct_pred):
    """Actually do dump.

    Parameters
    ----------
    out_file: File
        Output file

    ct_pred: RSTTree
        Binary RST tree
    """
    res_str = '\n'.join(tree_str_gen(ct_pred))  # or str(ct_pred) ?
    out_file.write(res_str)


def dump_rst_dis_file(out_file, ctree):
    """Dump a binary RST tree to a file.

    Parameters
    ----------
    out_file: string
        Path to the output file

    ctree: RSTTree
        Binary RST tree
    """
    with open(out_file, 'w') as f:
        _dump_rst_dis_file(f, ctree)
# end educe.rst_dt.datasets.rst_dis_format


# move to educe.rst_dt.datasets.dep_dis_format ?
def dump_dep_dis_file(out_file, dtree):
    """Dump a (RST) dependency tree to a file.

    Parameters
    ----------
    out_file: string
        Path to the output file

    dtree: RstDepTree
        RST dependency tree
    """
    with open(out_file, 'w') as f:
        res = '\n'.join('{}\t{}'.format(hd, lbl)
                        for hd, lbl in zip(dtree.heads, dtree.labels))
        f.write(res)
# end attelo.datasets.dep_dis_format


# move to educe.rst_dt.attelo_out_format
#
# this function is only called by `convert_attelo_output_file_to_dis_files`
#
# FIXME: find ways to read the right (not necessarily TEST) section
# and only the required documents
def load_trees_from_attelo_output_file(att_output_file):
    """Load predicted RST trees from attelo's output file.

    Parameters
    ----------
    att_output_file: string
        Path to the file that contains attelo's output

    Returns
    -------
    ctrees_pred: dict(string, SimpleRSTTree)
        Predicted SimpleRSTTree for each document, indexed by its name
    """
    # get predicted tree for each doc
    # these currently come in the form of edges on attelo EDUs
    edges_pred = load_attelo_output_file(att_output_file)

    # get educe EDUs
    edus = dict()
    # FIXME: parameterize this, cf. function-wide FIXME above
    rst_reader = RstReader(CD_TEST)
    rst_corpus = rst_reader.slurp()
    for doc_id, rtree_true in sorted(rst_corpus.items()):
        doc_name = doc_id.doc
        edus[doc_name] = rtree_true.leaves()

    # re-build predicted trees from predicted edges and educe EDUs
    dtree_pred = dict()  # predicted dtrees
    ctree_pred = dict()  # predicted ctrees
    skipped_docs = set()  # docs skipped because non-projective structures
    for doc_name, es_pred in sorted(edges_pred.items()):
        # map from EDU id to EDU num
        # EDU id should be common to educe and attelo
        id2num = {edu.identifier(): edu.num for edu in edus[doc_name]}
        # create pred dtree
        dt_pred = RstDepTree(edus[doc_name])
        for src_id, tgt_id, lbl in es_pred:
            if src_id == 'ROOT':
                if lbl == 'ROOT':
                    dt_pred.set_root(id2num[tgt_id])
                else:
                    raise ValueError('Weird root label: {}'.format(lbl))
            else:
                dt_pred.add_dependency(id2num[src_id], id2num[tgt_id], lbl)
        dtree_pred[doc_name] = dt_pred
        # create pred ctree
        try:
            ctree_pred[doc_name] = deptree_to_simple_rst_tree(dt_pred)
        except RstDtException:
            skipped_docs.add(doc_name)
            if False:
                print('\n'.join('{}: {}'.format(edu.text_span(), edu)
                                for edu in edus[doc_name]))
            # raise
    if skipped_docs:
        print('Skipped {} docs over {}'.format(len(skipped_docs),
                                               len(edges_pred)))

    return ctree_pred
# end educe.rst_dt.attelo_out_format


# move to educe.datasets.rst_dis_format
def convert_attelo_output_file_to_dis_files(output_dir, att_output_file):
    """Convert attelo's output file to a set of dis files in output_dir.

    Parameters
    ----------
    output_dir: string
        Path of the directory for the dis files
    output_file: string
        Path to the file that contains attelo's output

    Returns
    -------
    ctrees_pred: dict(string, SimpleRSTTree)
        Predicted SimpleRSTTree for each document, indexed by its name
    """
    if not os.path.exists(output_dir):
        raise ValueError('Absent path: {}'.format(output_dir))

    ctree_pred = load_trees_from_attelo_output_file(att_output_file)
    # output each SimpleRSTTree to a dis file
    for doc_name, ct_pred in ctree_pred.items():
        out_fname = os.path.join(output_dir, doc_name + '.dis')
        dump_rst_dis_file(out_fname, ct_pred)
        # DEBUG
        sys.exit()
# end educe.datasets.rst_dis_format


# ??
def load_gold():
    """Load gold structures from RST-WSJ/TEST.

    Returns
    -------
    data: dictionary that should be akin to a sklearn Bunch,
        with interesting keys 'filenames', 'doc_names', 'rst_ctrees',
        'rst_dtrees'.
    """
    # TODO make this the only place where the gold is loaded
    # shared between evals of both CODRA and attelo's outputs
    filenames = []  # TODO
    # load doc names and reference trees
    rst_reader = RstReader(CD_TEST)
    rst_corpus = rst_reader.slurp()
    doc_names = []
    rst_ctrees = []
    for doc_id, rst_ctree in sorted(rst_corpus.items(),
                                    key=lambda kv: kv[0].doc):
        doc_names.append(doc_id.doc)
        rst_ctrees.append(rst_ctree)
    # RESUME HERE (or not)
    raise NotImplementedError
# end ??
