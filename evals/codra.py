"""This module enables to load the output of Joty's discourse parser CODRA.

"""

import codecs
from collections import defaultdict
import glob
import itertools
import os

from educe.rst_dt.deptree import RstDepTree
from educe.rst_dt.parse import parse_rst_dt_tree


def load_codra_output_files(container_path, level='doc'):
    """Load ctrees output by CODRA on the TEST section of RST-WSJ.

    Parameters
    ----------
    container_path: string
        Path to the main folder containing CODRA's output

    level: {'doc', 'sent'}, optional (default='doc')
        Level of decoding: document-level or sentence-level

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
    if level == 'doc':
        file_ext = '.doc_dis'
    elif level == 'sent':
        file_ext = '.sen_dis'
    else:
        raise ValueError("level {} not in ['doc', 'sent']".format(level))

    # find all files with the right extension
    pathname = os.path.join(container_path, '*{}'.format(file_ext))
    # filenames are sorted by name to avoid having to realign data
    # loaded with different functions
    filenames = sorted(glob.glob(pathname))  # glob.glob() returns a list

    # find corresponding doc names
    doc_names = [os.path.splitext(os.path.basename(filename))[0] + '.out'
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


def load_codra_ctrees(codra_out_dir, rel_conv):
    """Load the ctrees output by CODRA as .dis files.

    This currently runs on the document-level files (.doc_dis).

    Parameters
    ----------
    codra_out_dir: str
        Path to the base directory containing the output files.

    Returns
    -------
    ctree_pred: dict(str, RSTTree)
        RST ctree for each document.
    """
    # load predicted trees
    data_pred = load_codra_output_files(codra_out_dir)
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
        ctree_pred[doc_name] = ct_pred

    return ctree_pred


def load_codra_dtrees(codra_out_dir, rel_conv, nary_enc='chain',
                      ctree_pred=None):
    """Get the dtrees that correspond to the ctrees output by CODRA.

    Parameters
    ----------
    codra_out_dir: str
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
        data_pred = load_codra_output_files(codra_out_dir)
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


# TODO move this generic util to a more appropriate place.
# This implementation is quite ad-hoc, tailored for RST e.g. to retrieve
# the edu_num, so I would need to generalize this code first.
def get_edu2sent(att_edus):
    """Get edu2sent mapping, from a list of attelo EDUs.

    Parameters
    ----------
    att_edus: list of attelo EDUs
        List of attelo EDUs, as produced by `load_edus`.

    Returns
    -------
    doc_name2edu2sent: dict(str, [int])
        For each document, get the sentence index for every EDU.

    Example:
    ```
    att_edus = load_edus(edus_file)
    doc_name2edu2sent = get_edu2sent(att_edus)
    for doc_name, edu2sent in doc_name2edu2sent.items():
        dtree[doc_name].edu2sent = edu2sent
    ```

    """
    edu2sent_idx = defaultdict(dict)
    for att_edu in att_edus:
        doc_name = att_edu.grouping
        edu_num = int(att_edu.id.rsplit('_', 1)[1])
        sent_idx = int(att_edu.subgrouping.split('_sent')[1])
        edu2sent_idx[doc_name][edu_num] = sent_idx
    # sort EDUs by num
    # rebuild educe-style edu2sent ; prepend 0 for the fake root
    doc_name2edu2sent = {
        doc_name: ([0] +
                   [s_idx for e_num, s_idx in sorted(edu2sent.items())])
        for doc_name, edu2sent in edu2sent_idx.items()
    }
    return doc_name2edu2sent
