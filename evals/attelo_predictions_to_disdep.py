"""Small utility script to convert predictions from attelo to dis_dep files.
"""

from __future__ import absolute_import, print_function

from collections import defaultdict
from glob import glob
import os

from attelo.io import load_edus, load_predictions
from attelo.metrics.util import barebones_rst_deptree
from attelo.table import UNRELATED
from educe.corpus import FileId
from educe.learning.disdep_format import dump_disdep_files
from educe.rst_dt.dep2con import (DummyNuclearityClassifier,
                                  InsideOutAttachmentRanker)


def attelo_predictions_to_disdep_files(edus_file_glob, edges_file, out_dir,
                                       nary_enc_pred='tree'):
    """Generate disdep files from a file dump of attelo predictions.

    Parameters
    ----------
    edus_file_glob: str
        Regex for `edu_input` file paths.
    edges_file: str
        Path to the file that contains attelo predictions (edges as
        triples).
    out_dir: str
        Path to the output folder.
    nary_enc_pred: one of {'chain', 'tree'}
        Encoding for n-ary cnodes in the predicted dtree ; here it
        currently triggers the strictness of the order assumed by the
        dtree postprocessor: nary_enc_pred='chain' implies order='strict',
        nary_enc_pred='tree' implies order='weak'.
    """
    order = 'weak' if nary_enc_pred == 'tree' else 'strict'
    # set up heuristic classifiers for nuclearity and rank
    nuc_clf = DummyNuclearityClassifier(strategy='unamb_else_most_frequent')
    nuc_clf.fit([], [])  # dummy fit
    rnk_clf = InsideOutAttachmentRanker(strategy='sdist-edist-rl',
                                        prioritize_same_unit=True,
                                        order=order)
    rnk_clf.fit([], [])  # dummy fit

    # load EDUs
    doc_edus = dict()
    id2doc = dict()
    for edu_input_file in glob(edus_file_glob):
        doc_name = os.path.basename(edu_input_file).rsplit('.', 4)[0]  # FRAGILE
        edus = load_edus(edu_input_file)
        assert doc_name == edus[0].grouping
        # map doc_name to list of EDUs ; populate reverse mapping from
        # EDU id to doc_name, so that we can dispatch edges to their
        # document
        # we keep the list of EDUs sorted as in edu_input, hence we
        # assume edu_input follows the linear order of EDUs
        doc_edus[doc_name] = edus
        for edu in edus:
            id2doc[edu.id] = doc_name
    # load edges and dispatch them to their doc
    edges_pred = load_predictions(edges_file)
    # for each doc, list edges
    doc_edges = defaultdict(list)
    for gov_id, dep_id, lbl in edges_pred:
        if lbl != UNRELATED:
            doc_name = id2doc[dep_id]
            doc_edges[doc_name].append((gov_id, dep_id, lbl))

    # for each doc, get a full-fledged RstDepTree, nuclearity and ranking
    # are currently determined heuristically
    doc_dtree = dict()
    for doc_name, edus in doc_edus.items():
        # comply with current API for barebones_rst_deptree:
        # for each doc, create a dict with one item (doc_name, list of edges)
        dep_edges = doc_edges[doc_name]
        # create a barebones RST dep tree: head and label only
        dtree, edu2sent = barebones_rst_deptree(dep_edges, edus, strict=False)
        # set its origin
        dtree.origin = FileId(doc_name, None, None, None)
        # flesh out with heuristically-determined nuclearity
        dtree.nucs = nuc_clf.predict([dtree])[0]
        # and heuristically-determined rank (needs edu2sent to prioritize
        # intra-sentential attachments over inter-sentential ones)
        dtree.sent_idx = edu2sent  # DIRTY
        dtree.ranks = rnk_clf.predict([dtree])[0]
        doc_dtree[doc_name] = dtree

    # write the disdep files
    dump_disdep_files(doc_dtree.values(), out_dir)


if __name__ == '__main__':
    nary_enc_pred = 'tree'
    edus_file_glob = os.path.join('TMP', 'latest', 'data', 'TEST',
                                  '*.edu-pairs.sparse.edu_input')
    edges_file_glob = os.path.join(
        'TMP', 'latest', 'scratch-current',
        'combined',
        # 'output.*'
        'output.maxent-iheads-global-AD.L-jnt-eisner'
    )
    # attelo predictions are currently stored in one big file
    edges_files = glob(edges_file_glob)
    assert len(edges_files) == 1
    edges_file = edges_files[0]
    # paths to the resulting disdep files
    out_dir = os.path.join('TMP_disdep', nary_enc_pred, 'ours', 'test')
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    # do the conversion
    attelo_predictions_to_disdep_files(edus_file_glob, edges_file, out_dir,
                                       nary_enc_pred=nary_enc_pred)
