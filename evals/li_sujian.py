"""TODO

"""

from __future__ import absolute_import, print_function
import os

# educe
from educe.learning.edu_input_format import load_edu_input_file
from educe.rst_dt.dep2con import deptree_to_rst_tree
from educe.rst_dt.deptree import NUC_S, RstDepTree, RstDtException
from educe.rst_dt.metrics.rst_parseval import rst_parseval_report
# attelo
from attelo.metrics.deptree import compute_uas_las as att_compute_uas_las


# output of Li et al.'s parser
SAVE_DIR = "/home/mmorey/melodi/rst/replication/li_sujian/TextLevelDiscourseParser/mybackup/mstparser-code-116-trunk/mstparser/save"
COARSE_FILES = [
    "136.0detailedOutVersion2.txt",
    "151.0detailedOut.txt",
    "164.0detailedOut.txt",
    "177.0detailedOut.txt",
    "335.0detailedOut.txt",
    "37.0detailedOut.txt",
    "424.0detailedOut.txt",
    "448.0detailedOut.txt",
    "455.0detailedOutVersion2.txt",
    "513.0detailedOutVersion2.txt",
    "529.0detailedOut.txt",
    "615.0detailedOutVersion2.txt",
    "712.0detailedOut.txt",
    "917.0detailedOut.txt",
]
FINE_FILES = [
    "190.0detailedOut.txt",
    "473.0detailedOutVersion2.txt",
    "561.0detailedOut.txt",
    "723.0detailedOut.txt",
    "747.0detailedOutVersion2.txt",
    "825.0detailedOut.txt",
    "947.0detailedOut.txt",
    "965.0detailedOutVersion2.txt",
]
# different format for predicted labels and description of EDU
COARSE_FEAT_FILES = [
    "441.0detailedOut.txt",
]

# default file to include ; I picked a coarse-grained one with good scores
DEFAULT_FILE = os.path.join(SAVE_DIR, "712.0detailedOut.txt")


def load_output_file(out_file):
    """Load an output file from Li et al.'s dep parser.
    """
    doc_names = []
    heads_true = []
    labels_true = []
    heads_pred = []
    labels_pred = []
    with open(out_file) as f:
        for line in f:
            if line.startswith(".\\testdata"):
                # file
                doc_name = line.strip().split("\\")[2][:12]  # drop .edus or else
                # print(doc_name)
                doc_names.append(doc_name)
                heads_true.append([-1])  # initial pad for fake root
                labels_true.append([''])
                heads_pred.append([-1])
                labels_pred.append([''])
            else:
                edu_idx, hd_true, hd_pred, lbl_true, lbl_pred, edu_str = line.strip().split(' ', 5)
                if lbl_pred == '<no-type>':
                    # not sure whether this should be enabled
                    lbl_pred = 'Elaboration'
                heads_true[-1].append(int(hd_true))
                labels_true[-1].append(lbl_true)
                heads_pred[-1].append(int(hd_pred))
                labels_pred[-1].append(lbl_pred)
    res = {
        'doc_names': doc_names,
        'heads_true': heads_true,
        'labels_true': labels_true,
        'heads_pred': heads_pred,
        'labels_pred': labels_pred,
    }
    return res


def load_li_sujian_dep_dtrees(out_file, rel_conv_dtree, edus_file_pat,
                              nuc_clf, rnk_clf):
    """Load the dtrees output by Li Sujian et al.'s dep parser.

    Parameters
    ----------
    out_file : str
        Path to the file containing all the predictions.

    rel_conv_dtree : RstRelationConverter
        Converter to map relation labels to (normalized) coarse-grained
        classes.

    edus_file_pat : str
        Pattern for the .edu_input files.

    nuc_clf : NuclearityClassifier
        Nuclearity classifier

    rnk_clf : RankClassifier
        Rank classifier

    Returns
    -------
    dtree_pred : dict(str, RstDepTree)
        RST dtree for each doc.
    """
    dtree_pred = dict()

    dep_bunch = load_output_file(out_file)
    # load and process _pred
    for doc_name, heads_pred, labels_pred in zip(
            dep_bunch['doc_names'], dep_bunch['heads_pred'],
            dep_bunch['labels_pred']):
        # create dtree _pred
        edus_data = load_edu_input_file(edus_file_pat.format(doc_name),
                                        edu_type='rst-dt')
        edus = edus_data['edus']
        edu2sent = edus_data['edu2sent']
        dt_pred = RstDepTree(edus)
        # add predicted edges
        for dep_idx, (gov_idx, lbl) in enumerate(zip(
                heads_pred[1:], labels_pred[1:]), start=1):
            if lbl == '<no-type>':
                lbl = 'Elaboration'
            lbl = lbl.lower()
            dt_pred.add_dependency(gov_idx, dep_idx, lbl)
        # map to relation classes
        dt_pred = rel_conv_dtree(dt_pred)
        dt_pred.labels = ['ROOT' if x == 'root' else x
                          for x in dt_pred.labels]
        # attach edu2sent, for later use by rnk_clf
        dt_pred.sent_idx = [0] + edu2sent  # 0 for fake root + dirty
        dtree_pred[doc_name] = dt_pred
        # end WIP

    for doc_name in sorted(dtree_pred.keys()):
        dt_pred = dtree_pred[doc_name]
        # enrich d-tree with nuc and order
        dt_pred.ranks = rnk_clf.predict([dt_pred])[0]
        dt_pred.nucs = nuc_clf.predict([dt_pred])[0]
        dtree_pred[doc_name] = dt_pred

    return dtree_pred


def load_li_sujian_dep_ctrees(out_file, rel_conv_dtree, edus_file_pat,
                              nuc_clf, rnk_clf):
    """Load the ctrees for the dtrees output by Li Sujian et al.'s parser.

    Parameters
    ----------
    out_file : str
        Path to the file containing all the predictions.

    rel_conv_dtree : RstRelationConverter
        Converter to map relation labels to (normalized) coarse-grained
        classes.

    edus_file_pat : str
        Pattern for the .edu_input files.

    nuc_clf : NuclearityClassifier
        Nuclearity classifier

    rnk_clf : RankClassifier
        Rank classifier

    Returns
    -------
    ctree_pred : dict(str, RSTTree)
        RST ctree for each doc.
    """
    ctree_pred = dict()

    dtree_pred = load_li_sujian_dep_dtrees(
        out_file, rel_conv_dtree, edus_file_pat, nuc_clf, rnk_clf)
    for doc_name, dt_pred in sorted(dtree_pred.items()):
        ct_pred = deptree_to_rst_tree(dt_pred)
        ctree_pred[doc_name] = ct_pred
    return ctree_pred


def twisted_eval(out_file, rel_conv_dtree, setup_dtree_postprocessor,
                 ctree_true, dtree_true, edus_file_pat):
    """Perform a twisted eval.

    Parameters
    ----------
    setup_dtree_postprocessor : function
        Function that sets up nuc_clf and rnk_clf.

    ctree_true : dict(str, RSTTree)
        Gold ctrees

    dtree_true : dict(str, DepRstTree)
        Gold dtrees

    out_file : str
        Path to the output file.
    """
    # setup conversion from c- to d-tree and back, and eval type
    nary_enc = 'chain'
    # reconstruction of the c-tree
    order = 'strict'
    nuc_strategy = 'constant'
    nuc_constant = NUC_S
    rnk_strategy = 'lllrrr'
    rnk_prioritize_same_unit = False
    # eval
    add_trivial_spans = True

    nuc_clf, rnk_clf = setup_dtree_postprocessor(
        nary_enc=nary_enc, order=order, nuc_strategy=nuc_strategy,
        nuc_constant=nuc_constant, rnk_strategy=rnk_strategy,
        rnk_prioritize_same_unit=rnk_prioritize_same_unit)

    ctree_true = dict()
    dtree_true = dict()
    for doc_name, dt_true in sorted(dtree_true.items()):
        # dirty hack: lowercase ROOT
        dt_true.labels = [x.lower() if x == 'ROOT' else x
                          for x in dt_true.labels]

    # load parser output
    dtree_pred = load_li_sujian_dep_dtrees(
        out_file, rel_conv_dtree, edus_file_pat, nuc_clf, rnk_clf)
    ctree_pred = load_li_sujian_dep_ctrees(
        out_file, rel_conv_dtree, edus_file_pat, nuc_clf, rnk_clf)

    # use our heuristics to replace the true nuc and order in
    # dt_true with a predicted one, replace ct_true with its
    # twisted version
    for doc_name, dt_true in dtree_true.items():
        dt_pred = dtree_pred[doc_name]
        # twiste dt_true
        dt_true.sent_idx = dt_pred.sent_idx
        dt_true.ranks = rnk_clf.predict([dt_true])[0]
        dt_true.nucs = nuc_clf.predict([dt_true])[0]
        # re-gen ct_true
        try:
            ct_true = deptree_to_rst_tree(dt_true)
        except RstDtException as rst_e:
            print(rst_e)
            raise
        ctree_true[doc_name] = ct_true

    # compute UAS and LAS on the _true values from the corpus and
    # _pred Educe RstDepTrees re-built from their output files
    doc_names = sorted(dtree_true.keys())
    dtree_true_list = [dtree_true[doc_name] for doc_name in doc_names]
    dtree_pred_list = [dtree_pred[doc_name] for doc_name in doc_names]
    sc_uas, sc_las, sc_las_n, sc_las_o, sc_las_no = att_compute_uas_las(
        dtree_true_list, dtree_pred_list, include_ls=False,
        include_las_n_o_no=True)
    print(("{}\tUAS={:.4f}\tLAS={:.4f}\tLAS+N={:.4f}\tLAS+O={:.4f}\t"
           "LAS+N+O={:.4f}").format(
               out_file, sc_uas, sc_las, sc_las_n, sc_las_o, sc_las_no))

    # compute RST-Parseval of these c-trees
    ctree_true_list = [ctree_true[doc_name] for doc_name in doc_names]
    ctree_pred_list = [ctree_pred[doc_name] for doc_name in doc_names]
    print(rst_parseval_report(ctree_true_list, ctree_pred_list,
                              ctree_type='RST', digits=4,
                              per_doc=False,
                              add_trivial_spans=add_trivial_spans,
                              stringent=False))
