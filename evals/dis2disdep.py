"""Convert RST trees to their dependency version (.dis to .dis_dep).

TODO
----
* [ ] support intra-sentential level document parsing ; required to score
      Joty's .sen_dis files

"""
from __future__ import absolute_import, print_function
import argparse
import os

from educe.corpus import FileId
from educe.learning.disdep_format import dump_disdep_files
from educe.rst_dt.corpus import Reader, RstRelationConverter
from educe.rst_dt.deptree import RstDepTree
from educe.rst_dt.feng import load_feng_output_files
from educe.rst_dt.rst_wsj_corpus import (DOUBLE_FOLDER, TEST_FOLDER,
                                         TRAIN_FOLDER)

from evals.codra import load_codra_output_files
from evals.gcrf_tree_format import load_gcrf_dtrees
from evals.hayashi_cons import load_hayashi_hilda_dtrees
from evals.hayashi_deps import load_hayashi_dep_dtrees
from evals.ji import load_ji_dtrees
from evals.showdown import (setup_dtree_postprocessor, NUC_STRATEGY,
                            NUC_CONSTANT, RNK_STRATEGY, RNK_PRIORITY_SU)


# original RST corpus
RST_CORPUS = os.path.join('/home/mmorey/corpora/rst_discourse_treebank/data')
RST_MAIN_TRAIN = os.path.join(RST_CORPUS, TRAIN_FOLDER)
RST_MAIN_TEST = os.path.join(RST_CORPUS, TEST_FOLDER)
RST_DOUBLE = os.path.join(RST_CORPUS, DOUBLE_FOLDER)

# get edu2sent, set up rnk_clf and nuc_clf to predict rank and order for
# the output of Hayashi's MST parser
# * new style .edu_input: one file per doc in test set
EDUS_FILE_PAT = "TMP/latest/data/TEST/{}.relations.edu-pairs.sparse.edu_input"

# relation converter (fine- to coarse-grained labels)
RELMAP_FILE = os.path.join('/home/mmorey/melodi/educe',
                           'educe', 'rst_dt',
                           'rst_112to18.txt')
REL_CONV_BASE = RstRelationConverter(RELMAP_FILE)
REL_CONV = REL_CONV_BASE.convert_tree
REL_CONV_DTREE = REL_CONV_BASE.convert_dtree
# output of Joty's parser
OUT_JOTY = os.path.join('/home/mmorey/melodi/rst/joty/Doc-level/')
# output of Feng & Hirst's parsers
FENG_BASEDIR = '/home/mmorey/melodi/rst/feng_hirst'
OUT_FENG = os.path.join(FENG_BASEDIR, 'phil/tmp/')
OUT_FENG2 = os.path.join(FENG_BASEDIR,
                         'gCRF_dist/texts/results/test_batch_gold_seg')
# output of Ji's parser
JI_BASEDIR = '/home/mmorey/melodi/rst/ji_eisenstein'
OUT_JI = os.path.join(JI_BASEDIR, 'DPLP/data/docs/test/')
# output of Hayashi et al.'s parsers
HAYASHI_BASEDIR = '/home/mmorey/melodi/rst/hayashi/SIGDIAL/'
OUT_HAYASHI_MST = os.path.join(HAYASHI_BASEDIR, 'auto_parse/dep/li/')
OUT_HAYASHI_HILDA = os.path.join(HAYASHI_BASEDIR, 'auto_parse/cons/trans_li/')


def main():
    """Main"""
    parser = argparse.ArgumentParser(
        description='Convert .dis files to .dis_dep'
    )
    parser.add_argument('--nary_enc', default='chain',
                        choices=['chain', 'tree'],
                        help="Encoding for n-ary nodes")
    parser.add_argument('--author', default='gold',
                        choices=['gold', 'silver',
                                 'joty', 'feng', 'feng2', 'ji',
                                 'hayashi_hilda', 'hayashi_mst'],
                        help="Author of the version of the corpus")
    parser.add_argument('--split', default='test',
                        choices=['train', 'test', 'double'],
                        help="Relevant part of the corpus")
    parser.add_argument('--out_root', default='TMP_disdep',
                        help="Root directory for the output")
    args = parser.parse_args()
    # precise output path, by default: TMP_disdep/chain/gold/train
    out_dir = os.path.join(args.out_root, args.nary_enc, args.author,
                           args.split)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    # read RST trees
    nary_enc = args.nary_enc
    author = args.author
    corpus_split = args.split

    if author == 'gold':
        if corpus_split == 'train':
            corpus_dir = RST_MAIN_TRAIN
        elif corpus_split == 'test':
            corpus_dir = RST_MAIN_TEST
        elif corpus_split == 'double':
            raise NotImplementedError("Gold trees for 'double'")
        reader = Reader(corpus_dir)
        rtrees = reader.slurp()
        dtrees = {doc_name: RstDepTree.from_rst_tree(rtree, nary_enc=nary_enc)
                  for doc_name, rtree in rtrees.items()}
    elif author == 'silver':
        if corpus_split == 'double':
            corpus_dir = RST_DOUBLE
        else:
            raise ValueError("'silver' annotation is available for the "
                             "'double' split only")
    elif author == 'joty':
        if corpus_split != 'test':
            raise ValueError("The output of Joty's parser is available for "
                             "the 'test' split only")
        data_pred = load_codra_output_files(OUT_JOTY, level='doc')
        doc_names = data_pred['doc_names']
        rtrees = data_pred['rst_ctrees']
        dtrees = {doc_name: RstDepTree.from_rst_tree(rtree, nary_enc=nary_enc)
                  for doc_name, rtree in zip(doc_names, rtrees)}
        # set reference to the document in the RstDepTree (required by
        # dump_disdep_files)
        for doc_name, dtree in dtrees.items():
            dtree.origin = FileId(doc_name, None, None, None)
    elif author == 'feng':
        if corpus_split != 'test':
            raise ValueError("The output of Feng & Hirst's parser is "
                             "available for the 'test' split only")
        data_pred = load_feng_output_files(OUT_FENG)
        doc_names = data_pred['doc_names']
        rtrees = data_pred['rst_ctrees']
        dtrees = {doc_name: RstDepTree.from_rst_tree(rtree, nary_enc=nary_enc)
                  for doc_name, rtree in zip(doc_names, rtrees)}
        # set reference to the document in the RstDepTree (required by
        # dump_disdep_files)
        for doc_name, dtree in dtrees.items():
            dtree.origin = FileId(doc_name, None, None, None)

    elif author == 'feng2':
        if corpus_split != 'test':
            raise ValueError("The output of Feng & Hirst's parser is "
                             "available for the 'test' split only")
        dtrees = load_gcrf_dtrees(OUT_FENG2, REL_CONV)
        for doc_name, dtree in dtrees.items():
            dtree.origin = FileId(doc_name, None, None, None)

    elif author == 'ji':
        if corpus_split != 'test':
            raise ValueError("The output of Ji & Eisenstein's parser is "
                             "available for the 'test' split only")
        dtrees = load_ji_dtrees(OUT_JI, REL_CONV)
    elif author == 'hayashi_mst':
        if corpus_split != 'test':
            raise ValueError("The output of Hayashi et al.'s parser is "
                             "available for the 'test' split only")
        # setup nuc_clf, rnk_clf
        nuc_clf, rnk_clf = setup_dtree_postprocessor(
            nary_enc='tree', order='weak',
            nuc_strategy=NUC_STRATEGY,
            nuc_constant=NUC_CONSTANT,
            rnk_strategy=RNK_STRATEGY,
            rnk_prioritize_same_unit=RNK_PRIORITY_SU)
        # end setup
        dtrees = load_hayashi_dep_dtrees(
            OUT_HAYASHI_MST, REL_CONV_DTREE, EDUS_FILE_PAT,
            nuc_clf, rnk_clf)
    elif author == 'hayashi_hilda':
        if corpus_split != 'test':
            raise ValueError("The output of Hayashi et al.'s parser is "
                             "available for the 'test' split only")
        dtrees = load_hayashi_hilda_dtrees(OUT_HAYASHI_HILDA, REL_CONV)

    # do dump
    dump_disdep_files(dtrees.values(), out_dir)


if __name__ == '__main__':
    main()
