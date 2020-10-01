"""This utility script outputs a dataset of the relation of RST edges.

Given the path to the RST-DT corpus and a dataset of candidate RST
dependencies labelled with their gold coarse (class) RST relation (or
none if they are unrelated), produce a filtered version of the dataset
for the task of relation labelling.

As of 2017-12-14, we filter out the instances for unrelated pairs of EDUs
and dependencies headed by the fake root.
The resulting dataset describes a n-ary classification problem whose
labelset is the set of (coarse-grained) classes of RST relations.
"""

from __future__ import absolute_import, print_function

import argparse
import codecs
import itertools
import os

from educe.rst_dt.annotation import NUC_N, NUC_S
from educe.rst_dt.corpus import RstRelationConverter, RELMAP_112_18_FILE
from educe.rst_dt.dep_corpus import read_corpus
from educe.rst_dt.deptree import RstDepTree


def main(corpus, dataset, out_dir, nary_enc, model_split):
    """Do prepare the RST relation dataset.

    Parameters
    ----------
    corpus : str
        Path to the RST-DT "main" corpus.
    dataset : str
        Path to the existing dataset labelled with coarse relations.
    out_dir : str
        Path to the output folder.
    model_split : str, one of {'none', 'sent', 'sent-para'}
        If not 'none', use distinct models for subsets of instances:
        * 'sent': intra- vs inter-sentential,
        * 'sent-para': intra-sentential, intra-paragraph, rest (doc-level).
    """
    # (re-)create a d-corpus from the RST-DT c-corpus
    corpus_subset = os.path.basename(dataset).split('.')[0]
    if corpus_subset not in ('TRAINING', 'TEST'):
        raise ValueError("dataset must be a filepath that starts with"
                         "one of {'TRAINING', 'TEST'}")
    if corpus_subset == 'TRAINING':
        section = 'train'
    else:  # 'TEST'
        section = 'test'
    rst_ccorpus = read_corpus(corpus, section=section)
    rel_conv = RstRelationConverter(RELMAP_112_18_FILE).convert_dtree
    rst_dcorpus = dict()  # FileId.doc -> RstDepTree
    for doc_key, rst_ctree in rst_ccorpus[section].items():
        rst_dtree = RstDepTree.from_rst_tree(rst_ctree, nary_enc=nary_enc)
        rst_dtree_coarse = rel_conv(rst_dtree)
        rst_dcorpus[doc_key.doc] = rst_dtree_coarse
    # for each candidate dependency in the dataset, read the nuclearity
    # from the RST d-corpus
    # Nota: we stream through the dataset to avoid loading it entirely in
    # memory ; we don't need to open the vocabulary file (.vocab), nor the
    # description of the EDUs (.edu_input)
    pairings = dataset + '.pairings'
    # edu_desc = dataset + '.edu_input'
    if model_split == 'none':
        new_dataset = os.path.join(out_dir, os.path.basename(dataset))
        new_pairs = os.path.join(out_dir, os.path.basename(pairings))
        if ((os.path.abspath(new_dataset) == os.path.abspath(dataset) or
             os.path.abspath(new_pairs) == os.path.abspath(pairings))):
            raise ValueError("I won't let you erase your base dataset")
        with codecs.open(dataset, mode='rb', encoding='utf-8') as f_data:
            with codecs.open(pairings, mode='rb', encoding='utf-8') as f_pairs:
                with codecs.open(new_dataset, mode='wb', encoding='utf-8') as data_out:
                    with codecs.open(new_pairs, mode='wb', encoding='utf-8') as pairs_out:
                        # read header line in svmlight file
                        header = f_data.readline()
                        header_prefix = '# labels: '
                        assert header.startswith(header_prefix)
                        labels = header[len(header_prefix):].split()
                        int2lbl = dict(enumerate(labels, start=1))
                        lbl2int = {lbl: i for i, lbl in int2lbl.items()}
                        unrelated = lbl2int["UNRELATED"]
                        root = lbl2int["ROOT"]
                        # write labels in header of new svmlight file, here
                        # we just copy the existing header (even if it has
                        # ROOT and UNRELATED that should never appear here)
                        print(header, file=data_out)
                        # stream through lines
                        for pair, line in itertools.izip(f_pairs, f_data):
                            # read candidate pair of EDUs
                            src_id, tgt_id = pair.strip().split('\t')
                            if src_id == 'ROOT':
                                continue
                            # now both src_id and tgt_id are of form "docname_int"
                            # ex: "wsj_0600.out_1"
                            src_idx = int(src_id.rsplit('_', 1)[1])
                            doc_name, tgt_idx = tgt_id.rsplit('_', 1)
                            tgt_idx = int(tgt_idx)
                            # read corresponding ref class (label), feature vector
                            lbl_idx, feat_vector = line.strip().split(' ', 1)
                            lbl_idx = int(lbl_idx)  # lbl currently encoded as int
                            if lbl_idx in (unrelated, root):
                                continue
                            try:
                                lbl = int2lbl[lbl_idx]
                            except KeyError:
                                # the test set in RST-DT 1.0 has an error:
                                # wsj_1189.out [8-9] is labelled "span" instead of
                                # "Consequence" ; some runs used this erroneous
                                # version, hence had a class "0" (unknown) for
                                # this line in the dataset
                                if ((doc_name == 'wsj_1189.out' and
                                     src_idx == 7 and
                                     tgt_idx == 9)):
                                    lbl = 'cause'
                                    lbl_idx = lbl2int[lbl]
                                else:
                                    print(doc_name, src_idx, tgt_idx)
                                    raise
                            # print(src_id, tgt_id, lbl)
                            dtree = rst_dcorpus[doc_name]
                            assert dtree.heads[tgt_idx] == src_idx
                            assert dtree.labels[tgt_idx] == lbl
                            print(str(lbl_idx) + ' ' + feat_vector,
                                  file=data_out)
                            print(pair.strip(), file=pairs_out)
    elif model_split == 'sent':
        # 2 datasets: intra- and inter-sentential
        new_dataset = (
            os.path.join(out_dir + '_intrasent', os.path.basename(dataset)),
            os.path.join(out_dir + '_intersent', os.path.basename(dataset))
        )
        new_pairs = (
            os.path.join(out_dir + '_intrasent', os.path.basename(pairings)),
            os.path.join(out_dir + '_intersent', os.path.basename(pairings))
        )
        if ((os.path.abspath(new_dataset[0]) == os.path.abspath(dataset) or
             os.path.abspath(new_pairs[0]) == os.path.abspath(pairings) or
             os.path.abspath(new_dataset[1]) == os.path.abspath(dataset) or
             os.path.abspath(new_pairs[1]) == os.path.abspath(pairings))):
            raise ValueError("I won't let you erase your base dataset")
        with codecs.open(dataset, mode='rb', encoding='utf-8') as f_data:
            with codecs.open(pairings, mode='rb', encoding='utf-8') as f_pairs:
                with codecs.open(new_dataset[0], mode='wb', encoding='utf-8') as data_out_intra:
                    with codecs.open(new_pairs[0], mode='wb', encoding='utf-8') as pairs_out_intra:
                        with codecs.open(new_dataset[1], mode='wb', encoding='utf-8') as data_out_inter:
                            with codecs.open(new_pairs[1], mode='wb', encoding='utf-8') as pairs_out_inter:
                                # read header line in svmlight file
                                header = f_data.readline()
                                header_prefix = '# labels: '
                                assert header.startswith(header_prefix)
                                labels = header[len(header_prefix):].split()
                                int2lbl = dict(enumerate(labels, start=1))
                                lbl2int = {lbl: i for i, lbl in int2lbl.items()}
                                unrelated = lbl2int["UNRELATED"]
                                root = lbl2int["ROOT"]
                                # write labels in header of new svmlight file
                                print(header, file=data_out_intra)
                                print(header, file=data_out_inter)
                                # stream through lines
                                for pair, line in itertools.izip(f_pairs, f_data):
                                    # read candidate pair of EDUs
                                    src_id, tgt_id = pair.strip().split('\t')
                                    if src_id == 'ROOT':
                                        continue
                                    # now both src_id and tgt_id are of form "docname_int"
                                    # ex: "wsj_0600.out_1"
                                    src_idx = int(src_id.rsplit('_', 1)[1])
                                    doc_name, tgt_idx = tgt_id.rsplit('_', 1)
                                    tgt_idx = int(tgt_idx)
                                    # read corresponding ref class (label), feature vector
                                    lbl_idx, feat_vector = line.strip().split(' ', 1)
                                    lbl_idx = int(lbl_idx)  # lbl currently encoded as int
                                    if lbl_idx in (unrelated, root):
                                        continue
                                    try:
                                        lbl = int2lbl[lbl_idx]
                                    except KeyError:
                                        # the test set in RST-DT 1.0 has an error:
                                        # wsj_1189.out [8-9] is labelled "span" instead of
                                        # "Consequence" ; some runs used this erroneous
                                        # version, hence had a class "0" (unknown) for
                                        # this line in the dataset
                                        if ((doc_name == 'wsj_1189.out' and
                                             src_idx == 7 and
                                             tgt_idx == 9)):
                                            lbl = 'cause'
                                            lbl_idx = lbl2int[lbl]
                                        else:
                                            print(doc_name, src_idx, tgt_idx)
                                            raise
                                    # print(src_id, tgt_id, lbl)
                                    dtree = rst_dcorpus[doc_name]
                                    assert dtree.heads[tgt_idx] == src_idx
                                    assert dtree.labels[tgt_idx] == lbl
                                    if ((' 269:' in feat_vector or
                                         ' 303:' in feat_vector) and
                                        (' 103:' in feat_vector or
                                         ' 158:' in feat_vector or
                                         ' 234:' in feat_vector or
                                         ' 314:' in feat_vector)):
                                        # 269 is same_sentence_intra_right
                                        # 303 is same_sentence_intra_left ;
                                        # 103 is same_para_inter_right
                                        # 158 is same_para_inter_left
                                        # 234 is same_para_intra_right
                                        # 314 is same_para_intra_left
                                        # FIXME find a cleaner way
                                        print(str(lbl_idx) + ' ' + feat_vector,
                                              file=data_out_intra)
                                        print(pair.strip(),
                                              file=pairs_out_intra)
                                    else:
                                        # inter-sentential
                                        print(str(lbl_idx) + ' ' + feat_vector,
                                              file=data_out_inter)
                                        print(pair.strip(),
                                              file=pairs_out_inter)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Prepare a relation dataset.'
    )
    parser.add_argument('--corpus',
                        help='Path to the RST-DT "main" corpus',
                        default=os.path.join(
                            os.path.expanduser('~'),
                            'corpora/rst-dt/rst_discourse_treebank/data',
                            'RSTtrees-WSJ-main-1.01'
                        ))
    parser.add_argument('--dataset',
                        help='Base file of the dataset',
                        default=os.path.join(
                            os.path.expanduser('~'),
                            'melodi/rst/irit-rst-dt/TMP/syn_pred_coarse',
                            'TRAINING.relations.sparse'
                        ))
    parser.add_argument('--out_dir',
                        help='Output folder',
                        default=os.path.join(
                            os.path.expanduser('~'),
                            'melodi/rst/irit-rst-dt/TMP/syn_pred_coarse_REL'
                        ))
    parser.add_argument('--nary_enc',
                        help='Encoding for n-ary nodes',
                        choices=['chain', 'tree'],
                        default='chain')
    parser.add_argument('--model_split',
                        help='Separate models for subsets of instances',
                        choices=['none', 'sent', 'sent-para'],
                        default='none')
    args = parser.parse_args()
    main(args.corpus, args.dataset, args.out_dir, args.nary_enc,
         args.model_split)
