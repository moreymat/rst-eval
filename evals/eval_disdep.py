"""Evaluation procedure for discourse dependency (disdep) files.

Computes UAS and flavours of LAS for labels, nuclearity, rank and
their combinations.
"""

from __future__ import absolute_import, print_function
import argparse
import codecs
import csv
from glob import glob
import os


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Evaluate dis_dep trees against a given reference")
    parser.add_argument('authors_pred', nargs='+',
                        choices=['gold', 'silver',
                                 'joty', 'feng', 'feng2', 'ji',
                                 'hayashi_hilda', 'hayashi_mst',
                                 'ours'],
                        help="Author(s) of the predictions")
    parser.add_argument('--author_true', default='gold',
                        choices=['gold', 'silver',
                                 'joty', 'feng', 'feng2', 'ji',
                                 'hayashi_hilda', 'hayashi_mst',
                                 'ours'],
                        help="Author of the reference")
    parser.add_argument('--nary_enc', default='chain',
                        choices=['tree', 'chain'],
                        help="Encoding of n-ary nodes")
    # TODO add argparse param for split
    args = parser.parse_args()
    author_true = args.author_true
    authors_pred = args.authors_pred
    nary_enc = args.nary_enc
    # reference
    dir_true = os.path.join('TMP_disdep', nary_enc, author_true, 'test')
    files_true = {os.path.basename(f).rsplit('.')[0]: f
                  for f in glob(os.path.join(dir_true, '*.dis_dep'))}
    # table header
    len_author_str = max(len(x) for x in authors_pred)
    print('\t'.join([
        '{parser_name: <{width}}'.format(
            parser_name='parser', width=len_author_str),
        'a', 'l', 'n', 'r',
        'al', 'an', 'ar',
        'aln', 'alr',
        'alnr',
        'support'
    ]))

    for author_pred in authors_pred:
        dir_pred = os.path.join('TMP_disdep', nary_enc, author_pred, 'test')
        files_pred = {os.path.basename(f).rsplit('.')[0]: f
                      for f in glob(os.path.join(dir_pred, '*.dis_dep'))}
        assert sorted(files_true.keys()) == sorted(files_pred.keys())

        cnt_tot = 0  # total deps
        cnt_a = 0  # correct heads (attachments)
        cnt_l = 0  # correct labels
        cnt_n = 0  # correct nuclearity
        cnt_r = 0  # correct ranks
        cnt_al = 0  # correct labelled attachments
        cnt_an = 0  # correct attachment + nuc
        cnt_ar = 0  # correct attachment + rank
        cnt_aln = 0  # correct attachment + label + nuc
        cnt_alr = 0  # correct attachment + label + rank
        cnt_alnr = 0  # correct attachment + label + nuc + rank

        for doc_name, f_true in files_true.items():
            f_pred = files_pred[doc_name]
            with codecs.open(f_true, 'r', encoding='utf-8') as f_true:
                with codecs.open(f_pred, 'r', encoding='utf-8') as f_pred:
                    reader_true = csv.reader(f_true, dialect=csv.excel_tab)
                    reader_pred = csv.reader(f_pred, dialect=csv.excel_tab)
                    for line_true, line_pred in zip(reader_true, reader_pred):
                        # i, txt, head, label, clabel, nuc, rank
                        assert line_true[0] == line_pred[0]  # safety check
                        ok_a = line_true[2] == line_pred[2]
                        ok_l = line_true[4] == line_pred[4]  # use clabel
                        ok_n = line_true[5] == line_pred[5]
                        ok_r = line_true[6] == line_pred[6]
                        # update running counters
                        cnt_tot += 1
                        if ok_a:
                            cnt_a += 1
                        if ok_l:
                            cnt_l += 1
                        if ok_n:
                            cnt_n += 1
                        if ok_r:
                            cnt_r += 1
                        if ok_a and ok_l:
                            cnt_al += 1
                        if ok_a and ok_n:
                            cnt_an += 1
                        if ok_a and ok_r:
                            cnt_ar += 1
                        if ok_a and ok_l and ok_n:
                            cnt_aln += 1
                        if ok_a and ok_l and ok_r:
                            cnt_alr += 1
                        if ok_a and ok_l and ok_n and ok_r:
                            cnt_alnr += 1
        print('\t'.join(
            ['{parser_name: <{width}}'.format(
                parser_name=author_pred, width=len_author_str)]
            + ['{:.4f}'.format(float(cnt_x) / cnt_tot)
               for cnt_x in [cnt_a, cnt_l, cnt_n, cnt_r,
                             cnt_al, cnt_an, cnt_ar,
                             cnt_aln, cnt_alr,
                             cnt_alnr]]
            + [str(cnt_tot)]
        ))
