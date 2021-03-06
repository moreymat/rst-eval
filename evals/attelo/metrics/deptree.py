"""Common metrics on dependency trees.

As of 2017-05-18, all implementations assume that _true and _pred both
rely on the same segmentation.
"""

from __future__ import absolute_import, print_function

from collections import Counter
import itertools

import numpy as np


def compute_uas_las(dtree_true, dtree_pred, metrics=None, doc_names=None):
    """Compute dependency metrics for trees in dtree_pred wrt dtree_true.

    The computed metrics are the traditional UAS and LAS, plus LS
    for Labelling Score (counts of correct labels, regardless of head).

    Parameters
    ----------
    dtree_true : list of RstDepTree
        Reference trees

    dtree_pred : list of RstDepTree
        Predicted trees

    metrics : list of str
        If None, defaults to ['U', 'R'] aka. UAS and LAS. Possible
        values in {'U', 'R', 'R+N', 'R+O', 'O+N', 'F', 'tag_R'}.

    Returns
    -------
    res : tuple of float
        Score for each metric in order.
    """
    # 'U': correct unlabelled deps ; was: nb_ua_ok
    # 'R': correct labelled deps ; was: nb_la_ok
    # 'tag_R': correct labellings: right labels, possibly wrong heads ; was: nb_l_ok
    # 'R+N': relation and nuclearity
    # 'R+O': relation and order
    # 'F': relation, order, nuclearity
    nb_tp = Counter({k: 0 for k in metrics})
    nb_tot = 0  # total deps

    for i, (dt_true, dt_pred) in enumerate(
            zip(dtree_true, dtree_pred)):
        if doc_names is not None:
            doc_name = doc_names[i]  # for verbose/debug
        tp = dict()
        tp_bins = dict()
        # exclude fake root from metrics
        # head : dependencies
        if any(x in set(['U', 'N', 'R', 'O', 'R+N', 'R+O', 'O+N', 'F'])
               for x in metrics):
            heads_true = np.array(dt_true.heads[1:])
            heads_pred = np.array(dt_pred.heads[1:])
            tp['U'] = heads_true == heads_pred
            tp_bins['U'] = heads_true[tp['U']]
        # relation tag
        if any(x in set(['tag_R', 'R', 'R+N', 'R+O', 'F']) for x in metrics):
            labels_true = np.array(dt_true.labels[1:])
            labels_pred = np.array(dt_pred.labels[1:])
            tp['tag_R'] = labels_true == labels_pred
            tp_bins['tag_R'] = labels_true[tp['tag_R']]
            # dep
            tp['R'] = np.logical_and(tp['U'], tp['tag_R'])
            tp_bins['R'] = labels_true[tp['R']]
        # nuclearity tag
        if any(x in set(['tag_N', 'N', 'R+N', 'O+N', 'F']) for x in metrics):
            nucs_true = np.array(dt_true.nucs[1:])
            nucs_pred = np.array(dt_pred.nucs[1:])
            tp['tag_N'] = nucs_true == nucs_pred
            tp_bins['tag_N'] = nucs_true[tp['tag_N']]
            # dep
            tp['N'] = np.logical_and(tp['U'], tp['tag_N'])
            tp_bins['N'] = nucs_true[tp['N']]
        # order tag
        if any(x in set(['tag_O', 'O', 'R+O', 'O+N', 'F']) for x in metrics):
            rnks_true = np.array(dt_true.ranks[1:])
            rnks_pred = np.array(dt_pred.ranks[1:])
            tp['tag_O'] = rnks_true == rnks_pred
            tp_bins['tag_O'] = rnks_true[tp['tag_O']]
            # dep
            tp['O'] = np.logical_and(tp['U'], tp['tag_O'])
            tp_bins['O'] = rnks_true[tp['O']]

        # dep on complex labels, build on simpler labelled deps
        if 'R+O' in metrics:
            tp['R+O'] = np.logical_and(tp['R'], tp['O'])
            tp_bins['R+O'] = np.array([
                (x, y) for x, y in zip(
                    labels_true[tp['R+O']], rnks_true[tp['R+O']])
            ])
        if 'R+N' in metrics:
            tp['R+N'] = np.logical_and(tp['R'], tp['N'])
            tp_bins['R+N'] = np.array([
                (x, y) for x, y in zip(
                    labels_true[tp['R+N']], nucs_true[tp['R+N']])
            ])
        if 'O+N' in metrics:
            tp['O+N'] = np.logical_and(tp['N'], tp['O'])
            tp_bins['O+N'] = np.array([
                (x, y) for x, y in zip(
                    labels_true[tp['O+N']], rnks_true[tp['O+N']])
            ])
        # full
        if 'F' in metrics:
            tp['F'] = np.logical_and.reduce((tp['R'], tp['N'], tp['O']))
            tp_bins['F'] = np.array([
                (x, y, z) for x, y, z in zip(
                    labels_true[tp['F']], nucs_true[tp['F']],
                    rnks_true[tp['F']])
            ])

        # for each metric, update the number of true positives with
        # the count for the current instance
        for k, v in nb_tp.items():
            nb_tp[k] = v + len(tp_bins[k])

        nb_tot += len(heads_true)

    scores = {k: float(nb_tp[k]) / nb_tot for k in nb_tp}
    res = tuple([scores[k] for k in metrics])
    return res


def compute_uas_las_listcomp(dtree_true, dtree_pred):
    """Compute dependency metrics for trees in dtree_pred wrt dtree_true.

    The computed metrics are the traditional UAS and LAS, plus LS
    for Labelling Score (counts of correct labels, regardless of head).

    Alternative implementation that uses list comprehensions.

    Parameters
    ----------
    dtree_true: list of RstDepTree
        Reference trees

    dtree_pred: list of RstDepTree
        Predicted trees

    Returns
    -------
    (uas, las, ls): (float, float, float)
        The Unlabelled and Labelled Attachment Scores, plus the
        Labelling Score (new).
    """
    nb_ua_ok = 0  # correct unlabelled deps
    nb_la_ok = 0  # correct labelled deps
    nb_l_ok = 0  # correct labellings (right labels, possibly wrong heads)
    nb_tot = 0  # total deps
    for dt_true, dt_pred in zip(dtree_true, dtree_pred):
        # heads and labels are stored as two lists
        # exclude fake root from metrics
        heads_true = dt_true.heads[1:]
        labels_true = dt_true.labels[1:]

        heads_pred = dt_pred.heads[1:]
        labels_pred = dt_pred.labels[1:]

        # list comprehensions to do pseudo-vectorized operations
        gov_ok = [heads_pred[i] == heads_true[i]
                  for i in range(len(heads_pred))]
        lbl_ok = [labels_pred[i] == labels_true[i]
                  for i in range(len(labels_pred))]
        gov_lbl_ok = [gov_ok[i] and lbl_ok[i]
                      for i in range(len(gov_ok))]
        # update counts
        nb_ua_ok += sum(gov_ok)
        nb_la_ok += sum(gov_lbl_ok)
        nb_l_ok += sum(lbl_ok)
        nb_tot += len(heads_pred)

    score_uas = float(nb_ua_ok) / nb_tot
    score_las = float(nb_la_ok) / nb_tot
    score_ls = float(nb_l_ok) / nb_tot  # NEW

    return (score_uas, score_las, score_ls)


def compute_uas_las_np(dtree_true, dtree_pred):
    """Compute dependency metrics for trees in dtree_pred wrt dtree_true.

    The computed metrics are the traditional UAS and LAS, plus LS
    for Labelling Score (counts of correct labels, regardless of head).

    Alternative implementation that uses numpy.

    Parameters
    ----------
    dtree_true: list of RstDepTree
        Reference trees

    dtree_pred: list of RstDepTree
        Predicted trees

    Returns
    -------
    (uas, las, ls): (float, float, float)
        The Unlabelled and Labelled Attachment Scores, plus the
        Labelling Score (new).
    """
    uas_num = 0  # correct unlabelled deps
    las_num = 0  # correct labelled deps
    ls_num = 0  # correct labellings (right labels, possibly wrong heads)
    nb_tot = 0  # total deps
    for dt_true, dt_pred in zip(dtree_true, dtree_pred):
        # heads and labels are stored as two lists
        # exclude fake root from metrics
        heads_true = dt_true.heads[1:]
        labels_true = dt_true.labels[1:]

        heads_pred = dt_pred.heads[1:]
        labels_pred = dt_pred.labels[1:]

        # use numpy's truly vectorized operations:
        gov_ok = np.equal(heads_true, heads_pred)
        # element-wise comparison of arrays of strings is properly defined
        # only with the infix operator "=="
        # TODO change type of labels_* to arrays of ints, and use
        # np.equal(labels_true, labels_pred)
        lbl_ok = np.array(labels_true) == np.array(labels_pred)
        #
        uas_num += np.count_nonzero(gov_ok)
        las_num += np.count_nonzero(np.logical_and(gov_ok, lbl_ok))
        ls_num += np.count_nonzero(lbl_ok)
        nb_tot += gov_ok.size

    score_uas = float(uas_num) / nb_tot
    score_las = float(las_num) / nb_tot
    score_ls = float(ls_num) / nb_tot  # NEW

    return (score_uas, score_las, score_ls)


# 2016-09-30 undirected variants
def compute_uas_las_undirected(dtree_true, dtree_pred):
    """Compute dependency metrics for trees in dtree_pred wrt dtree_true.

    The computed metrics are the traditional UAS and LAS, plus LS
    for Labelling Score (counts of correct labels, regardless of head).

    Parameters
    ----------
    dtree_true: list of RstDepTree
        Reference trees

    dtree_pred: list of RstDepTree
        Predicted trees

    Returns
    -------
    (uas, las, ls): (float, float, float)
        The Unlabelled and Labelled Attachment Scores, plus the
        Labelling Score (new).
    """
    nb_ua_ok = 0  # correct unlabelled deps
    nb_la_ok = 0  # correct labelled deps
    nb_tot = 0  # total deps

    for dt_true, dt_pred in zip(dtree_true, dtree_pred):
        # undirected dependencies are equivalent to the span they cover
        # each span is a tuple with a tuple inside ((fst, snd), lbl)
        spans_true = set((tuple(sorted((gov, dep))), lbl)
                         for dep, (gov, lbl)
                         in enumerate(zip(dt_true.heads[1:], dt_true.labels[1:]),
                                      start=1))
        spans_pred = set((tuple(sorted((gov, dep))), lbl)
                         for dep, (gov, lbl)
                         in enumerate(zip(dt_pred.heads[1:], dt_pred.labels[1:]),
                                      start=1))
        nb_tot += len(spans_pred)
        nb_ua_ok += len(set(x[0] for x in spans_true).intersection(
            set(x[0] for x in spans_pred)))
        nb_la_ok += len(spans_true.intersection(spans_pred))

    score_uas = float(nb_ua_ok) / nb_tot
    score_las = float(nb_la_ok) / nb_tot

    return (score_uas, score_las)


def dep_compact_report(parser_true, d_preds, dep_metrics, doc_names,
                       labelset_true, digits=3, percent=False,
                       out_format='text'):
    """Compact textual report of parser accuracies with dependency metrics.

    Parameters
    ----------
    parser_true : str
        Name of the parser used as reference.
    d_preds : list of (str, dict from str to RstDepTree)
        List of predicted head-ordered d-trees for each parser.
    dep_metrics : list of str
        List of dependency metrics to include in the report.
    doc_names : list of str?
        TODO
    labelset_true : ?
        TODO
    digits : int, defaults to 3
        Significant digits for rounding.
    percent : boolean, defaults to False
        Display scores as percentages.
    out_format : one of {'text', 'latex'}
        Output format.

    Returns
    -------
    report : str
        Textual report
    """
    out_format_options = ('text', 'latex')
    if out_format not in out_format_options:
        raise ValueError('out_format has to be one of ' +
                         str(out_format_options))

    # report
    # * table format
    headers = dep_metrics
    headers = ["parser"] + headers
    if out_format == 'latex':
        # bold font for column headers
        headers = ['\\textbf{{{}}}'.format(x) for x in headers]
    # width of first column (parser name)
    width = max([len(parser_name) for parser_name, _ in d_preds] +
                [len(headers[0])])
    fmt = '%% %ds' % width  # first col: parser name
    if out_format == 'latex':
        fmt += ' &'
        fmt += ' &'.join(['% {}s'.format(len(x)) for x in headers[1:]])
        fmt += ' \\\\'  # print "\\"
    else:  # if out_format == 'text':
        fmt += '  '
        fmt += ' '.join(['% 9s' for _ in headers[1:]])
    fmt += '\n'

    report = ""
    if out_format == 'latex':
        report += '\n'.join([
            '\\begin{table}[h]',
            '\\caption{\\label{dtree-eval} Dependency evaluation. U = unlabelled dependencies, O = dependencies labelled with the order of attachment, N = dependencies labelled with the nuclearity alone, O+N = order and nuclearity, R = relation, R+N = relation and nuclearity, F = fully labelled dependencies.}',
            '\\begin{center}',
            '\\begin{tabular}{' + 'l' * len(headers) +'}',
            '\\toprule',
            ''
        ])
    report += fmt % tuple(headers)
    if out_format == 'latex':
        report += '\\midrule\n'
    else:
        report += '\n'

    # display percentages
    dep_digits = digits - 2 if percent else digits
    # end table format and header line

    # * table content
    # dtree_true_list = [dtree_true[doc_name] for doc_name in doc_names]
    # FIXME
    dtree_true_list = []
    for parser_name, dtree_pred in d_preds:
        if parser_name == parser_true:
            dtree_true_list = [dtree_pred[doc_name] for doc_name in doc_names]
            break
    # end FIXME
    # _pred
    for parser_name, dtree_pred in d_preds:
        try:
            dtree_pred_list = [dtree_pred[doc_name] for doc_name in doc_names]
        except KeyError:
            print(parser_name)
            raise
        # check that labelset_pred is a subset of labelset_true
        labelset_pred = set(itertools.chain.from_iterable(
            x.labels for x in dtree_pred_list))
        try:
            assert labelset_pred.issubset(labelset_true)
        except AssertionError:
            print(parser_name)
            print('T', sorted(x for x in labelset_true if x is not None))
            print('P', sorted(x for x in labelset_pred if x is not None))
            print('T & P', sorted(labelset_true.intersection(labelset_pred)))
            print('T - P', sorted(labelset_true - labelset_pred))
            print('P - T', sorted(labelset_pred - labelset_true))
            raise
        # end check
        all_scores = []
        all_scores += list(compute_uas_las(
            dtree_true_list, dtree_pred_list, metrics=dep_metrics,
            doc_names=doc_names))
        # append to report
        values = ['{pname: <{fill}}'.format(pname=parser_name, fill=width)]
        for v in all_scores:
            if percent:
                v = v * 100.0
            values += ["{0:0.{1}f}".format(v, dep_digits)]
        report += fmt % tuple(values)
    # LaTeX footer if relevant
    if out_format == 'latex':
        report += '\n'.join([
            '\\bottomrule',
            '\\end{tabular}',
            '\\end{center}',
            '\\end{table}'
        ])
    # end table content

    # replace underscores in parser names etc
    report = report.replace('_', ' ')
    return report


def dep_similarity(d_preds, doc_names, labelset_true, dep_metric=None,
                   digits=3, percent=False, out_format='text'):
    """Compact textual report of parser accuracies with dependency metrics.

    Parameters
    ----------
    d_preds : list of (str, dict from str to RstDepTree)
        List of predicted head-ordered d-trees for each parser.
    doc_names : list of str
        List of document names.
    labelset_true : list of str
        List of true labels.
    dep_metric : str, optional
        Dependency metric to use ; defaults to 'U' (aka UAS).
    digits : int, defaults to 3
        Significant digits for rounding.
    percent : boolean, defaults to False
        Display scores as percentages.
    out_format : one of {'text', 'latex'}
        Output format.

    Returns
    -------
    report : str
        Textual report
    """
    out_format_options = ('text', 'latex')
    if out_format not in out_format_options:
        raise ValueError('out_format has to be one of ' +
                         str(out_format_options))

    if dep_metric is None:
        dep_metric = 'U'

    # prepare scaffold for report
    width = max(len(parser_name) for parser_name, _ in d_preds)
    headers = [k[:7] for k, v in d_preds]
    # if we wanted to print the support, would be here for col name
    fmt = '%% %ds' % width  # first col: parser name
    if out_format == 'latex':
        fmt += ' &'
        fmt += '&'.join(['% 9s' for _ in headers])
        fmt += '\\\\'  # print "\\"
    else:  # if out_format == 'text':
        fmt += '  '
        fmt += ' '.join(['% 9s' for _ in headers])
    fmt += '\n'
    headers = [""] + headers

    report = ""
    if out_format == 'latex':
        report += '\n'.join([
            '\\begin{table}[h]',
            '\\caption{\\label{dtree-sim} Pairwise similarity between parsers predictions, dependency metric U.}',
            '\\begin{center}',
            '\\begin{tabular}{' + 'l' * len(headers) +'}',
            '\\toprule',
            ''
        ])
    report += fmt % tuple(headers)
    report += '\n'
    if out_format == 'latex':
        report += '\\midrule\n'

    # display percentages
    if percent:
        digits = digits - 2

    # use each parser as reference, in turn
    for parser_true, dtree_true in d_preds:
        values = [parser_true]  # name of row
        # get list of dtrees
        dtree_true_list = [dtree_true[doc_name] for doc_name in doc_names]
        for parser_name, dtree_pred in d_preds:
            dtree_pred_list = [dtree_pred[doc_name] for doc_name in doc_names]
            # compute score
            f1 = compute_uas_las(
                dtree_true_list, dtree_pred_list, metrics=[dep_metric],
                doc_names=doc_names)[0]
            # fill report
            values += ["{0:0.{1}f}".format(f1 * 100.0 if percent else f1,
                                           digits)]
        report += fmt % tuple(values)

    if out_format == 'latex':
        report += '\n'.join([
            '\\bottomrule',
            '\\end{tabular}',
            '\\end{center}',
            '\\end{table}'
        ])
    report = report.replace('_', ' ')

    return report
