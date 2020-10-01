"""This module implements a dumper for the svmlight format

See `sklearn.datasets.svmlight_format`
"""

from __future__ import absolute_import

import itertools


def _dump_svmlight(X_gen, y_gen, f, comment):
    """Actually do dump"""
    if comment:
        f.write('# {}\n'.format(comment))

    # define string formatting patterns for values and lines
    value_pattern = '{fid}:{fv}'

    line_pattern = '{yi}'
    line_pattern += ' {s}\n'

    for X, y in itertools.izip(X_gen, y_gen):
        for x, yi in itertools.izip(X, y):
            # sort features by their index
            x = sorted(x)
            # zero values need not be written in the svmlight format
            x = [(feat_id, feat_val) for feat_id, feat_val in x
                 if feat_val != 0]
            # feature ids in libsvm are one-based, so feat_id + 1
            # TODO use unicode all along, then encode to ascii at the last
            # possible moment (aka here), e.g.
            # s = u' '.join(...) ; f.write(... .encode('ascii'))
            s = ' '.join(value_pattern.format(fid=str(feat_id + 1),
                                              fv=feat_val)
                         for feat_id, feat_val in x)
            f.write(line_pattern.format(yi=yi, s=s))


def dump_svmlight_file(X_gen, y_gen, f, zero_based=True, comment=None,
                       query_id=None):
    """Dump the dataset in svmlight file format.
    """
    with open(f, 'wb') as f:
        _dump_svmlight(X_gen, y_gen, f, comment)
