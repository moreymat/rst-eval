#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Authors: Eric Kow
# License: CeCILL-B (BSD3-like)

"""
Convert RST trees to dependency trees and back.
"""

import itertools

import numpy as np

from .annotation import EDU, _binarize, NUC_N, NUC_S  # , NUC_R
from ..internalutil import treenode


class RstDtException(Exception):
    """
    Exceptions related to conversion between RST and DT trees.
    The general expectation is that we only raise these on bad
    input, but in practice, you may see them more in cases of
    implementation error somewhere in the conversion process.
    """
    def __init__(self, msg):
        super(RstDtException, self).__init__(msg)


_ROOT_HEAD = 0
_ROOT_LABEL = 'ROOT'

DEFAULT_HEAD = _ROOT_HEAD
DEFAULT_LABEL = _ROOT_LABEL
DEFAULT_NUC = NUC_N
DEFAULT_RANK = 0


# helper function for conversion from binary to nary relations
def binary_to_nary(nary_enc, pairs):
    """Retrieve nary relations from a set of binary relations.

    Parameters
    ----------
    nary_enc: one of {"chain", "tree"}
        Encoding from n-ary to binary relations.
    pairs: iterable of pairs of identifier (ex: integer, string...)
        Binary relations.

    Return
    ------
    nary_rels: list of tuples of identifiers
        Nary relations.
    """
    nary_rels = []
    open_ends = []  # companion to nary_rels: open end
    for gov_idx, dep_idx in pairs:
        try:
            # search for an existing fragmented EDU this same-unit
            # could belong to
            open_frag = open_ends.index(gov_idx)
        except ValueError:
            # start a new fragmented EDU
            nary_rels.append([gov_idx, dep_idx])
            if nary_enc == 'chain':
                open_ends.append(dep_idx)
            else:  # 'tree'
                open_ends.append(gov_idx)
        else:
            # append dep_idx to an existing fragmented EDU
            nary_rels[open_frag].append(dep_idx)
            # NB: if "tree", no need to update the open end
            if nary_enc == 'chain':
                open_ends[open_frag] = dep_idx
    nary_rels = [tuple(x) for x in nary_rels]
    return nary_rels


class RstDepTree(object):
    """RST dependency tree

    Attributes
    ----------
    edus : list of EDU
        List of the EDUs of this document.

    origin : Document?, optional
        TODO

    nary_enc : one of {'chain', 'tree'}, optional
        Type of encoding used for n-ary relations: 'chain' or 'tree'.
        This determines for example how fragmented EDUs are resolved.
    """

    def __init__(self, edus=[], origin=None, nary_enc='chain'):
        # WIP 2016-07-20 nary_enc to resolve fragmented EDUs
        if nary_enc not in ['chain', 'tree']:
            raise ValueError("nary_enc must be in {'tree', 'chain'}")
        self.nary_enc = nary_enc
        # FIXME find a clean way to avoid generating a new left padding EDU
        # here
        _lpad = EDU.left_padding()
        self.edus = [_lpad] + edus
        # mapping from EDU num to idx
        self.idx = {e.num: i for i, e in enumerate(self.edus)}
        # init tree structure
        nb_edus = len(self.edus)
        self.heads = [DEFAULT_HEAD for _ in range(nb_edus)]
        self.labels = [DEFAULT_LABEL for _ in range(nb_edus)]
        # NEW nuclearity and ranking of attachment
        # first trial: ranks default to 0, nuclearity to S
        self.nucs = [DEFAULT_NUC for _ in range(nb_edus)]
        self.ranks = [DEFAULT_RANK for _ in range(nb_edus)]
        # end NEW

        # set special values for fake root
        self.heads[0] = -1
        self.labels[0] = None
        self.nucs[0] = None
        self.ranks[0] = -1

        # set fake root's origin and context to be the same as the first
        # real EDU's
        context = edus[0].context if edus else None
        origin = edus[0].origin if (origin is None and edus) else origin
        # update origin and context for fake root
        self.edus[0].set_context(context)
        self.edus[0].set_origin(origin)
        # update the dep tree's origin
        self.set_origin(origin)

    def append_edu(self, edu):
        """Append an EDU to the list of EDUs"""
        self.edus.append(edu)
        self.idx[edu.num] = len(self.edus) - 1
        # set default values for the tree structure
        self.heads.append(DEFAULT_HEAD)
        self.labels.append(DEFAULT_LABEL)
        self.nucs.append(DEFAULT_NUC)
        self.ranks.append(DEFAULT_RANK)

    def add_dependency(self, gov_num, dep_num, label=None, nuc=NUC_S,
                       rank=None):
        """Add a dependency between two EDUs.

        Parameters
        ----------
        gov_num: int
            Number of the head EDU
        dep_num: int
            Number of the modifier EDU
        label: string, optional
            Label of the dependency
        nuc: string, one of [NUC_S, NUC_N]
            Nuclearity of the modifier
        rank: integer, optional
            Rank of the modifier in the order of attachment to the head.
            `None` means it is not given declaratively and it is instead
            inferred from the rank of modifiers previously attached to
            the head.
        """
        _idx_gov = self.idx[gov_num]
        _idx_dep = self.idx[dep_num]
        self.heads[_idx_dep] = _idx_gov
        self.labels[_idx_dep] = label
        self.nucs[_idx_dep] = nuc
        if rank is None:  # assign first free rank
            # was: rank = len(self.deps[_idx_gov])
            sisters = [i for i, hd in enumerate(self.heads)
                       if hd == _idx_gov]
            rank = max(self.ranks[i] for i in sisters) + 1
        self.ranks[_idx_dep] = rank

    def add_dependencies(self, gov_num, dep_nums, labels=None, nucs=None,
                         rank=None):
        """Add a set of dependencies with a unique governor and rank.

        Parameters
        ----------
        gov_num : int
            Number of the head EDU

        dep_nums : list of int
            Number of the modifier EDUs

        labels : list of string, optional
            Labels of the dependencies

        nuc : list of string, one of [NUC_S, NUC_N]
            Nuclearity of the modifiers

        rank : integer, optional
            Rank of the modifiers in the order of attachment to the head.
            `None` means it is not given declaratively and it is instead
            inferred from the rank of modifiers previously attached to
            the head.
        """
        # locate common governor, get common rank
        _idx_gov = self.idx[gov_num]
        if rank is None:  # assign first free rank
            sisters = [i for i, hd in enumerate(self.heads)
                       if hd == _idx_gov]
            if not sisters:
                # ranks are 1-based, so first set of dependents has rank 1
                rank = 1
            else:
                rank = max(self.ranks[i] for i in sisters) + 1

        # default values for labels and nucs, if necessary
        if labels is None:
            labels = [None for _ in dep_nums]
        if nucs is None:
            nucs = [NUC_S for _ in dep_nums]

        # finally, add dependencies
        for dep_num, label, nuc in zip(dep_nums, labels, nucs):
            _idx_dep = self.idx[dep_num]
            self.heads[_idx_dep] = _idx_gov
            self.labels[_idx_dep] = label
            self.nucs[_idx_dep] = nuc
            # common rank
            self.ranks[_idx_dep] = rank

    def get_dependencies(self, lbl_type='rel'):
        """Get the list of dependencies in this dependency tree.

        Each dependency is a 3-uple (gov, dep, label),
        gov and dep being EDUs.

        Parameters
        ----------
        lbl_type: one of {'rel', 'rel+nuc'} (TODO 'rel+nuc+rnk'?)
            Type of the labels.
        """
        if lbl_type not in ['rel', 'rel+nuc']:
            raise ValueError("lbl_type needs to be one of {'rel', 'rel+nuc'}")

        edus = self.edus

        deps = self.edus[1:]
        gov_idxs = self.heads[1:]
        if lbl_type == 'rel':
            labels = self.labels[1:]
        elif lbl_type == 'rel+nuc':
            labels = list(zip(self.labels[1:],
                              ['N' + nuc[0] for nuc in self.nucs[1:]]))
        else:
            raise NotImplementedError("WIP")

        result = [(edus[gov_idx], dep, lbl)
                  for gov_idx, dep, lbl
                  in itertools.izip(gov_idxs, deps, labels)]

        return result

    def set_root(self, root_num):
        """Designate an EDU as a real root of the RST tree structure"""
        _idx_fake_root = _ROOT_HEAD
        _idx_root = self.idx[root_num]
        _lbl_root = _ROOT_LABEL
        self.heads[_idx_root] = _idx_fake_root
        self.labels[_idx_root] = _lbl_root
        self.nucs[_idx_root] = DEFAULT_NUC
        # calculate rank (for a unique root, should always be 0)
        sisters = [i for i, hd in enumerate(self.heads)
                   if hd == _idx_fake_root]
        rank = max(self.ranks[i] for i in sisters) + 1
        self.ranks[_idx_root] = rank

    def deps(self, gov_idx):
        """Get the ordered list of dependents of an EDU"""
        # TODO a numpy version should be cleaner
        ranked_deps = sorted((rk, i) for i, rk in enumerate(self.ranks)
                             if self.heads[i] == gov_idx)
        sorted_deps = [i for rk, i in ranked_deps]
        return sorted_deps

    def fragmented_edus(self):
        """Get the fragmented EDUs in this RST tree.

        Fragmented EDUs are made of two or more EDUs linked by
        "same-unit" relations.

        Returns
        -------
        frag_edus: list of tuple of int
            Each fragmented EDU is given as a tuple of the indices of
            the fragments.
        """
        nary_enc = self.nary_enc
        su_deps = [(gov_idx, dep_idx) for dep_idx, (gov_idx, lbl)
                   in enumerate(zip(self.heads[1:], self.labels[1:]),
                                start=1)
                   if lbl.lower() == 'same-unit']
        frag_edus = binary_to_nary(nary_enc, su_deps)
        return frag_edus

    def real_roots_idx(self):
        """Get the list of the indices of the real roots"""
        return self.deps(_ROOT_HEAD)

    def set_origin(self, origin):
        """Update the origin of this annotation.

        Parameters
        ----------
        origin : FileId
            File identifier of the origin of this annotation.
        """
        self.origin = origin

    def spans(self):
        """For each EDU, get the tree span it dominates (on EDUs).

        Dominance here is recursively defined.

        Returns
        -------
        span_beg: array of int
            Index of the leftmost EDU dominated by an EDU.
        span_end: array of int
            Index of the rightmost EDU dominated by an EDU.
        """
        span_beg = np.array([i for i, e in enumerate(self.edus)])
        span_end = np.array([i for i, e in enumerate(self.edus)])
        while True:
            span_new_beg = np.copy(span_beg)
            span_new_end = np.copy(span_end)
            for i, hd in enumerate(self.heads[1:], start=1):
                span_new_beg[hd] = min(span_new_beg[i], span_new_beg[hd])
                span_new_end[hd] = max(span_new_end[i], span_new_end[hd])
            if ((np.array_equal(span_new_beg, span_beg) and
                 np.array_equal(span_new_end, span_end))):
                # fixpoint reached
                break
            # otherwise, we'll go for another round
            span_beg = span_new_beg
            span_end = span_new_end
        return span_beg, span_end

    @classmethod
    def from_simple_rst_tree(cls, rtree):
        """Converts a ̀SimpleRSTTree` to an `RstDepTree`"""
        edus = sorted(rtree.leaves(), key=lambda x: x.span.char_start)
        # building a SimpleRSTTree requires to binarize the original
        # RSTTree first, so 'chain' is the only possibility
        dtree = cls(edus, nary_enc='chain')

        def walk(tree):
            """
            Recursively walk down tree, collecting dependency information
            between EDUs as we go.
            Return/percolate the num of the head found in our descent.
            """
            if len(tree) == 1:  # pre-terminal
                # edu = tree[0]
                edu_num = treenode(tree).edu_span[0]
                return edu_num
            else:
                rel = treenode(tree).rel
                left = tree[0]
                right = tree[1]
                nscode = treenode(tree).nuclearity
                lhead = walk(left)
                rhead = walk(right)

                if nscode == "NS":
                    head = lhead
                    child = rhead
                    nuc_child = NUC_S
                elif nscode == "SN":
                    head = rhead
                    child = lhead
                    nuc_child = NUC_S
                elif nscode == "NN":
                    head = lhead
                    child = rhead
                    nuc_child = NUC_N
                else:
                    raise RstDtException("Don't know how to handle %s trees" %
                                         nscode)
                dtree.add_dependency(head, child, label=rel, nuc=nuc_child,
                                     rank=None)
                return head

        root = walk(rtree)  # populate dtree structure and get its root
        dtree.set_root(root)

        return dtree

    @classmethod
    def from_rst_tree(cls, rtree, nary_enc='tree'):
        """Converts an ̀RSTTree` to an `RstDepTree`.

        Parameters
        ----------
        nary_enc : one of {'chain', 'tree'}
            If 'chain', the given RSTTree is binarized first.
        """
        edus = sorted(rtree.leaves(), key=lambda x: x.span.char_start)
        # if 'chain', binarize the tree first
        if nary_enc == 'chain':
            rtree = _binarize(rtree)
        dtree = cls(edus, nary_enc=nary_enc)

        def walk(tree):
            """
            Recursively walk down tree, collecting dependency information
            between EDUs as we go.
            Return/percolate the num of the head found in our descent.
            """
            if len(tree) == 1:  # pre-terminal
                # edu = tree[0]
                edu_num = treenode(tree).edu_span[0]
                return edu_num
            else:
                # first, recurse and get the head of each subtree
                kid_heads = [walk(kid) for kid in tree]
                kid_rels = [treenode(kid).rel for kid in tree]
                kid_nucs = [treenode(kid).nuclearity for kid in tree]
                # use heads and nucs to pick head, defined as the leftmost
                # nucleus
                head_idx = kid_nucs.index('Nucleus')
                head = kid_heads[head_idx]
                deps = [kid_head for i, kid_head in enumerate(kid_heads)
                        if i != head_idx]
                nucs = [kid_nuc for i, kid_nuc in enumerate(kid_nucs)
                        if i != head_idx]
                rels = [kid_rel for i, kid_rel in enumerate(kid_rels)
                        if i != head_idx]

                dtree.add_dependencies(head, deps, labels=rels, nucs=nucs,
                                       rank=None)
                return head

        root = walk(rtree)  # populate dtree structure and get its root
        dtree.set_root(root)

        return dtree
