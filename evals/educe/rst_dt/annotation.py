#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Authors: Philippe Muller, Eric Kow
# License: CeCILL-B (BSD-3 like)

# disable "pointless string" warning because we want attribute docstrings
# pylint: disable=W0105

"""
Educe-style representation for RST discourse treebank trees
"""

import base64
import copy
import functools
import os
import subprocess
import tempfile

# nltk.draw for rendering in PS, PDF, PNG ; see RSTTree.to_ps()
from nltk.draw.tree import tree_to_treesegment
from nltk.draw.util import CanvasFrame
from nltk.internals import find_binary

from educe.annotation import Standoff, Span
from educe.external.parser import SearchableTree
from ..internalutil import treenode


# nuclearities
NUC_N = "Nucleus"
NUC_S = "Satellite"
NUC_R = "Root"

# ghostscript parameters to generate images in different formats
_GS_PARAMS = {
    'png': '-sDEVICE=png16m -r90 -dTextAlphaBits=4 -dGraphicsAlphaBits=4',
    'pdf': '-sDEVICE=pdfwrite',
}


class RSTTreeException(Exception):
    """
    Exceptions related to RST trees not looking like we would
    expect them to
    """
    def __init__(self, msg):
        super(RSTTreeException, self).__init__(msg)


# pylint: disable=R0903
# I would just used a namedtuple here except that I also want
# to associate the fields with docstrings
class RSTContext(object):
    """
    Additional annotations or contextual information that could
    accompany a RST tree proper. The idea is to have each subtree
    pointing back to the same context object for easy retrieval.
    """
    def __init__(self, text, sentences, paragraphs):
        self._text = text
        "original text on which standoff annotations are based"

        self.sentences = sentences
        "sentence annotations pointing back to the text"

        self.paragraphs = paragraphs
        "Paragraph annotations pointing back to the text"

    def text(self, span=None):
        """
        Return the text associated with these annotations (or None),
        optionally limited to a span
        """
        if self._text is None:
            return None
        elif span is None:
            return self._text
        else:
            return self._text[span.char_start:span.char_end]
# pylint: enable=R0903


# pylint: disable=R0913
class EDU(Standoff):
    """
    An RST leaf node
    """
    _SUMMARY_LEN = 20

    def __init__(self, num, span, text, context=None, origin=None):
        super(EDU, self).__init__(origin)

        self.num = num
        "EDU number (as used in tree node `edu_span`)"

        self.span = span
        "text span"

        self.raw_text = text
        """
        text that was in the EDU annotation itself

        This is not the same as the text that was in the annotated
        document, on which all standoff annotations and spans
        are based.
        """

        self.context = context
        """
        See the `RSTContext` object
        """

    def set_origin(self, origin):
        """
        Update the origin of this annotation and any contained within

        Parameters
        ----------
        origin : FileId
            File identifier of the origin of this annotation.
        """
        self.origin = origin

    def set_context(self, context):
        """
        Update the context of this annotation.
        """
        self.context = context

    def identifier(self):
        """
        A global identifier (assuming the origin can be used to
        uniquely identify an RST tree)
        """
        # idiosyncratic
        if self.is_left_padding():
            return 'ROOT'
        # end idiosyncratic
        if self.origin:
            return self.origin.mk_global_id(str(self.num))
        else:
            return str(self.num)

    def __repr__(self):
        txt = self.text()
        if len(txt) > self._SUMMARY_LEN + 3:
            txt = txt[:self._SUMMARY_LEN] + "..."
        return "EDU:[%s]" % txt

    # EXPERIMENTAL for convenient display of RST trees in PNG images
    # (replace RSTTool)
    def __str__(self):
        # wrap tokens (roughly) at _SUMMARY_LEN
        raw_toks = [x for x in self.text().split(' ')]
        wrapped_toks = [[]]
        wrapped_toks[-1].append("({})".format(self.num))  # prepend EDU num
        for tok in raw_toks:
            # special case to handle very long tokens
            while len(tok) > self._SUMMARY_LEN:
                wrapped_toks.append([tok[:self._SUMMARY_LEN] + '-'])
                tok = tok[self._SUMMARY_LEN:]
            # regular case: optional newline, then append to current line
            if len(' '.join(wrapped_toks[-1] + [tok])) > self._SUMMARY_LEN:
                wrapped_toks.append([])
            wrapped_toks[-1].append(tok)
        return '\n'.join(' '.join(tok for tok in group_toks)
                         for group_toks in wrapped_toks)

    def text(self):
        """
        Return the text associated with this EDU. We try to return
        the underlying annotated text if we have the necessary
        context; if we not, we just fall back to the raw EDU text
        """
        if self.context:
            return self.context.text(self.span)
        else:
            return self.raw_text

    # left padding EDUs
    _lpad_num = 0
    _lpad_span = Span(0, 0)
    _lpad_txt = ''

    @classmethod
    def left_padding(cls, context=None, origin=None):
        """Return a left padding EDU"""
        return cls(cls._lpad_num, cls._lpad_span, cls._lpad_txt,
                   context, origin)

    def is_left_padding(self):
        """Returns True for left padding EDUs"""
        return (self.num == self._lpad_num and
                self.span == self._lpad_span)
# pylint: enable=R0913


class Node(object):
    """
    A node in an `RSTTree` or `SimpleRSTTree`.
    """

    def __init__(self, nuclearity, edu_span, span, rel, context=None):
        self.nuclearity = nuclearity
        "one of Nucleus, Satellite, Root"

        self.edu_span = edu_span
        "pair of integers denoting edu span by count"

        self.span = span
        "span"

        self.rel = rel
        """
        relation label (see `SimpleRSTTree` for a note on the different
        interpretation of `rel` with this and `RSTTree`)
        """

        self.context = context
        "See the `RSTContext` object"

    def __repr__(self):
        return "%s %s %s" % (self.nuclearity,
                             "%s-%s" % self.edu_span,
                             self.rel)

    # EXPERIMENTAL for convenient display of RST trees in PNG images
    # (replace RSTTool)
    def __str__(self):
        return "%s %s %s" % (
            "%s-%s" % self.edu_span,
            self.nuclearity,
            self.rel)

    def __eq__(self, other):
        return\
            self.nuclearity == other.nuclearity and\
            self.edu_span == other.edu_span and\
            self.span == other.span and\
            self.rel == other.rel

    def __ne__(self, other):
        return not self.__eq__(other)

    def is_nucleus(self):
        """
        A node can either be a nucleus, a satellite, or a root node.
        It may be easier to work with SimpleRSTTree, in which nodes
        can only either be nucleus/satellite or much more rarely,
        root.
        """
        return self.nuclearity == NUC_N

    def is_satellite(self):
        """
        A node can either be a nucleus, a satellite, or a root node.
        """
        return self.nuclearity == NUC_S


# pylint: disable=R0904, E1103
class RSTTree(SearchableTree, Standoff):
    """
    Representation of RST trees which sticks fairly closely to the
    raw RST discourse treebank one.
    """

    def __init__(self, node, children, origin=None, verbose=False):
        """
        See `educe.rst_dt.parse` to build trees from strings
        """
        SearchableTree.__init__(self, node, children)
        Standoff.__init__(self, origin)
        # WIP 2016-11-10 store num of head in node
        if len(children) == 1 and isinstance(children[0], EDU):
            # pre-terminal: head is num of terminal (EDU)
            node.head = children[0].num
        else:
            # internal node
            kids_nuclei = [i for i, kid in enumerate(children)
                           if kid.label().nuclearity == NUC_N]
            if len(kids_nuclei) == 1:
                # 1 nucleus, 1-n satellites: n mono-nuc relations
                pass
            elif len(kids_nuclei) == len(children):
                # all children are nuclei: 1 multi-nuc relation
                kid_rels = [kid.label().rel for kid in children]
                if len(set(kid_rels)) > 1:
                    if verbose:
                        err_msg = ('W: More than one label in multi-nuclear'
                                   ' relation {}'.format(children))
                        print(err_msg)
            else:
                # corner case, should not happen
                err_msg = 'E: Unknown pattern in children'
                print(err_msg)
            # its head is the head of its leftmost nucleus child
            lnuc = children[kids_nuclei[0]]
            node.head = lnuc.label().head
        # end WIP head

    def set_origin(self, origin):
        """Update the origin of this annotation and any contained within

        Parameters
        ----------
        origin : FileId
            File identifier of the origin of this annotation.
        """
        self.origin = origin
        for child in self:
            child.set_origin(origin)

    def text_span(self):
        return treenode(self).span

    def _members(self):
        return list(self)  # children

    def __repr__(self):
        return self.pformat()

    # image representations, copied and adapted from nltk.tree.Tree._repr_png_
    # for:
    # * modularity, with PS, PNG, PDF formats using the same codebase
    # * customized visual appearance (fonts, spacing)
    def _repr_png_(self):
        """Draws and outputs in PNG for ipython.

        PNG is used instead of PDF, since it can be displayed in the qt
        console and has wider browser support.
        """
        with tempfile.NamedTemporaryFile() as f_tmp:
            in_path = '{0:}.ps'.format(f_tmp.name)
            out_path = '{0:}.png'.format(f_tmp.name)
            # generate PostScript using the drawing utils of NLTK
            self.to_ps(in_path)
            # convert to PNG with ghostscript
            subprocess.call(
                [find_binary('gs',
                             binary_names=['gswin32c.exe', 'gswin64c.exe'],
                             env_vars=['PATH'], verbose=False)] +
                '-q -dEPSCrop {2:} -dSAFER -dBATCH -dNOPAUSE -sOutputFile={0:} {1:}'
                .format(out_path, in_path, _GS_PARAMS['png']).split())
            # this function will return the encoded+decoded bytes of the PNG
            # file
            with open(out_path, 'rb') as sr:
                res = sr.read()
            os.remove(in_path)
            os.remove(out_path)
            return base64.b64encode(res).decode()

    def to_ps(self, filename):
        """Export as a PostScript image.

        This function is used by `_repr_png_`.
        """
        _canvas_frame = CanvasFrame()
        # WIP customization of visual appearance
        # NB: conda-provided python and tk cannot access most fonts on the
        # system, thus it currently falls back on the default font
        widget = tree_to_treesegment(_canvas_frame.canvas(), self,
                                     tree_yspace=35,
                                     node_font=('Verdana', -18, 'bold'),
                                     leaf_font=('Verdana', -18))
        _canvas_frame.add_widget(widget)
        x, y, w, h = widget.bbox()
        # print_to_file uses scrollregion to set the width and height of the
        # pdf
        _canvas_frame.canvas()['scrollregion'] = (0, 0, w, h)
        # print to file
        _canvas_frame.print_to_file(filename)
        _canvas_frame.destroy_widget(widget)

    def to_pdf(self, filename):
        """Image representation in PDF.
        """
        # generate PostScript using the drawing utils of NLTK
        root, ext = os.path.splitext(filename)
        in_path = '{0:}.ps'.format(root)
        self.to_ps(in_path)
        # convert to PDF with ghostscript
        subprocess.call(
            [find_binary('gs',
                         binary_names=['gswin32c.exe', 'gswin64c.exe'],
                         env_vars=['PATH'], verbose=False)] +
            '-q -dEPSCrop {2:} -dSAFER -dBATCH -dNOPAUSE -sOutputFile={0:} {1:}'
            .format(filename, in_path, _GS_PARAMS['pdf']).split())
        os.remove(in_path)

    def edu_span(self):
        """
        Return the span of the tree in terms of EDU count
        See `self.span` refers more to the character offsets
        """
        return treenode(self).edu_span

    def get_spans(self, subtree_filter=None, exclude_root=False,
                  span_type='edus'):
        """Get the spans of a constituency tree.

        Each span is described by a triplet (edu_span, nuclearity,
        relation).

        Parameters
        ----------
        subtree_filter : function, defaults to None
            Function to filter all local trees.

        exclude_root : boolean, defaults to False
            If True, exclude the span of the root node. This cannot be
            expressed with `subtree_filter` because the latter is limited
            to properties local to each subtree in isolation. Or maybe I
            just missed something.

        span_type : one of {'edus', 'chars'}
            Whether each span is expressed on EDU or character indices.
            Character indices are useful to compare spans from trees
            whose EDU segmentation differs.

        Returns
        -------
        spans: list of tuple((int, int), str, str)
            List of tuples, each describing a span with a tuple
            ((edu_start, edu_end), nuclearity, relation).
        """
        tnodes = [x.label() for x in self.subtrees(filter=subtree_filter)
                  if isinstance(x, RSTTree)]
        if exclude_root:
            tnodes = tnodes[1:]
        # 2016-11-10 add a 4th element: head
        # 2017-04-12 enable char spans
        if span_type == 'chars':
            spans = [((tn.span.char_start, tn.span.char_end),
                      tn.nuclearity, tn.rel, tn.head)
                     for tn in tnodes]
        else:
            spans = [(tn.edu_span, tn.nuclearity, tn.rel, tn.head)
                     for tn in tnodes]
        return spans

    def text(self):
        """
        Return the text corresponding to this RST subtree.
        If the context is set, we return the appropriate
        segment from the subset of the text.
        If not we just concatenate the raw text of all
        EDU leaves.
        """
        node = treenode(self)
        if node.context:
            return node.context.text(node.span)
        else:
            return " ".join(l.raw_text for l in self.leaves())


class SimpleRSTTree(SearchableTree, Standoff):
    """
    Possibly easier representation of RST trees to work with:

    * binary
    * relation labels on parent nodes instead of children

    Note that `RSTTree` and `SimpleRSTTree` share the same
    `Node` type but because of the subtle difference in
    interpretation you should be extremely careful not to
    mix and match.
    """

    def __init__(self, node, children, origin=None):
        """
        Note, you should use `SimpleRSTTree.from_RSTTree(tree)`
        to create this tree instead
        """
        SearchableTree.__init__(self, node, children)
        Standoff.__init__(self, origin)
        # WIP 2016-11-10 store num of head in node
        if len(children) == 1 and isinstance(children[0], EDU):
            node.head = children[0].num
        else:
            # head is head of the leftmost nucleus child
            lnuc_idx = node.nuclearity.index('N')
            node.head = children[lnuc_idx].label().head
        # end WIP head

    def set_origin(self, origin):
        """Recursively update the origin for this annotation, ie.
        a little link to the document metadata for this annotation.

        Parameters
        ----------
        origin : FileId
            File identifier of the origin of this annotation.
        """
        self.origin = origin
        for child in self:
            child.set_origin(origin)

    def text_span(self):
        return treenode(self).span

    def _members(self):
        return list(self)  # children

    def get_spans(self, subtree_filter=None, exclude_root=False,
                  span_type='edus'):
        """Get the spans of a constituency tree.

        Each span is described by a triplet (edu_span, nuclearity,
        relation).

        Parameters
        ----------
        subtree_filter : function, defaults to None
            Function to filter all local trees.

        exclude_root : boolean, defaults to False
            If True, exclude the span of the root node. This cannot be
            expressed with `subtree_filter` because the latter is limited
            to properties local to each subtree in isolation. Or maybe I
            just missed something.

        span_type : one of {'edus', 'chars'}
            Whether each span is expressed on EDU or character indices.
            Character indices are useful to compare spans from trees
            whose EDU segmentation differs.

        Returns
        -------
        spans: list of tuple((int, int), str, str)
            List of tuples, each describing a span with a tuple
            ((edu_start, edu_end), nuclearity, relation).
        """
        tnodes = [x.label() for x in self.subtrees(filter=subtree_filter)
                  if isinstance(x, SimpleRSTTree)]
        if exclude_root:
            tnodes = tnodes[1:]
        # 2016-11-10 add a 4th element: head
        # 2017-04-12 enable char spans
        if span_type == 'chars':
            spans = [((tn.span.char_start, tn.span.char_end),
                      tn.nuclearity, tn.rel, tn.head)
                     for tn in tnodes]
        else:
            spans = [(tn.edu_span, tn.nuclearity, tn.rel, tn.head)
                     for tn in tnodes]
        return spans

    @classmethod
    def from_rst_tree(cls, tree):
        """
        Build and return a `SimpleRSTTree` from an `RSTTree`
        """
        return cls._from_binary_rst_tree(_binarize(tree))

    @classmethod
    def _from_binary_rst_tree(cls, tree):
        """
        Helper to from_rst_tree; hoist the relation from the
        satellite node to the parent. If there is no satellite
        (ie. we have a multinuclear relation), take it from the
        left node.
        """
        if len(tree) == 1:
            node = copy.copy(treenode(tree))
            node.rel = "leaf"
            node.nuclearity = "leaf"  # WIP
            return SimpleRSTTree(node, tree, tree.origin)
        else:
            left = tree[0]
            right = tree[1]
            node = copy.copy(treenode(tree))
            lnode = treenode(left)
            rnode = treenode(right)
            node.rel = rnode.rel if rnode.is_satellite() else lnode.rel
            # WIP move nuclearity up too
            node.nuclearity = ''.join(x.label().nuclearity[0] for x in tree)
            # end WIP
            kids = [cls._from_binary_rst_tree(kid) for kid in tree]
            return SimpleRSTTree(node, kids, tree.origin)

    @classmethod
    def incorporate_nuclearity_into_label(cls, tree):
        """Integrate nuclearity of the children into each node's label.

        Nuclearity of the children is incorporated in one of two forms,
        NN for multi- and NS for mono-nuclear relations.

        Parameters
        ----------
        tree: SimpleRSTTree
            The tree of which we want a version with nuclearity incorporated

        Returns
        -------
        mod_tree: SimpleRSTTree
            The same tree but with the type of nuclearity incorporated

        Note
        ----
        This is probably not the best way to provide this functionality.
        In other words, refactoring is much needed here.
        """
        if len(tree) == 1:
            node = copy.copy(treenode(tree))
            return SimpleRSTTree(node, tree, tree.origin)
        else:
            node = copy.copy(treenode(tree))
            # convenient string representation of what the children look like
            # here one of NS, SN, NN
            nscode = "".join(treenode(kid).nuclearity[0] for kid in tree)
            assert nscode in frozenset(['NS', 'SN', 'NN'])
            rel_sfx = 'NS' if nscode in ['NS', 'SN'] else 'NN'
            # or rel_sfx = nscode if... to get the same 41 relations as Joty
            node.rel = node.rel + '-' + rel_sfx
            # recurse
            kids = [cls.incorporate_nuclearity_into_label(kid)
                    for kid in tree]
            return SimpleRSTTree(node, kids, tree.origin)

    @classmethod
    def to_binary_rst_tree(cls, tree, rel='---', nuc=NUC_R):
        """
        Build and return a binary `RSTTree` from a `SimpleRSTTree`.

        This function is recursive, it essentially pushes the
        relation label from the parent to the satellite child
        (for mononuclear relations) or to all nucleus children
        (for multinuclear relations).

        Parameters
        ----------
        tree : SimpleRSTTree
            SimpleRSTTree to convert

        rel : string, optional
            Relation for the root node of the output

        nuc : string, optional
            Nuclearity for the root node of the output

        Returns
        -------
        rtree : RSTTree
            The (binary) RSTTree that corresponds to the given
            SimpleRSTTree
        """
        if len(tree) == 1:
            node = copy.copy(treenode(tree))
            node.rel = rel
            node.nuclearity = nuc
            return RSTTree(node, tree, tree.origin)
        else:
            node = copy.copy(treenode(tree))
            # standard RST trees mark relations on the satellite
            # child (mononuclear relations) or on each nucleus
            # child (multinuclear relations)
            sat_idx = [i for i, nuc0 in enumerate(node.nuclearity)
                       if nuc0 == NUC_S[0]]
            if sat_idx:
                # mononuclear
                kids = [
                    cls.to_binary_rst_tree(
                        kid,
                        rel=(node.rel if node.nuclearity[i] == NUC_S[0]
                             else 'span'),
                        nuc=(NUC_S if node.nuclearity[i] == NUC_S[0]
                             else NUC_N))
                    for i, kid in enumerate(tree)
                ]
            else:
                # multinuclear
                kids = [cls.to_binary_rst_tree(kid, rel=node.rel, nuc=NUC_N)
                        for kid in tree]
            # update the rel and nuc in the current node
            node.rel = rel
            node.nuclearity = nuc
            return RSTTree(node, kids, tree.origin)


def _chain_to_binary(rel, kids):
    """
    (binarize helper)

    Fold a list of RST trees into a single binary tree given a relation
    that is expected to hold over each consequenctive pair of subtrees.
    """

    def builder(right, left):
        "function to fold with"
        lnode = treenode(left)
        rnode = treenode(right)
        edu_span = (lnode.edu_span[0], rnode.edu_span[1])
        span = lnode.span.merge(rnode.span)
        newnode = Node(NUC_N, edu_span, span, rel)
        return RSTTree(newnode, [left, right], origin=left.origin)
    return functools.reduce(builder, kids[::-1])


def is_binary(tree):
    """
    True if the given RST tree or SimpleRSTTree is indeed binary
    """
    if isinstance(tree, EDU):
        return True
    elif len(tree) > 2:
        return False
    else:
        return all(is_binary(x) for x in tree)


def _binarize(tree, branching='right_mixed'):
    """
    Slightly rearrange an RST tree as a binary tree.  The non-trivial
    cases here are

    * `X(sns) => X(N(sn),s)` Given a hypotactic relation with exactly two
      satellites (left and right), lower the left most satellite-nucleus
      pair into a subtree with a nuclear head.  As an example, given
      `X(attribution:S1, N, explanation-argumentative:S2)`, we would
      return something like this:
      `X(span:N(attribution:S1, N), explanation-argumentative:S2)`,


    * `X(nnn...)` => X(n,N(n,N(...))) (multi-nuclear, 0 satellites)
      Straightforwardly build a chain of cons cells glued together
      by new Nuclear nodes.

      For example, given `X(List:N1, List:N2, List:N3)`, we would
      return `X(List:N1, List:N(List:N2, List:N3))`

    Parameters
    ----------
    branching : str, one of {'left', 'right', 'right_mixed'}
        Direction of the branching ; defaults to 'right_mixed', which
        transforms n-ary multinuclear relations to a cascade of
        right-branching binary trees, and SNS n-ary nodes into
        left-branching binary trees.
    """
    branching_vals = ('left', 'right', 'right_mixed')
    if branching not in branching_vals:
        raise ValueError("branching must be one of {{}}".format(
            branching_vals))

    if isinstance(tree, EDU):
        return tree
    elif len(tree) == 1 and not isinstance(tree[0], EDU):
        raise RSTTreeException("Ill-formed RST tree? Unary non-terminal: " +
                               str(tree))
    elif len(tree) <= 2:
        return RSTTree(treenode(tree),
                       [_binarize(x, branching=branching) for x in tree],
                       origin=tree.origin)
    else:
        # convenient string representation of what the children look like
        # eg. NS, SN, NNNNN, SNS
        nscode = "".join(treenode(kid).nuclearity[0] for kid in tree)

        nuclei = [kid for kid in tree if treenode(kid).is_nucleus()]
        satellites = [kid for kid in tree if treenode(kid).is_satellite()]
        if len(nuclei) + len(satellites) != len(tree):
            raise Exception("Nodes that are neither Nuclei nor Satellites\n%s"
                            % tree)

        if len(nuclei) == 0:
            raise Exception("No nucleus:\n%s" % tree)
        elif len(nuclei) > 1:  # multi-nuclear chain
            if satellites:
                raise Exception("Multinuclear with satellites:\n%s" % tree)
            kids = [_binarize(x, branching=branching) for x in tree]
            if branching in ('right', 'right_mixed'):  # right-branching
                left = kids[0]
                right = _chain_to_binary(treenode(left).rel, kids[1:])
            else:  # left-branching
                right = kids[-1]
                left = _chain_to_binary(treenode(right).rel, kids[:-1])
            return RSTTree(treenode(tree), [left, right], origin=tree.origin)
        elif nscode == 'SNS':
            if branching in ('left', 'right_mixed'):  # left-branching
                left = _chain_to_binary('span', tree[:2])
                right = _binarize(tree[2], branching=branching)
            else:  # right-branching
                left = _binarize(tree[0], branching=branching)
                right = _chain_to_binary('span', tree[1:])
            return RSTTree(treenode(tree), [left, right], origin=tree.origin)
        else:
            raise RSTTreeException(
                ("Don't know how to handle %s trees" % nscode))
