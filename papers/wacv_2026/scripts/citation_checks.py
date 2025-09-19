#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "ubelt",
#   "pylatexenc",
#   "xdev",
#   "rich",
# ]
# ///
import ubelt as ub
import pylatexenc
from pylatexenc.latexnodes.nodes import LatexNodesVisitor
from pylatexenc.latex2text import LatexNodes2Text
from pylatexenc.latexnodes.parsers import LatexGeneralNodesParser


class CitationVisitor(LatexNodesVisitor):
    """
    SeeAlso:
        https://pylatexenc.readthedocs.io/en/latest/latexnodes.nodes/#pylatexenc.latexnodes.nodes.LatexNodesVisitor
        ~/code/pylatexenc/pylatexenc/latexnodes/_latex_recomposer.py
        ~/code/pylatexenc/pylatexenc/latexnodes/nodes.py
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.citations = []
        ...

    VERBOSE = 0

    def visit_chars_node(self, node, **kwargs):
        if self.VERBOSE:
            print('visit_chars_node')
        ...

    def visit_group_node(self, node, visited_results_nodelist, **kwargs):
        if self.VERBOSE:
            print('visit_group_node')
        ...

    def visit_comment_node(self, node, **kwargs):
        if self.VERBOSE:
            print('visit_group_node')
        ...

    def visit_macro_node(self, node, visited_results_arguments, **kwargs):
        if self.VERBOSE:
            print('visit_group_node')
        if node.macroname in {'cite'}:
            self.citations.append(node.latex_verbatim())
            print(node.latex_verbatim())

    def visit_environment_node(self, node, visited_results_arguments,
                               visited_results_body, **kwargs):
        if self.VERBOSE:
            print('visit_environment_node')
        ...

    def visit_specials_node(self, node, visited_results_arguments, **kwargs):
        if self.VERBOSE:
            print('visit_specials_node')
        ...

    def visit_math_node(self, node, visited_results_nodelist, **kwargs):
        if self.VERBOSE:
            print('visit_math_node')

    def visit_node_list(self, nodelist, visited_results_nodelist, **kwargs):
        if self.VERBOSE:
            print('visit_node_list')

    def visit_parsed_arguments(self, parsed_args, visited_results_argnlist, **kwargs):
        if self.VERBOSE:
            print('visit_parsed_arguments')

    def visit_unknown_node(self, node, **kwargs):
        if self.VERBOSE:
            print('visit_unknown_node')


def main():
    # from pylatexenc.latexwalker import LatexWalker
    document_fpath = ub.Path('~/code/shitspotter/papers/wacv_2026/main.tex').expanduser()

    latex_text = document_fpath.read_text()
    print(LatexNodes2Text().latex_to_text(latex_text))

    walker = pylatexenc.latexwalker.LatexWalker(latex_text)
    parser = LatexGeneralNodesParser()
    result, parser_parsing_state_delta = walker.parse_content(parser)

    visitor = CitationVisitor()
    visitor.start(result)

    import pylatex
    # Dont use fontenc, lmodern, or textcomp
    # https://tex.stackexchange.com/questions/179778/xelatex-under-ubuntu
    doc = pylatex.Document('citation_check', inputenc=None,
                           page_numbers=False, indent=False, fontenc=None,
                           lmodern=1,
                           textcomp=False,
                           # geometry_options='paperheight=10in,paperwidth=18in,margin=.1in',
                           )
    doc.preamble.append(pylatex.Package('hyperref'))  # For PNG images
    doc.append(pylatex.Section('Citations'))
    doc.append(pylatex.NoEscape(
        ' '.join(visitor.citations)
    ))
    doc.append(pylatex.NoEscape(ub.codeblock(
        r'''
        {
         \bibliographystyle{ieee_fullname}
         \bibliography{citations}
        }
        ''')
    ))
    print(doc.dumps())
    print('generate pdf')

    pdf_fpath = ub.Path('~/code/shitspotter/papers/wacv_2026/citation_checks.pdf').expand()
    pdf_fpath.parent.ensuredir()
    doc.generate_pdf(pdf_fpath.augment(ext=''), clean_tex=True)

    import xdev
    xdev.startfile(pdf_fpath)


if __name__ == '__main__':
    """
    CommandLine:
        python ~/code/shitspotter/papers/wacv_2026/citation_checks.py
    """
    main()
