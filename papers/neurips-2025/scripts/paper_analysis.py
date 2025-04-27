"""
Latest attempt at some

References:
    https://pylatexenc.readthedocs.io/en/latest/
    https://github.com/phfaist/pylatexenc

Requirements:
    pip install "pylatexenc>=3.0a30"
"""
import rich
import ubelt as ub
import pylatexenc
from pylatexenc.latex2text import LatexNodes2Text
from pylatexenc.latexnodes.parsers import LatexGeneralNodesParser
# from pylatexenc.latexwalker import LatexWalker
document_fpath = ub.Path('~/code/shitspotter/papers/neurips-2025/main.tex').expanduser()

latex_text = document_fpath.read_text()
print(LatexNodes2Text().latex_to_text(latex_text))

walker = pylatexenc.latexwalker.LatexWalker(latex_text)
nodes = walker.get_latex_nodes()

parser = LatexGeneralNodesParser()
result, parser_parsing_state_delta = walker.parse_content(parser)


len(result)
for item in result:
    print(type(item))
    print(str(item)[0:80])


walker = ub.IndexableWalker(result, list_cls=(list, tuple, pylatexenc.latexnodes.nodes.LatexNodeList))

# for p, v in walker:
#     if isinstance(v, pylatexenc.latexnodes.nodes.LatexCommentNode):
#         continue
#     length = v.pos_end - v.pos
#     print(p, length, type(v), v.display_str())


v = result[110]

for item in result:
    if isinstance(item, pylatexenc.latexnodes.nodes.LatexEnvironmentNode):
        break


walker = ub.IndexableWalker(item.nodelist, list_cls=(list, tuple, pylatexenc.latexnodes.nodes.LatexNodeList))


class Accumulator:
    def __init__(accum):
        accum.section_toc = []
        accum.new_section(None, None)

    def new_section(accum, type, name):
        accum.current = {'type': type, 'name': name, 'accum': []}
        accum.section_toc.append(accum.current)

    def append_text(self, text):
        accum.current['accum'].append(text)
        ...


accum = Accumulator()

type_to_nodes = ub.ddict(list)
for p, item in walker:
    if isinstance(item, pylatexenc.latexnodes.nodes.LatexCommentNode):
        continue
    if isinstance(item, pylatexenc.latexnodes.nodes.LatexEnvironmentNode):
        if item.environmentname == 'comment':
            continue
        if item.environmentname in {'figure', 'table', 'figure*', 'table*'}:
            continue

    if isinstance(item, pylatexenc.latexnodes.nodes.LatexMacroNode):
        if item.macroname in {'label'}:
            continue
        if item.macroname in {'textbf', 'emph'}:
            ...
        if item.macroname in {'section', 'subsection', 'subsubsection'}:
            assert len(item.nodeargs) == 1
            arg = item.nodeargs[0]
            name = arg.nodelist[0].latex_verbatim()
            accum.new_section(item.macroname, name)
            continue
        if item.macroname in {'cite'}:
            for arg in item.nodeargs:
                if arg is not None:
                    cite_text = arg.nodelist.latex_verbatim()
                    accum.append_text(f'[{cite_text}]')
            continue

    accum.append_text(item.latex_verbatim())

    length = item.pos_end - item.pos
    print(p, length, type(item), item.display_str())
    type_to_nodes[type(item).__name__].append(item)


for toc_item in accum.section_toc:
    toc_item['type']
    name = toc_item['name']
    rich.print(f'\n\n[white]--- {name} --- \n')
    body = ''.join(toc_item['accum'])
    import re
    parts = re.split('\n *\n *', body)
    parts = [p.strip() for p in parts]
    parts = [p for p in parts if p]
    new_body = '\n\n'.join(parts)
    print(new_body)


histo = ub.udict(type_to_nodes).map_values(len)
print(f'histo = {ub.urepr(histo, nl=1)}')

type_to_nodes['LatexSpecialsNode']
type_to_nodes['LatexMacroNode']
type_to_nodes['LatexCharsNode']
type_to_nodes['LatexEnvironmentNode']
type_to_nodes['LatexMathNode']

for item in type_to_nodes['LatexSpecialsNode']:
    # print(item.specials_chars)
    orig_text = latex_text[item.pos:item.pos_end]
    print(f'orig_text = {ub.urepr(orig_text, nl=1)}')

for item in type_to_nodes['LatexMacroNode']:
    print(item.macroname)
    orig_text = item.latex_verbatim()
    print(f'orig_text = {ub.urepr(orig_text, nl=1)}')

for item in type_to_nodes['LatexEnvironmentNode']:
    orig_text = item.latex_verbatim()
    print(item.environmentname)
    print(f'orig_text = {ub.urepr(orig_text, nl=1)}')

for item in type_to_nodes['LatexMathNode']:
    orig_text = item.latex_verbatim()
    print(f'orig_text = {ub.urepr(orig_text, nl=1)}')

for item in type_to_nodes['LatexGroupNode']:
    orig_text = item.latex_verbatim()
    print(f'orig_text = {ub.urepr(orig_text, nl=1)}')
