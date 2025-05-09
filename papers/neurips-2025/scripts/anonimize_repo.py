"""
Anonymize supplemental materials for peer review
"""


def main():
    import ubelt as ub
    import shitspotter
    init_fpath = ub.Path(shitspotter.__file__)
    module_dpath = init_fpath.parent
    repo_dpath = module_dpath.parent

    anon_dpath = repo_dpath.parent / f'anonimized-repo-{ub.timestamp()}'
    anon_dpath.ensuredir()

    if not (repo_dpath / '.git').exists():
        raise AssertionError('Can only run this on the actual repo')

    info = ub.cmd('git ls-files .', cwd=repo_dpath)
    tracked_rel_paths = info.stdout.strip().split('\n')

    # Dont anonimize these files, because that may actually deanonmize us!
    blocklist = {
        'citations.bib',
    }

    import kwutil
    import xdev
    idenfifiers = []
    idenfifiers.append({
        'patterns': [
            kwutil.Pattern.from_regex('Jon Crall', ignorecase=True),
            kwutil.Pattern.from_regex('joncrall', ignorecase=True),
            kwutil.Pattern.from_regex('erotemic', ignorecase=True),
            kwutil.Pattern.from_regex('jon.crall', ignorecase=True),
            kwutil.Pattern.from_regex('Jonathan Crall', ignorecase=True),
        ],
        'replacement': '<ANONIMIZED_AUTHOR>',
    })

    idenfifiers.append({
        'patterns': [
            kwutil.Pattern.from_regex('Anthony Hoogs', ignorecase=True),
        ],
        'replacement': '<ANONIMIZED_PERSON>',
    })

    idenfifiers.append({
        'patterns': [
            kwutil.Pattern.from_regex('albany', ignorecase=True),
            kwutil.Pattern.from_regex('new york', ignorecase=True),
            kwutil.Pattern.from_regex('\\b' + 'NY' + '\\b', ignorecase=True),
        ],
        'replacement': '<ANONIMIZED_LOCATION>',
    })

    idenfifiers.append({
        'patterns': [
            kwutil.Pattern.from_regex('Kitware', ignorecase=True),
        ],
        'replacement': '<ANONIMIZED_ORGANIZATION>',
    })

    for p in anon_dpath.glob('*'):
        p.delete()

    for rel_path in tracked_rel_paths:
        src_path = repo_dpath / rel_path
        if src_path.name in blocklist:
            continue
        dst_path = anon_dpath / rel_path
        if src_path.is_file():
            try:
                orig_text = src_path.read_text()
            except Exception:
                print(f'Failed to read src_path = {ub.urepr(src_path, nl=1)}')
                raise
            text = orig_text
            for idpats in idenfifiers:
                for pat in idpats['patterns']:
                    text = pat.sub(idpats['replacement'], text)
            for idpats in idenfifiers:
                for pat in idpats['patterns']:
                    if pat.search(text):
                        raise Exception
            if text != orig_text:
                print(f'Anonimized rel_path={rel_path}')
                print(xdev.difftext(orig_text, text, colored=True, context_lines=3))
            dst_path.parent.ensuredir()
            dst_path.write_text(text)
    import rich
    rich.print(f'Pred Dpath: [link={anon_dpath}]{anon_dpath}[/link]')


if __name__ == '__main__':
    """
    CommandLine:
        python ~/code/shitspotter/papers/neurips-2025/scripts/anonimize_repo.py
    """
    main()
