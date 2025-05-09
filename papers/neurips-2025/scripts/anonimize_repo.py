"""
Anonymize supplemental materials for peer review
"""


import fnmatch
import re
from pathlib import Path
from typing import Union, Dict, Optional


class GitAttributes:
    """
    References:
        https://chat.deepseek.com/a/chat/s/6b61006c-ef0a-4fe7-94ff-238714c94c40
    """
    def __init__(self, patterns):
        """Initialize with pre-parsed patterns."""
        self.patterns = patterns

    @classmethod
    def from_text(cls, text: str) -> 'GitAttributes':
        """
        Create a parser from a string containing gitattributes content.

        Args:
            text: String containing the contents of a .gitattributes file

        Returns:
            GitAttributesParser instance
        """
        patterns = cls._parse_gitattributes_text(text)
        return cls(patterns)

    @classmethod
    def from_file(cls, file_path: Union[str, Path]) -> 'GitAttributes':
        """
        Create a parser from a .gitattributes file.

        Args:
            file_path: Path to the .gitattributes file

        Returns:
            GitAttributes instance

        Raises:
            FileNotFoundError: If the file doesn't exist
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Gitattributes file not found: {file_path}")

        with open(path, 'r', encoding='utf-8') as f:
            text = f.read()

        return cls.from_text(text)

    @staticmethod
    def _parse_gitattributes_text(text: str) -> list:
        """Parse gitattributes text and return patterns with attributes."""
        patterns = []

        for line in text.splitlines():
            line = line.strip()
            # Skip comments and empty lines
            if not line or line.startswith('#'):
                continue

            # Split pattern from attributes
            parts = re.split(r'\s+', line, maxsplit=1)
            if len(parts) == 0:
                continue

            pattern = parts[0]
            attributes = {}
            if len(parts) > 1:
                # Parse attributes like filter=myfilter, diff, merge, etc.
                for attr in parts[1].split():
                    if '=' in attr:
                        key, value = attr.split('=', 1)
                        attributes[key] = value
                    else:
                        attributes[attr] = True

            patterns.append((pattern, attributes))

        return patterns

    def get_filters_for_file(self, file_path: Union[str, Path]) -> Dict[str, str]:
        """
        Return all filter attributes that apply to the given file path.

        Args:
            file_path: The file path to check against the patterns

        Returns:
            Dictionary of filter attributes that apply to the file
            (key is the filter name, value is the filter spec)
        """
        file_path = str(file_path)
        filters = {}

        for pattern, attributes in self.patterns:
            # Convert gitattributes pattern to fnmatch pattern
            # Handle directory patterns (ending with /)
            if pattern.endswith('/'):
                pattern = pattern.rstrip('/') + '/*'

            # Handle negation patterns (starting with !)
            negate = False
            if pattern.startswith('!'):
                negate = True
                pattern = pattern[1:]

            # Check if the file matches the pattern
            if fnmatch.fnmatch(file_path, pattern):
                if negate:
                    # Remove any previously matched filters
                    for key in list(filters.keys()):
                        if key.startswith('filter='):
                            del filters[key]
                else:
                    # Add all filter attributes
                    for key, value in attributes.items():
                        if key == 'filter' or key.startswith('filter='):
                            filters[key] = value

        return filters

    def get_filter_spec_for_file(self, file_path: Union[str, Path]) -> Optional[str]:
        """
        Get the specific filter spec (the value after filter=) for a file.

        Args:
            file_path: The file path to check

        Returns:
            The filter spec if found, None otherwise
        """
        filters = self.get_filters_for_file(file_path)
        return filters.get('filter')


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

    attributes = (repo_dpath / '.gitattributes')
    gitattr = GitAttributes.from_file(attributes)

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

        spec = gitattr.get_filter_spec_for_file(rel_path)
        if spec == 'crypt':
            # Skip any encrypted files
            continue

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
