from __future__ import annotations
import os
import itertools
import functools
import re
import unicodedata

def cmp(str1, str2):
    """Rough, slow implementation of `new Intl.Collator('en', { numeric: true, sensitivity: 'base' }).compare`"""
    # split s into [string, number, string, ...]
    def split_numeric(s: str) -> list[str]:
        return re.split(r'(\d+)', s)

    # Compare numeric values if present, otherwise compare strings
    def compare_parts(parts1: list[str], parts2: list[str]):
        for a, b in zip(parts1, parts2):
            if a.isdigit() and b.isdigit():
                a, b = int(a), int(b)
                if a != b:
                    return 1 if a > b else -1
            elif a != b:
                return 1 if a > b else -1
        return 0
    
    # sensitivity: base means case insensitive
    str1 = latinize(str(str1)).lower()
    str2 = latinize(str(str2)).lower()

    parts1 = split_numeric(str1)
    parts2 = split_numeric(str2)
    return compare_parts(parts1, parts2)

# https://github.com/leeoniya/uFuzzy/blob/1.0.14/src/uFuzzy.js#L880
_latinize_accents = {
    'A': 'ÁÀÃÂÄĄ',
    'a': 'áàãâäą',
    'E': 'ÉÈÊËĖ',
    'e': 'éèêëę',
    'I': 'ÍÌÎÏĮ',
    'i': 'íìîïį',
    'O': 'ÓÒÔÕÖ',
    'o': 'óòôõö',
    'U': 'ÚÙÛÜŪŲ',
    'u': 'úùûüūų',
    'C': 'ÇČĆ',
    'c': 'çčć',
    'L': 'Ł',
    'l': 'ł',
    'N': 'ÑŃ',
    'n': 'ñń',
    'S': 'ŠŚ',
    's': 'šś',
    'Z': 'ŻŹ',
    'z': 'żź'
}
_latinize_rev = {accent: latin for latin, accents in _latinize_accents.items() for accent in accents}
_latinize_trans = str.maketrans(_latinize_rev)

@functools.lru_cache()
def _latinize(string: str):
    string = string.replace("ł", "l").replace("Ł", "L") # unicode normalization doesn't work on these...
    nfkd_form = unicodedata.normalize('NFKD', string)
    ascii_text = nfkd_form.encode('ASCII', 'ignore').decode('utf-8')
    if len(ascii_text) == len(string):
        return ascii_text
    else:
        # fallback in case unicode normalization drops a character
        return str.translate(string, _latinize_trans)

def latinize(strings: list[str] | str):
    """util for replacing common diacritics/accents"""
    if isinstance(strings, str):
        return _latinize(strings)
    else:
        return [_latinize(s) for s in strings]


def _mark(part: str, matched: bool):
    return f"<mark>{part}</mark>" if matched else part

def _append(acc, part):
    return acc + part

def highlight(str, ranges: list, mark=_mark, accum='', append=_append):
    """util for highlighting matched substr parts of a result"""
    accum = append(accum, mark(str[0:ranges[0]], False)) or accum

    for i in range(0, len(ranges), 2):
        fr = ranges[i]
        to = ranges[i+1]

        accum = append(accum, mark(str[fr:to], True)) or accum

        if i < len(ranges) - 3:
            accum = append(accum, mark(str[ranges[i+1]:ranges[i+2]], False)) or accum

    accum = append(accum, mark(str[ranges[-1]:], False)) or accum

    return accum

def permute(arr: list):
    """util for creating out-of-order permutations of a needle terms array"""
    # TODO: need to test that this is equivalent to js algorithm for stability
    idxs = list(itertools.permutations(range(len(arr))))
    idxs.sort()
    return [[arr[i] for i in pi] for pi in idxs]

class MatchLike:
    __slots__ = ('_start', '_end', '_groups', 'pos', 'endpos')
    def __init__(self, m: re.Match | MatchLike, groups: list[str] | None = None):
        if m is None: return

        self.pos = m.pos
        self.endpos = m.endpos
        # self.string = m.string
        # self.re = m.re
        # self.regs = m.regs
        # self.lastgroup = m.lastgroup
        # self.lastindex = m.lastindex
        # self._m = m
        # m.span()
        self._start: int = m.start() or 0
        self._end: int
        self._groups: list[str]
        if groups is None:
            self._groups = list(m.groups())
            self._end = m.end() or 0
        else:
            if groups[0] == m.group(0) and len(groups) > len(m.groups()):
                groups = groups[1:]
            self._groups = groups
            end = m.end() or 0
            orig_len = sum(len(g) for g in m.groups())
            new_len = sum(len(g) for g in groups)
            end += new_len - orig_len
            self._end = end
    
    # @functools.lru_cache(maxsize=1)
    # @functools.cached_property
    def groups(self) -> list[str]:
        return self._groups
    
    # @functools.lru_cache(maxsize=8)
    # @functools.cached_property
    def group(self, g: int = 0) -> str:
        return self._groups[g]
    
    # @functools.lru_cache(maxsize=1)
    # @functools.cached_property
    def start(self): #,  __group: int | str = 0, /):
        return self._start
        # return self._m.start()
    
    # @functools.lru_cache(maxsize=1)
    # @functools.cached_property
    def end(self): #, __group: int | str = 0, /):
        return self._end
        # return self._m.end()
    
    def span(self): #, __group: int | str = 0, /):
        return (self._start, self._end)
