from __future__ import annotations
import os
import re
import sys
import math
import functools
import time
from dataclasses import dataclass, field
from enum import Enum, IntEnum
from typing import Tuple, List, Dict, Optional, Union

from helpers import *

dataclass_slots = dataclass(slots=True) if sys.version_info >= (3, 10) else dataclass

# ---------------------------------------------------------------------
# Types
# https://github.com/leeoniya/uFuzzy/blob/1.0.14/dist/uFuzzy.d.ts#L60

# for python <= 3.8, need to use typing.X in type aliases like this
Terms = List[str]
"""needle's terms"""
HaystackIdxs = List[int]
"""subset of idxs of a haystack array"""
InfoIdxOrder = List[int]
"""sorted order in which info facets should be iterated"""
AbortedResult = Tuple[None, None, None]
FilteredResult = Tuple[HaystackIdxs, None, None]

PartialRegExp = str

class BoundMode(IntEnum):
    """what should be considered acceptable term bounds"""
    Any = 0
    """will match 'man' substr anywhere. e.g. tasmania"""
    Loose = 1
    """will match 'man' at whitespace, punct, case-change, and alpha-num boundaries. e.g. mantis, SuperMan, fooManBar, 0007man"""
    Strict = 2
    """will match 'man' at whitespace, punct boundaries only. e.g. mega man, walk_man, man-made, foo.man.bar"""

class IntraMode(IntEnum):
    MultiInsert = 0
    """allows any number of extra char insertions within a term, but all term chars must be present for a match"""
    SingleError = 1
    """allows for a single-char substitution, transposition, insertion, or deletion within terms (excluding first and last chars)"""

IntraSliceIdxs = Tuple[Optional[int], Optional[int]]

# ---------------------------------------------------------------------

DEBUG = False

def escapeRegExp(str):
    return re.escape(str)

# meh, magic tmp placeholder, must be tolerant to toLocaleLowerCase(), interSplit, and intraSplit
EXACT_HERE = 'eexxaacctt'

LATIN_UPPER: PartialRegExp = 'A-Z'
LATIN_LOWER: PartialRegExp = 'a-z'

# https://github.com/leeoniya/uFuzzy/blob/1.0.14/src/uFuzzy.js#L16
def swapAlpha(string: PartialRegExp, upper: str, lower: str):
    return string.replace(LATIN_UPPER, upper).replace(LATIN_LOWER, lower)

# https://github.com/leeoniya/uFuzzy/blob/1.0.14/src/uFuzzy.js#L156
# https://github.com/leeoniya/uFuzzy/blob/1.0.14/dist/uFuzzy.d.ts#L148
@dataclass_slots
class IntraRules:
    intraSlice: IntraSliceIdxs
    intraIns: int | None
    intraSub: int
    intraTrn: int
    intraDel: int

# https://github.com/leeoniya/uFuzzy/blob/1.0.14/src/uFuzzy.js#L18
# https://github.com/leeoniya/uFuzzy/blob/1.0.14/dist/uFuzzy.d.ts#L102
class Options:
    def __init__(self):
        self.unicode: bool = False
        """whether regexps use a /u unicode flag"""

        self.alpha: PartialRegExp | None = None
        """regexp character class [] of chars which should be treated as letters (case insensitive)"""

        # term segmentation & punct/whitespace merging
        self.interSplit: PartialRegExp = r"[^A-Za-z\d']+"
        self.intraSplit: PartialRegExp | None = r"[a-z][A-Z]"

        self.intraBound: PartialRegExp | None = r"[A-Za-z]\d|\d[A-Za-z]|[a-z][A-Z]"
        """intra bounds that will be used to increase lft1/rgt1 info counters"""

        # inter-bounds mode
        # 2 = strict (will only match 'man' on whitepace and punct boundaries: Mega Man, Mega_Man, mega.man)
        # 1 = loose  (plus allowance for alpha-num and case-change boundaries: MegaMan, 0007man)
        # 0 = any    (will match 'man' as any substring: megamaniac)
        self.interLft: BoundMode = BoundMode.Any
        self.interRgt: BoundMode = BoundMode.Any

        # allowance between terms
        self.interChars: PartialRegExp = '.'
        self.interIns: int | None = None # Inf

        # allowance between chars in terms
        self.intraChars: PartialRegExp = r"[a-z\d']"
        self.intraIns: int | None = 0

        self.intraContr = r"'[a-z]{1,2}\b"

        self.intraMode: IntraMode = IntraMode.MultiInsert
        """multi-insert or single-error mode"""

        self.intraSlice: IntraSliceIdxs = (1, None) # [1, Infinity]

        # single-error tolerance toggles
        self.intraSub: int | None = 1
        """max substitutions (when intraMode: 1)"""
        self.intraTrn: int | None = 1
        """max transpositions (when intraMode: 1)"""
        self.intraDel: int | None = 1
        """max omissions/deletions (when intraMode: 1)"""

    # deprecated API
    # @property
    # def letters(self):
    #     return self.alpha
    # @letters.setter
    # def letters(self, value):
    #     return self.alpha = value

    def intraFilt(self, term: str, match: str, index: int) -> bool: # should this also accept WIP info?
        """can post-filter matches that are too far apart in distance or length

        (since intraIns is between each char, it can accum to nonsense matches)
        """
        return True

    def sort(self, info: Info, haystack: list[str], needle: str) -> InfoIdxOrder:
        # https://github.com/leeoniya/uFuzzy/blob/1.0.14/src/uFuzzy.js#L64
        idx, chars, terms, start = info.idx, info.chars, info.terms, info.start
        interLft2, interLft1 = info.interLft2, info.interLft1
        # interRgt2, interRgt1 = info.interRgt2, info.interRgt1
        intraIns, interIns = info.intraIns, info.interIns

        def sort_key(ia, ib):
            return (
                # most contig chars matched
                (chars[ib] - chars[ia]) or
                # least char intra-fuzz (most contiguous)
                (intraIns[ia] - intraIns[ib]) or
                # most prefix bounds, boosted by full term matches
                (
                    (terms[ib] + interLft2[ib] + 0.5 * interLft1[ib]) -
                    (terms[ia] + interLft2[ia] + 0.5 * interLft1[ia])
                ) or
                # highest density of match (least span)
            # span[ia] - span[ib] or
                # highest density of match (least term inter-fuzz)
                (interIns[ia] - interIns[ib]) or
                # earliest start of match
                (start[ia] - start[ib]) or
                # alphabetic
                cmp(haystack[idx[ia]], haystack[idx[ib]])
            )
        idxs = list(range(len(idx)))
        idxs.sort(key=functools.cmp_to_key(sort_key))
        return idxs

    def intraRules(self, p: str) -> IntraRules:
        """can dynamically adjust error tolerance rules per term in needle (when intraMode: 1)"""
        # https://github.com/leeoniya/uFuzzy/blob/1.0.14/src/uFuzzy.js#L156
        # https://github.com/leeoniya/uFuzzy/blob/1.0.14/dist/uFuzzy.d.ts#L148
        # default is exact term matches only
        _intra_slice = self.intraSlice # requires first char
        _intra_ins = 0
        _intra_sub = 0
        _intra_trn = 0
        _intra_del = 0

        # only-digits strings should match exactly, else special rules for short strings
        if re.search(r'[^\d]', p):
            plen = len(p)

            # prevent junk matches by requiring stricter rules for short terms
            if plen <= 4:
                if plen >= 3:
                    # one swap in non-first char when 3-4 chars
                    _intra_trn = min(self.intraTrn, 1)

                    # or one insertion when 4 chars
                    if plen == 4:
                        _intra_ins = min(self.intraIns, 1)
                # else exact match when 1-2 chars
            else:
                # use supplied opts
                _intra_slice = self.intraSlice
                _intra_ins = self.intraIns
                _intra_sub = self.intraSub
                _intra_trn = self.intraTrn
                _intra_del = self.intraDel
        
        return IntraRules(_intra_slice, _intra_ins, _intra_sub, _intra_trn, _intra_del)


def lazyRepeat(chars: str, limit: int | None) -> PartialRegExp:
    if limit == 0:
        return ''
    elif limit == 1:
        return chars + '??'
    elif limit is None or limit < 0 or math.isinf(limit):
        return chars + '*?'
    else:
        return f"{chars}{{0,{limit}}}?"

mode2Tpl: PartialRegExp = r"(?:\b|_)"

# @dataclass #(slots=True)
@dataclass_slots
class Info:
    # https://github.com/leeoniya/uFuzzy/blob/1.0.14/dist/uFuzzy.d.ts#L162
    idx: HaystackIdxs = field(default_factory=list)
    """matched idxs from haystack"""
    start: list[int] = field(default_factory=list)
    """match offsets"""

    interLft2: list[int] = field(default_factory=list)
    """number of left BoundMode.Strict term boundaries found"""
    interRgt2: list[int] = field(default_factory=list)
    """number of right BoundMode.Strict term boundaries found"""
    interLft1: list[int] = field(default_factory=list)
    """number of left BoundMode.Loose term boundaries found"""
    interRgt1: list[int] = field(default_factory=list)
    """number of right BoundMode.Loose term boundaries found"""

    intraIns: list[int] = field(default_factory=list)
    """total number of extra chars matched within all terms. higher = matched terms have more fuzz in them"""
    interIns: list[int] = field(default_factory=list)
    """total number of chars found in between matched terms. higher = terms are more sparse, have more fuzz in between them"""

    chars: list[int] = field(default_factory=list)
    """total number of matched contiguous chars (substrs but not necessarily full terms)"""

    terms: list[int] = field(default_factory=list)
    """number of exactly-matched terms (intra = 0) where both lft and rgt landed on a BoundMode.Loose or BoundMode.Strict boundary"""
    
    ranges: list[list[int] | None] = field(default_factory=list)
    """offset ranges within match for highlighting: [startIdx0, endIdx0, startIdx1, endIdx1,...]"""

    # https://github.com/leeoniya/uFuzzy/blob/1.0.14/src/uFuzzy.js#L402

RankedResult = Tuple[HaystackIdxs, Info, InfoIdxOrder]
SearchResult = Union[FilteredResult, RankedResult, AbortedResult]

# API goal: https://github.com/leeoniya/uFuzzy/blob/1.0.14/dist/uFuzzy.d.ts
class uFuzzy:
    def __init__(self, opts: Options | None = None):
        # https://github.com/leeoniya/uFuzzy/blob/1.0.14/src/uFuzzy.js#L110
        opts = opts or Options()
        self.opts = opts

        self.unicode = opts.unicode
        self.interLft = opts.interLft
        self.interRgt = opts.interRgt
        self.intraMode = opts.intraMode
        self.intraSlice = opts.intraSlice
        self.intraIns = opts.intraIns if opts.intraIns is not None else int(opts.intraMode)
        self.intraSub = opts.intraSub if opts.intraSub is not None else int(opts.intraMode)
        self.intraTrn = opts.intraTrn if opts.intraTrn is not None else int(opts.intraMode)
        self.intraDel = opts.intraDel if opts.intraDel is not None else int(opts.intraMode)
        self.intraContr = opts.intraContr
        self._intraSplit = opts.intraSplit or ''
        self._interSplit = opts.interSplit or ''
        self._intraBound = opts.intraBound or ''
        self.intraChars = opts.intraChars

        # self.alpha = opts.letters or opts.alpha
        self.alpha = opts.alpha

        if self.alpha:
            upper = self.alpha.upper() # toLocaleUpperCase
            lower = self.alpha.lower() # toLocaleLowerCase
            self._interSplit = swapAlpha(self._interSplit, upper, lower)
            self._intraSplit = swapAlpha(self._intraSplit, upper, lower)
            self._intraBound = swapAlpha(self._intraBound, upper, lower)
            self.intraChars = swapAlpha(self.intraChars, upper, lower)
            self.intraContr = swapAlpha(self.intraContr, upper, lower)
        
        quotedAny = r'".+?"'
        uFlag = re.UNICODE if self.unicode else 0
        iuFlag = uFlag | re.IGNORECASE
        self.EXACTS_RE = re.compile(quotedAny, iuFlag)
        self.NEGS_RE = re.compile(f"(?:\\s+|^)-(?:{self.intraChars}+|{quotedAny})", iuFlag)

        # https://github.com/leeoniya/uFuzzy/blob/1.0.14/src/uFuzzy.js#L155
        self.intraRules = opts.intraRules

        self.withIntraSplit = bool(self._intraSplit)
        
        self.intraSplit = re.compile(self._intraSplit, uFlag)
        self.interSplit = re.compile(self._interSplit, uFlag)

        self.trimRe = re.compile(f'^{self._interSplit}|{self._interSplit}$', uFlag)
        self.contrsRe = re.compile(self.intraContr, iuFlag)

        self.NUM_OR_ALPHA_RE = re.compile(r'[^\d]+|\d+', re.UNICODE)

        # https://github.com/leeoniya/uFuzzy/blob/1.0.14/src/uFuzzy.js#L399
        self.withIntraBound = bool(self._intraBound)
        self.interBound = re.compile(self._interSplit, uFlag)
        self.intraBound = re.compile(self._intraBound, uFlag)

    # https://github.com/leeoniya/uFuzzy/blob/1.0.14/src/uFuzzy.js#L208
    def split(self, needle: str) -> list[str]:
        exacts = []

        # TODO: this translation sucks
        needle = self.EXACTS_RE.sub(lambda m: (exacts.append(m.group(0)), EXACT_HERE)[-1], needle)

        needle = self.trimRe.sub('', needle).lower()

        if self.withIntraSplit:
            needle = self.intraSplit.sub(lambda m: m.group(0)[0] + ' ' + m.group(0)[1], needle)

        interSplit = self.interSplit
        return [exacts[j] if v == EXACT_HERE else v for j, v in enumerate(interSplit.split(needle)) if v]
    
    # https://github.com/leeoniya/uFuzzy/blob/1.0.14/src/uFuzzy.js#L227
    def prepQuery(self, needle: str, capt = 0, interOr = False) -> tuple[re.Pattern | None, list[str], list[str]]:
        # split on punct, whitespace, num-alpha, and upper-lower boundaries
        parts = self.split(needle)
        if not parts:
            return None, [], []
        
        contrsRe = self.contrsRe # regex to match contractions like "can't"
        intraMode, intraIns, intraSub, intraTrn, intraDel = self.intraMode, self.intraIns, self.intraSub, self.intraTrn, self.intraDel
        intraChars, interLft, interRgt = self.intraChars, self.interLft, self.interRgt
        NUM_OR_ALPHA_RE = self.NUM_OR_ALPHA_RE
        uFlag = re.UNICODE if self.unicode else 0
        iuFlag = uFlag | re.IGNORECASE
        
        # split out any detected contractions for each term that become required suffixes
        contrs = [''] * len(parts)
        for pi in range(len(parts)):
            p = parts[pi]
            if contrsRe.search(p):
                contrs[pi] = p
                parts[pi] = ''

        # array of regexp templates for each term
        reTpl = []

        # allows single mutations within each term
        if intraMode == IntraMode.SingleError: # == 1
            # TODO: this translation seems okay
            for pi, _p in enumerate(parts): # parts.map((p, pi) => {
                if not _p:
                    continue
                if _p[0] == '"': # unquote
                    reTpl.append(escapeRegExp(_p[1:-1]))
                else:
                    # split into numeric and alpha parts, so numbers are only matched as following punct or alpha boundaries, without swaps or insertions
                    resl = []
                    for m in NUM_OR_ALPHA_RE.finditer(_p): # for (let m of p.matchAll(NUM_OR_ALPHA_RE)) {
                        p = m.group(0) # let p = m[0];
                        
                        rules = self.intraRules(p)
                        intraSlice, intraIns, intraSub, intraTrn, intraDel = rules.intraSlice, rules.intraIns, rules.intraSub, rules.intraTrn, rules.intraDel
                        intraIns = intraIns or 0 # prevent None + 1

                        if intraIns + intraSub + intraTrn + intraDel == 0:
                            resl.append(p)
                            resl.append(contrs[pi])
                        else:
                            lftIdx, rgtIdx = intraSlice # idx may be None / inf
                            if lftIdx is None: lftIdx = 0
                            if rgtIdx is None: rgtIdx = len(p)

                            lftChar = p[:lftIdx] # prefix
                            rgtChar = p[rgtIdx:] # suffix
                            chars = p[lftIdx:rgtIdx]
                            # neg lookahead to prefer matching 'Test' instead of 'tTest' in ManifestTest or fittest
                            # but skip when search term contains leading repetition (aardvark, aaa)
                            if intraIns == 1 and len(lftChar) == 1 and lftChar != chars[0]:
                                lftChar += '(?!' + lftChar + ')'
                            numChars = len(chars)
                            variants = [p]

                            # variants with single char substitutions
                            if intraSub:
                                for i in range(numChars):
                                    variants.append(lftChar + chars[:i] + intraChars + chars[i+1:] + rgtChar)
                            # variants with single transpositions
                            if intraTrn:
                                for i in range(numChars - 1):
                                    if chars[i] != chars[i+1]:
                                        variants.append(lftChar + chars[:i] + chars[i+1] + chars[i] + chars[i+2:] + rgtChar)

                            # variants with single char omissions
                            if intraDel:
                                for i in range(numChars):
                                    variants.append(lftChar + chars[:i+1] + '?' + chars[i+1:] + rgtChar)

                            # variants with single char insertions
                            if intraIns:
                                intraInsTpl = lazyRepeat(intraChars, 1)
                                for i in range(numChars):
                                    variants.append(lftChar + chars[:i] + intraInsTpl + chars[i:] + rgtChar)

                            resl.append('(?:' + '|'.join(variants) + ')' + contrs[pi])
                    # console.log(reTpl);
                    reTpl.append(''.join(resl))
        else:
            # TODO: good!
            intraInsTpl = lazyRepeat(intraChars, intraIns)

            # capture at char level
            if capt == 2 and intraIns > 0:
                # sadly, we also have to capture the inter-term junk via parenth-wrapping .*?
				# to accum other capture groups' indices for \b boosting during scoring
                intraInsTpl = ')(' + intraInsTpl + ')('

            # https://github.com/leeoniya/uFuzzy/blob/1.0.14/src/uFuzzy.js#L329
            reTpl = []
            for pi, p in enumerate(parts):
                if not p:
                    continue # ?
                if p[0] == '"':
                    reTpl.append(escapeRegExp(p[1:-1]))
                else:
                    # p.split('').map((c, i, chars) => ...
                    resl = []
                    chars = p 
                    for i, c in enumerate(p):
                        # neg lookahead to prefer matching 'Test' instead of 'tTest' in ManifestTest or fittest
                        # but skip when search term contains leading repetition (aardvark, aaa)
                        if intraIns == 1 and i == 0 and len(chars) > 1 and c != chars[i+1]:
                            # c += '(?!' + c + ')'
                            resl.append(c + '(?!' + c + ')')
                        else:
                            resl.append(c)
                    reTpl.append(intraInsTpl.join(resl) + contrs[pi])
        
        # console.log(reTpl)
        # print(reTpl)

        # this only helps to reduce initial matches early when they can be detected
        # TODO: might want a mode 3 that excludes _
        preTpl = mode2Tpl if interLft == 2 else ''
        sufTpl = mode2Tpl if interRgt == 2 else ''

        interCharsTpl: PartialRegExp = sufTpl + lazyRepeat(self.opts.interChars, self.opts.interIns) + preTpl

        # capture at word level
        if capt > 0:
            if interOr:
                # this is basically for doing .matchAll() occurence counting and highlighting without needing permuted ooo needles
                reTplStr = preTpl + '(' + f"){sufTpl}|{preTpl}(".join(reTpl) + ')' + sufTpl
            else:
                # sadly, we also have to capture the inter-term junk via parenth-wrapping .*?
                # to accum other capture groups' indices for \b boosting during scoring
                reTplStr = '(' + f")({interCharsTpl})(".join(reTpl) + ')'
                reTplStr = f"(.??{preTpl}){reTplStr}({sufTpl}.*)" # nit: trailing capture here assumes interIns = None/Inf
        else:
            reTplStr = preTpl + interCharsTpl.join(reTpl) + sufTpl

        return re.compile(reTplStr, iuFlag), parts, contrs
    
    # https://github.com/leeoniya/uFuzzy/blob/1.0.14/src/uFuzzy.js#L371
    def filter(self, haystack: list[str], needle: str, idxs: HaystackIdxs | None) -> list[int]:
        if DEBUG:
            start_time = time.time()
        
        query, _, _ = self.prepQuery(needle)
        if query is None:
            return None

        out: list[int] = []
        if idxs:
            for idx in idxs:
                if 0 <= idx < len(haystack) and query.search(haystack[idx]):
                    out.append(idx)
        else:
            for i, h in enumerate(haystack):
                if query.search(h):
                    out.append(i)
        if DEBUG:
            print(f"filter time: {time.time() - start_time}")
        return out
    
    # https://github.com/leeoniya/uFuzzy/blob/1.0.14/src/uFuzzy.js#L402
    def info(self, idxs: HaystackIdxs, haystack: list[str], needle: str) -> Info:
        if DEBUG:
            start_time = time.time()

        query, parts, contrs = self.prepQuery(needle, 1)
        queryR, _, _ = self.prepQuery(needle, 2)
        partsLen = len(parts)
        _len = len(idxs)

        field = [0] * _len # Array(len).fill(0)
        info = Info(
            # idx in haystack
            idx = [0] * _len,

            # start of match
            start = field.copy(),
            # length of match
            # span = field.copy(),

            # contiguous chars matched
            chars = field.copy(),
            # contiguous (no fuzz) and bounded terms (intra=0, lft2/1, rgt2/1)
			# excludes terms that are contiguous but have < 2 bounds (substrings)
            terms = field.copy(),
            # cumulative length of unmatched chars (fuzz) within span
            interIns = field.copy(), # between terms
            intraIns = field.copy(), # within terms

            # interLft/interRgt counters
            interLft2 = field.copy(),
            interRgt2 = field.copy(),
            interLft1 = field.copy(),
            interRgt1 = field.copy(),

            ranges = [[]] * _len
        )

        # might discard idxs based on bounds checks
        mayDiscard = self.interLft == 1 or self.interRgt == 1

        uFlag = re.UNICODE if self.unicode else 0
        iuFlag = uFlag | re.IGNORECASE

        ii = 0
        for i in range(len(idxs)):
            mhstr = haystack[idxs[i]]

            # the matched parts are [full, junk, term, junk, term, junk]
            m: re.Match = query.search(mhstr)
            if not m: continue

            # leading junk
            start = m.start() + len(m.group(1))

            idxAcc = start
            # span = len(m.group(0))

            disc = False
            lft2, lft1 = 0, 0
            rgt2, rgt1 = 0, 0
            chars, terms, inter, intra = 0, 0, 0, 0
            refine = []
            k = 0
            for j in range(0, partsLen):
                k += 2
                group = (m.group(k) or '').lower()
                part = parts[j]
                term = part[1:-1] if part[0] == '"' else part + contrs[j]
                termLen = len(term)
                groupLen = len(group)
                fullMatch = group == term

                # this won't handle the case when an exact match exists across the boundary of the current group and the next junk
                # e.g. blob,ob when searching for 'bob' but finding the earlier `blob` (with extra insertion)
                if not fullMatch and len(m.group(k+1)) >= termLen:
                    idxOf = (m.group(k+1) or '').lower().find(term)
                    if idxOf > -1:
                        refine.extend([idxAcc, groupLen, idxOf, termLen])
                        m, incr = refineMatch(m, k, idxOf, termLen)
                        idxAcc += incr
                        group = term
                        groupLen = termLen
                        fullMatch = True
                        if j == 0:
                            start = idxAcc
                
                # https://github.com/leeoniya/uFuzzy/blob/1.0.14/src/uFuzzy.js#L497
                if mayDiscard or fullMatch:
                    # does group's left and/or right land on \b
                    lftCharIdx = idxAcc - 1
                    rgtCharIdx = idxAcc + groupLen
                    isPre = False
                    isSuf = False
                    # prefix info
                    if lftCharIdx == -1 or self.interBound.search(mhstr[lftCharIdx]):
                        if fullMatch:
                            lft2 += 1
                        isPre = True
                    elif self.interLft == 2:
                        disc = True
                        break
                    elif self.withIntraBound and self.intraBound.search(mhstr[lftCharIdx] + mhstr[lftCharIdx + 1]):
                        if fullMatch:
                            lft1 += 1
                        isPre = True
                    else:
                        if self.interLft == 1:
                            # regexps are eager, so try to improve the match by probing forward inter junk for exact match at a boundary
                            junk = m.group(k+1) or ''
                            junkIdx = idxAcc + groupLen
                            if len(junk) >= termLen:
                                idxOf = 0
                                found = False
                                _re = re.compile(term, iuFlag)
                                
                                for m2 in _re.finditer(junk): # while (m2 = re.exec(junk)) {
                                    if not m2:
                                        continue

                                    idxOf = m2.start()
                                    charIdx = junkIdx + idxOf
                                    lftCharIdx = charIdx - 1

                                    if lftCharIdx == -1 or self.interBound.search(mhstr[lftCharIdx]):
                                        lft2 += 1
                                        found = True
                                        break
                                    elif self.intraBound.search(mhstr[lftCharIdx] + mhstr[charIdx]):
                                        lft1 += 1
                                        found = True
                                        break
                                
                                if found:
                                    isPre = True

                                    # identical to exact term refinement pass above
                                    refine.extend([idxAcc, groupLen, idxOf, termLen])
                                    m, incr = refineMatch(m, k, idxOf, termLen)
                                    idxAcc += incr
                                    group = term
                                    groupLen = termLen
                                    fullMatch = True

                                    if j == 0:
                                        start = idxAcc
                            
                            if not isPre:
                                disc = True
                                break

                    # suffix info
                    if rgtCharIdx == len(mhstr) or self.interBound.search(mhstr[rgtCharIdx]):
                        if fullMatch:
                            rgt2 += 1
                        isSuf = True
                    else:
                        if self.interRgt == BoundMode.Strict: # == 2
                            disc = True
                            break
                        if self.withIntraBound and self.intraBound.search(mhstr[rgtCharIdx - 1] + mhstr[rgtCharIdx]):
                            if fullMatch:
                                rgt1 += 1
                            isSuf = True
                        else:
                            if self.interRgt == BoundMode.Loose: # == 1
                                disc = True
                                break
                    if fullMatch:
                        chars += termLen

                        if isPre and isSuf:
                            terms += 1
                
                if groupLen > termLen:
                    intra += groupLen - termLen # intraFuzz
                if j > 0:
                    inter += len(m.group(k-1)) # interFuzz

                # TODO: group here is lowercased, which is okay for length cmp, but not more case-sensitive filts
                if not self.opts.intraFilt(term, group, idxAcc):
                    disc = True
                    break

                if j < partsLen - 1:
                    idxAcc += groupLen + len(m.group(k+1))
            
            # https://github.com/leeoniya/uFuzzy/blob/1.0.14/src/uFuzzy.js#L620
            if not disc:
                m = queryR.search(mhstr)
                assert m
                if not m: continue

                info.idx[ii]       = idxs[i]
                info.interLft2[ii] = lft2
                info.interLft1[ii] = lft1
                info.interRgt2[ii] = rgt2
                info.interRgt1[ii] = rgt1
                info.chars[ii]     = chars
                info.terms[ii]     = terms
                info.interIns[ii]  = inter
                info.intraIns[ii]  = intra

                info.start[ii] = start
                # info.span[ii] = span

                # ranges
                idxAcc = m.start() + len(m.group(1))

                refLen = len(refine)
                ri = 0 if refLen > 0 else -1
                lastRi = refLen - 4

                for i in range(2, len(m.groups())):
                    _len = len(m.group(i))

                    if ri != -1 and ri <= lastRi and refine[ri] == idxAcc:
                        groupLen = refine[ri+1]
                        idxOf    = refine[ri+2]
                        termLen  = refine[ri+3]

                        # advance to end of original (full) group match that includes intra-junk
                        j = i
                        vl: list[str] = []
                        ll = 0
                        while ll < groupLen:
                            vl.append(m.group(j)) # v += m[j];
                            ll += len(m.group(j))
                            j += 1
                        v = ''.join(vl)
                        
                        # hit via 'superC'
                        # m.splice(i, j - i, v)
                        if m.group(i) != v:
                            m_groups = list(m.groups())
                            m_groups[i] = v
                            m = MatchLike(m, m_groups)

                        m, incr = refineMatch(m, i, idxOf, termLen)
                        idxAcc += incr

                        ri += 4
                    else:
                        idxAcc += _len
                        i += 1
                
                idxAcc = m.start() + len(m.group(1))

                ranges = info.ranges[ii] = []
                _from = idxAcc
                _to = idxAcc

                for i in range(2, len(m.groups())):
                    _len = len(m.group(i))

                    idxAcc += _len

                    if i % 2 == 0:
                        _to = idxAcc
                    elif _len > 0:
                        ranges.append(_from)
                        ranges.append(_to)
                        _from = _to = idxAcc
                
                if _to > _from:
                    ranges.append(_from)
                    ranges.append(_to)

                ii += 1
        
        # trim arrays
        if ii < len(idxs):
            for k in info.__slots__:
                setattr(info, k, getattr(info, k)[:ii])
		
        if DEBUG:
            print(f"info time: {time.time() - start_time}")
        return info
    

    # https://github.com/leeoniya/uFuzzy/blob/1.0.14/src/uFuzzy.js#L720
    # returns [idxs, info, order]
    def search(
            self,
            haystack: list[str],
            needle: str,
            outOfOrder: int | bool = False,
            infoThresh: int = 1e3,
            preFiltered: HaystackIdxs | None = None,
    ) -> SearchResult:
        OOO_TERMS_LIMIT = 5
        outOfOrder = 0 if not outOfOrder else (OOO_TERMS_LIMIT if outOfOrder == True else outOfOrder)

        needles = None
        matches = None
        negs = []

        def repl(m: re.Match) -> str:
            neg = (m.group(0) or '').strip()[1:]
            if neg[0] == '"':
                neg = escapeRegExp(neg[1:-1])
            negs.append(neg)
            return ''
        needle = re.sub(self.NEGS_RE, repl, needle)

        terms = self.split(needle)

        uFlag = re.UNICODE if self.unicode else 0
        iuFlag = uFlag | re.IGNORECASE
        if negs:
            negsRe = re.compile('|'.join(negs), iuFlag)

            if not terms:
                idxs = [i for i, item in enumerate(haystack) if not negsRe.search(item)]
                return [idxs, None, None]
        else:
            # abort search (needle is empty after pre-processing, e.g. no alpha-numeric chars)
            if not terms:
                return [None, None, None]
        # print(f"{negs=}");
        # print(f"{needle=}");
            
        # TODO: haven't checked below here
        if outOfOrder > 0:
			# since uFuzzy is an AND-based search, we can iteratively pre-reduce the haystack by searching
			# for each term in isolation before running permutations on what's left.
			# this is a major perf win. e.g. searching "test man ger pp a" goes from 570ms -> 14ms
            terms = self.split(needle)

            if len(terms) > 1:
                # longest -> shortest
                terms2 = sorted(terms, key=len, reverse=True)

                for ti in terms2:
                    # no haystack item contained all terms
                    if preFiltered is not None and len(preFiltered) == 0:
                        return [[], None, None]

                    preFiltered = self.filter(haystack, ti, preFiltered)

                # avoid combinatorial explosion by limiting outOfOrder to 5 terms (120 max searches)
				# fall back to just filter() otherwise
                if len(terms) > outOfOrder:
                    return [preFiltered, None, None]

                needles = [' '.join(perm) for perm in permute(terms)]

                # filtered matches for each needle excluding same matches for prior needles
                matches = []
                # keeps track of already-matched idxs to skip in follow-up permutations
                matchedIdxs = set()

                lenPreFiltered = len(preFiltered)
                for n in needles:
                    if len(matchedIdxs) < lenPreFiltered:
                        # filter further for this needle, exclude already-matched
                        preFiltered2 = [idx for idx in preFiltered if idx not in matchedIdxs]
                        matched = self.filter(haystack, n, preFiltered2)

                        matchedIdxs.update(matched)
                        matches.append(matched)
                    else:
                        matches.append([])
        
        # interOR
	    # print(f"{prepQuery(needle, 1, None, True)}")

		# non-ooo or ooo w/single term
        if needles is None:
            needles = [needle]
            matches = [preFiltered if preFiltered else self.filter(haystack, needle, None)]

        retInfo = None
        retOrder = None

        if negs:
            # TODO: not sure about this translation
            # matches = matches.map(idxs => idxs.filter(idx => !negsRe.test(haystack[idx])));
            matches = list(filter(lambda idx: idx is not None and not negsRe.search(haystack[idx]), idxs) for idxs in matches)

        matchCount = sum(len(idxs) for idxs in matches)

        # https://github.com/leeoniya/uFuzzy/blob/1.0.14/src/uFuzzy.js#L832
        # rank, sort, concat
        if matchCount <= infoThresh:
            retInfo = Info()
            retOrder = []

            for ni, idxs in enumerate(matches):
                if not idxs:
                    continue

                _needle = needles[ni]
                _info = self.info(idxs, haystack, _needle)
                order = self.opts.sort(_info, haystack, _needle)

                # offset idxs for concat'ing infos
                if ni > 0:
                    order = [idx + len(retOrder) for idx in order]

                # for (let k in _info)
                #   retInfo[k] = (retInfo[k] ?? []).concat(_info[k]);
                for k in getattr(_info, "__slots__", getattr(_info, "__dict__", [])):
                    final_list: list = getattr(retInfo, k) or []
                    current_list = getattr(_info, k)
                    final_list.extend(current_list)
                    setattr(retInfo, k, final_list)

                retOrder.extend(order)
        
        return (
            [idx for idxs in matches for idx in idxs], # [].concat(...matches)
            retInfo,
            retOrder,
        )
    
    def search_simple(
            self,
            haystack: list[str],
            needle: str,
            outOfOrder: int | bool = False,
            infoThresh: int = 1e3,
            preFiltered: HaystackIdxs | None = None,
    ) -> list[str]:
        idxs, info, order = self.search(haystack, needle, outOfOrder, infoThresh, preFiltered)
        if order:
            return [haystack[info.idx[oi]] for oi in order]
        else:
            return []
    
# https://github.com/leeoniya/uFuzzy/blob/1.0.14/src/uFuzzy.js#L708
def refineMatch(m: re.Match | MatchLike, k: int, idxInNext: int, termLen: int):
    if k == 0:
        g = [m.group(0)] # in js, groups includes group[0]
        g.extend(m.groups())
    else:
        g = list(m.groups())
        k -= 1
    # shift the current group into the prior junk
    prepend = g[k] + g[k+1][:idxInNext]
    g[k-1] += prepend
    g[k]    = g[k+1][idxInNext:idxInNext + termLen]
    g[k+1]  = g[k+1][idxInNext + termLen:]
    m2 = MatchLike(m, g)
    return m2, len(prepend)
