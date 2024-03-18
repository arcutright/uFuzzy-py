from __future__ import annotations
import os
import functools
import json
import time
import timeit
import unittest
from unittest.mock import create_autospec, MagicMock

from src import uFuzzy

@functools.lru_cache
def get_data() -> dict:
    cwd = os.path.abspath(os.path.dirname(__file__))
    fpath = os.path.join(cwd, 'testdata.json')
    with open(fpath, encoding='utf-8') as file:
        json_data = json.load(file)
    return json_data

class TestUFuzzy(unittest.TestCase):
    def assertStrEq(self, expected_str, actual_str, msg=None):
        expected_str = str(expected_str).strip().lower()
        actual_str = str(actual_str).strip().lower()
        self.assertEqual(expected_str, actual_str, msg)
        
    def test_search(self):
        data = get_data()
        haystack = data.get('steam_games_47000', [])
        options = uFuzzy.Options()
        uf = uFuzzy.uFuzzy(options)

        needle = "super ma"
        idxs, info, order = uf.search(haystack, needle)
        self.assertIsNotNone(info)
        self.assertIsNotNone(order)
        results = [haystack[info.idx[oi]] for oi in order]
        self.assertEqual(50, len(results))
        expected_results = [
            "Super Man Or Monster",
            "Super Markup Man",
            "The Adventures of Super Mario Bros. 3",
            "The Adventures of Super Mario Bros. 3: Crimes R Us / Life's Ruff",
            (9, "The Adventures of Super Mario Bros. 3: Princess Toadstool For President / Never Koop A Koopa"),
            (16, "Super Cane Magic ZERO"),
            (37, "Supermagical"),
            (38, "Supermagical - Soundtrack"),
        ]
        for i, expected in enumerate(expected_results):
            if isinstance(expected, tuple):
                ri, expected = expected
            else:
                ri = i
            if expected is None or expected == '*':
                continue
            self.assertStrEq(expected, results[ri], f"bad result at row {i}, expected \"{expected}\"")
 
    def test_search_singleerror(self):
        data = get_data()
        haystack = data.get('steam_games_47000', [])
        options = uFuzzy.Options()
        options.intraMode = uFuzzy.IntraMode.SingleError
        uf = uFuzzy.uFuzzy(options)

        needle = "superC"
        idxs, info, order = uf.search(haystack, needle)
        self.assertIsNotNone(info)
        self.assertIsNotNone(order)
        results = [haystack[info.idx[oi]] for oi in order]
        self.assertEqual(417, len(results))
        expected_results = [
            "Supercharged Robot VULKAISER",
            "Supercharged Robot VULKAISER Demo",
            "SuperCluster: Void",
            "Supercop",
            (82, "Super Cuber"),
            (233, "Super Toy Cars"),
            (405, "Rocksmith 2014 - Jane's Addiction - Superhero"),
        ]
        for i, expected in enumerate(expected_results):
            if isinstance(expected, tuple):
                ri, expected = expected
            else:
                ri = i
            if expected is None or expected == '*':
                continue
            self.assertStrEq(expected, results[ri], f"bad result at row {i}, expected \"{expected}\"")
