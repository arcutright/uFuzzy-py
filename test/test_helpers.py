from __future__ import annotations
import os
import functools
import json
import time
import timeit
import unittest
from unittest.mock import create_autospec, MagicMock

from src.uFuzzy import cmp

class TestHelpers(unittest.TestCase):
    # def test_latinizers_speed(self):
    #     test_strs = [
    #         "a",
    #         "my longer string but no diacritics",
    #         "my łõñgër śtrîng wíth dįáçrïtįćs",
    #         "all diacritics: ÁÀÃÂÄĄáàãâäąÉÈÊËĖéèêëęÍÌÎÏĮíìîïįÓÒÔÕÖóòôõöÚÙÛÜŪŲúùûüūųÇČĆçčćŁłÑŃñńŠŚšśŻŹżź"
    #     ]
    #     test_strs += [test_strs[1] * 10]
    #     test_strs += [test_strs[2] * 10]
    #     test_strs += [test_strs[3] * 10]
    #     for s in test_strs:
    #         # setup="from src import uFuzzy"
    #         setup="from src import helpers"
    #         print(f"test '{s}'")
    #         print("  latinize  :", timeit.timeit(f'helpers._latinize("{s}")', number=1000, setup=setup))
    #     x = 1
    
    def test_cmp(self):
        # tests based on js behavior of `let cmp = new Intl.Collator('en', { numeric: true, sensitivity: 'base' }).compare;`
        # basic alpha compares
        self.assertEqual(cmp("a", "a"), 0)
        self.assertLess(cmp("a", "b"), 0) # -1
        self.assertGreater(cmp("c", "b"), 0) # +1
        self.assertGreater(cmp(" ", ""), 0) # +1
        self.assertGreater(cmp("_", " "), 0) # +1
        self.assertEqual(cmp("hello", "hello"), 0)
        self.assertLess(cmp("hello", "hello "), 0) # -1
        self.assertGreater(cmp("hello", " hello"), 0) # +1
        # case insensitive
        self.assertEqual(cmp("a", "A"), 0)
        # numeric compares
        self.assertEqual(cmp("1", "1"), 0)
        self.assertEqual(cmp("10", "10"), 0)
        self.assertLess(cmp("1", "2"), 0) # -1
        self.assertGreater(cmp("10", "2"), 0) # +1
        self.assertLess(cmp("10", "20"), 0) # -1
        # numeric + leading zeros
        self.assertEqual(cmp("0010", "10"), 0)
        self.assertEqual(cmp("0", "000000"), 0)
        self.assertEqual(cmp("0000", "000000"), 0)
        # numeric unsupported by js
        self.assertGreater(cmp("-5", "-3"), 0) # +1
        self.assertGreater(cmp("1.0", "1"), 0) # +1
        self.assertLess(cmp("1e3", "1000"), 0) # -1
        # alpha vs numeric compares
        self.assertGreater(cmp("a", "4"), 0) # +1
        # alpha + numeric compares
        self.assertGreater(cmp("10a", "10"), 0) # +1
        self.assertLess(cmp("10a", "10b"), 0) # -1
        self.assertLess(cmp("10a", "20a"), 0) # -1
        self.assertLess(cmp("a10", "a20"), 0) # -1
        self.assertLess(cmp("a10", "a20"), 0) # -1
        self.assertLess(cmp("a10", "b10"), 0) # -1
        self.assertGreater(cmp("c10", "b10"), 0) # +1
        # diacritics
        self.assertEqual(cmp("o", "ó"), 0)
        self.assertEqual(cmp("o", "ô"), 0)
        self.assertEqual(cmp("o", "ö"), 0)
        self.assertEqual(cmp("n", "ñ"), 0)
        self.assertEqual(cmp("ł", "l"), 0) # simple unicode normalization fails on this
        self.assertEqual(cmp("Ł", "L"), 0)
        self.assertGreater(cmp("o", "ñ"), 0) # +1

