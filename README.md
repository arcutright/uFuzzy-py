## ▒ μFuzzy

A tiny, efficient, fuzzy search that doesn't suck

---
### Introduction

This is my fuzzy 🐈. [There are many like it](#similar-work), but this one is mine.

uFuzzy is a fuzzy search lib designed to match a relatively short search string (needle) against a large list of short-to-medium strings (haystack).
It might be best described as a more forgiving [String.prototype.indexOf()](https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/String/indexOf).

Common use cases are list filtering, auto-complete/suggest, title/name or description search, filenames or functions.
uFuzzy is case insensitive and expects all alpha-numeric characters in the needle to occur in the same order, with an adjustable tolerance for additionally inserted characters;
`horse cart` will not match `cart horse`, but would be matched for either `cart` or `horse`.
This disqualifies uFuzzy from being used as a spellcheck (due to omitted letters), or a fulltext/document search, where terms can be out of order.
However, it's easy to perform a separate uFuzzy search for each permutation of terms in the needle to achieve out-of-order matching without appreciable degredation in performance for most cases.

Now that you know what uFuzzy _isn't_, let's see what it can offer over existing solutions.

---
### Demo

Below are some comparisons using a diverse 162,000 string/phrase dataset derived mostly from concatenating the lists from [fuzzysort's demo](https://rawgit.com/farzher/fuzzysort/master/test.html), plus 9,000 metrics objects split into three lists (`metric_type_9000`, `metric_name_9000`, `metric_help_9000`):

```
[
  {
    "type": "gauge",
    "name": "coordinator_prometheus_engine_prometheus_engine_queries",
    "help": "The current number of queries being executed or waiting.",
  },
  ...
]
```

First, here is uFuzzy in isolation to demonstrate its performance.

**NOTE:** [testdata.json](https://github.com/leeoniya/uFuzzy/blob/main/demos/testdata.json) is 4MB, so first load may be slow due to network transfer. Try refreshing once it's been cached by your browser.

https://leeoniya.github.io/uFuzzy/demos/compare.html?libs=uFuzzy&search=super%20ma

Now the same comparison page, booted with [fuzzysort](https://github.com/farzher/fuzzysort), [QuickScore](https://fwextensions.github.io/quick-score-demo/), and [Fuse.js](https://fusejs.io/):

https://leeoniya.github.io/uFuzzy/demos/compare.html?libs=uFuzzy,fuzzysort,QuickScore,Fuse&search=super%20ma

Here is the full library list but with a reduced dataset (just `hearthstone_750`, `urls_and_titles_600`) to avoid crashing your browser:

https://leeoniya.github.io/uFuzzy/demos/compare.html?lists=hearthstone_750,urls_and_titles_600&search=moo

---
### Similar work

<!--
https://bestofjs.org/projects?tags=search
-->

<table>
    <thead>
        <tr>
            <th>Lib</th>
            <th>Stars</th>
            <th>Size (min)</th>
            <th>Init</th>
            <th>Search</th>
            <th>Heap (peak)</th>
            <th>Retained</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>
                <a href="https://github.com/leeoniya/uFuzzy">uFuzzy</a>
                (<a href="https://leeoniya.github.io/uFuzzy/demos/compare.html?libs=uFuzzy&search=super%20ma">try</a>)
            </td>
            <td>★ 0</td>
            <td>2.5KB</td>
            <td>0.3ms</td>
            <td>11.7ms</td>
            <td>7.7MB</td>
            <td>7.5MB</td>
        </tr>
        <tr>
            <td>
                <a href="https://github.com/krisk/Fuse">Fuse.js</a>
                (<a href="https://leeoniya.github.io/uFuzzy/demos/compare.html?libs=Fuse&search=super%20ma">try</a>)
            </td>
            <td>★ 14.8k</td>
            <td>23.5KB</td>
            <td>34ms</td>
            <td>600ms</td>
            <td>20.6MB</td>
            <td>14.2MB</td>
        </tr>
        <tr>
            <td>
                <a href="https://github.com/nextapps-de/flexsearch">FlexSearch (Light)</a>
                (<a href="https://leeoniya.github.io/uFuzzy/demos/compare.html?libs=FlexSearch&search=super%20ma">try</a>)
            </td>
            <td>★ 8.9k</td>
            <td>5.9KB</td>
            <td>3500ms</td>
            <td>8.1ms</td>
            <td>322MB</td>
            <td>323MB</td>
        </tr>
        <tr>
            <td>
                <a href="https://github.com/olivernn/lunr.js">Lunr.js</a>
                (<a href="https://leeoniya.github.io/uFuzzy/demos/compare.html?libs=Lunr&search=super%20ma">try</a>)
            </td>
            <td>★ 8.2k</td>
        </tr>
        <tr>
            <td>
                <a href="https://github.com/LyraSearch/lyra">LyraSearch</a>
                (<a href="https://leeoniya.github.io/uFuzzy/demos/compare.html?libs=LyraSearch&search=super%20ma">try</a>)
            </td>
            <td>★ 3.3k</td>
        </tr>
        <tr>
            <td>
                <a href="https://github.com/kentcdodds/match-sorter">match-sorter</a>
                (<a href="https://leeoniya.github.io/uFuzzy/demos/compare.html?libs=match-sorter&search=super%20ma">try</a>)
            </td>
            <td>★ 3.1k</td>
            <td>7.3KB</td>
            <td>0.03ms</td>
            <td>125ms</td>
            <td>13.1MB</td>
            <td>12.9MB</td>
        </tr>
        <tr>
            <td>
                <a href="https://github.com/farzher/fuzzysort">fuzzysort</a>
                (<a href="https://leeoniya.github.io/uFuzzy/demos/compare.html?libs=fuzzysort&search=super%20ma">try</a>)
            </td>
            <td>★ 3k</td>
            <td>5.5KB</td>
            <td>50ms</td>
            <td>12ms</td>
            <td>54.7MB</td>
            <td>54.4MB</td>
        </tr>
        <tr>
            <td>
                <a href="https://github.com/bevacqua/fuzzysearch">fuzzysearch</a>
                (<a href="https://leeoniya.github.io/uFuzzy/demos/compare.html?libs=fuzzysearch&search=super%20ma">try</a>)
            </td>
            <td>★ 2.6k</td>
        </tr>
        <tr>
            <td>
                <a href="https://github.com/weixsong/elasticlunr.js">Elasticlunr.js</a>
                (<a href="https://leeoniya.github.io/uFuzzy/demos/compare.html?libs=Elasticlunr&search=super%20ma">try</a>)
            </td>
            <td>★ 1.9k</td>
        </tr>
        <tr>
            <td>
                <a href="https://github.com/lucaong/minisearch">MiniSearch</a>
                (<a href="https://leeoniya.github.io/uFuzzy/demos/compare.html?libs=MiniSearch&search=super%20ma">try</a>)
            </td>
            <td>★ 1.5k</td>
            <td>22.4KB</td>
            <td>575ms</td>
            <td>0.7ms</td>
            <td>70.2MB</td>
            <td>70MB</td>
        </tr>
        <tr>
            <td>
                <a href="https://github.com/Glench/fuzzyset.js">Fuzzyset</a>
                (<a href="https://leeoniya.github.io/uFuzzy/demos/compare.html?libs=Fuzzyset&search=super%20ma">try</a>)
            </td>
            <td>★ 1.3k</td>
            <td>2.8KB</td>
            <td>2900ms</td>
            <td>31ms</td>
            <td>246MB</td>
            <td>244MB</td>
        </tr>
        <tr>
            <td>
                <a href="https://github.com/fergiemcdowall/search-index">search-index</a>
                (<a href="https://leeoniya.github.io/uFuzzy/demos/compare.html?libs=search-index&search=super%20ma">try</a>)
            </td>
            <td>★ 1.3k</td>
        </tr>
        <tr>
            <td>
                <a href="https://github.com/rmm5t/liquidmetal">LiquidMetal</a>
                (<a href="https://leeoniya.github.io/uFuzzy/demos/compare.html?libs=LiquidMetal&search=super%20ma">try</a>)
            </td>
            <td>★ 285</td>
        </tr>
        <tr>
            <td>
                <a href="https://github.com/itemsapi/itemsjs">ItemJS</a>
                (<a href="https://leeoniya.github.io/uFuzzy/demos/compare.html?libs=ItemJS&search=super%20ma">try</a>)
            </td>
            <td>★ 260</td>
        </tr>
        <tr>
            <td>
                <a href="https://github.com/wouter2203/fuzzy-search">FuzzySearch</a>
                (<a href="https://leeoniya.github.io/uFuzzy/demos/compare.html?libs=fuzzy-search&search=super%20ma">try</a>)
            </td>
            <td>★ 184</td>
        </tr>
        <tr>
            <td>
                <a href="https://github.com/jeancroy/FuzzySearch">FuzzySearch2</a>
                (<a href="https://leeoniya.github.io/uFuzzy/demos/compare.html?libs=FuzzySearch2&search=super%20ma">try</a>)
            </td>
            <td>★ 173</td>
        </tr>
        <tr>
            <td>
                <a href="https://github.com/fwextensions/quick-score">QuickScore</a>
                (<a href="https://leeoniya.github.io/uFuzzy/demos/compare.html?libs=QuickScore&search=super%20ma">try</a>)
            </td>
            <td>★ 131</td>
            <td>9.1KB</td>
            <td>26ms</td>
            <td>155ms</td>
            <td>26.1MB</td>
            <td>18.7MB</td>
        </tr>
        <tr>
            <td>
                <a href="https://github.com/jhawthorn/fzy.js/">fzy</a>
                (<a href="https://leeoniya.github.io/uFuzzy/demos/compare.html?libs=fzy&search=super%20ma">try</a>)
            </td>
            <td>★ 115</td>
        </tr>
        <tr>
            <td>
                <a href="https://github.com/grafana/grafana/blob/main/packages/grafana-ui/src/utils/fuzzy.ts">fuzzyMatch</a>
                (<a href="https://leeoniya.github.io/uFuzzy/demos/compare.html?libs=fuzzyMatch&search=super%20ma">try</a>)
            </td>
            <td>★ 0</td>
        </tr>
    </tbody>
</table>