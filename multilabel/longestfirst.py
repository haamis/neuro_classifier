#!/usr/bin/env python3

import sys
import re

from collections import defaultdict, namedtuple
from itertools import groupby

from ahocorasick import Automaton


Span = namedtuple('Span', ['start', 'end', 'max_len', 'tokenized'])

# Regex filtering BERT special tokens
FILTER_RE = re.compile(r'^\[(PAD|UNK|CLS|SEP|MASK|unused[0-9]+)\]')


class NoMatch(Exception):
    pass


def argparser():
    from argparse import ArgumentParser
    ap = ArgumentParser()
    ap.add_argument('vocab')
    ap.add_argument('text')
    return ap


class Tokenizer(object):
    def __init__(self, vocab):
        self.vocab = vocab
        if any(v.startswith('^^') for v in vocab):
            raise ValueError('"^^" in vocab')   # this would break things
        self.automata_by_len = self.build_automata(vocab)
        self.max_len = max(self.automata_by_len.keys())

    def _pick_match(self, matches):
        # Return (start, end) span of first permissible match from
        # ahocorasick Automaton.iter() results
        for match in matches:
            match_end, match_len = match
            match_end += 1    # ahocorasick end is inclusive
            match_start = match_end - match_len
            if match_start in (1, 2) or match_end < 3:
                # Text start "^^" marker must be matched as a part of
                # a longer word piece (e.g. "^^M")
                continue
            return match_start, match_end
        raise NoMatch()

    def _longest_match(self, text, max_len, start, end):
        for length in reversed(range(1, max_len+1)):
            matches = self.automata_by_len[length].iter(text, start, end)
            try:
                match_start, match_end = self._pick_match(matches)
            except NoMatch:
                continue    # nothing matched at this length
            return match_start, match_end, length
        raise NoMatch()    # nothing matched at all

    def _tokenize_iterative(self, text, max_len, start, end):
        if start == end:
            return []
        # Maintain a list of untokenized and tokenized text spans and
        # tokenize the first untokenized span until all are tokenized.
        make_span = lambda s, e, m, t: [] if s == e else [Span(s, e, m, t)]
        spans = make_span(start, end, max_len, False)
        first_untokenized = 0
        while True:
            for i in range(first_untokenized, len(spans)):
                if not spans[i].tokenized:
                    break
            else:
                break    # done, all tokenized
            span = spans[i]
            try:
                m_start, m_end, length = self._longest_match(
                    text, span.max_len, span.start, span.end)
            except NoMatch:
                raise ValueError('failed to tokenize: {}'.format(
                    text[span.start:span.end]))
            # length-1 for first part b/c match is leftmost of longest
            before = make_span(span.start, m_start, length-1, False)
            match = make_span(m_start, m_end, None, True)
            after = make_span(m_end, span.end, length, False)
            spans[i:i+1] = before + match + after
            first_untokenized = i
        return [text[s.start:s.end] for s in spans]
    
    def _tokenize_recursive(self, text, max_len, start, end):
        if start == end:
            return []
        try:
            match_start, match_end, length = self._longest_match(
                text, max_len, start, end)
            # length-1 for first part b/c match is leftmost of longest
            return (self._tokenize(text, length-1, start, match_start) +
                    [text[match_start:match_end]] +
                    self._tokenize(text, length, match_end, end))
        except NoMatch:
            raise ValueError('failed to tokenize: {}'.format(text[start:end]))

    def tokenize(self, text):
        text = '^^'+text    # see build_automata()
        # tokens = self._tokenize_recursive(text, self.max_len, 0, len(text))
        tokens = self._tokenize_iterative(text, self.max_len, 0, len(text))
        equal = ''.join(tokens) == text, 'internal error'
        # Map to wordpiece continuation convention
        wptokens = [t[2:] if t.startswith('^^') else '##'+t for t in tokens]
        return wptokens
        
    @classmethod
    def load(cls, vocab_path):
        vocab = cls.load_vocab(vocab_path)
        return cls(vocab)

    @staticmethod
    def build_automata(vocab):
        # Build Aho-Corasick matching automata for vocabulary items
        # grouped by length. The wordpiece convention is inverted for
        # matching: continuations are unmarked (instead of "##") and
        # string start is marked by "^^".
        strings = [v[2:] if v.startswith('##') else '^^'+v for v in vocab]
        max_len = max(len(s) for s in strings)
        strings.sort(key=lambda s: len(s))
        strings_by_len = defaultdict(list)
        for k, g in groupby(strings, lambda s: len(s)):
            strings_by_len[k] = list(g)
        automata_by_len = {}
        for i in range(1, max_len+1):
            a = Automaton()
            for s in strings_by_len[i]:
                a.add_word(s, i)
            a.make_automaton()
            automata_by_len[i] = a
        return automata_by_len

    @staticmethod
    def load_vocab(path):
        vocab = set()
        with open(path) as f:
            for l in f:
                l = l.rstrip('\n')
                if FILTER_RE.match(l):
                    continue
                if l.isspace() or not l:
                    continue
                vocab.add(l)
        return vocab


def main(argv):
    args = argparser().parse_args(argv[1:])
    tokenizer = Tokenizer.load(args.vocab)
    with open(args.text) as f:
        for l in f:
            text = l.rstrip()
            try:
                tokens = tokenizer.tokenize(text)
            except:
                print('failed: {}'.format(text), file=sys.stderr)
                raise
            print(' '.join(tokens))
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
