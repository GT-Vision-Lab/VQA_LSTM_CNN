"""Prepare a spaCy vocabulary file that performs nearest-neighbor paraphrasing"""
from __future__ import unicode_literals, print_function
import io
import os
import argparse
from collections import Counter
import json
import numpy
import random

import spacy.en
import sputnik.util
import sense2vec.vectors


def build_vocab(tokenizer, questions, count_thr):
    # count up the number of words
    counts = Counter()
    for string in questions:
        counts.update(w.text for w in tokenizer(string) if not w.is_space)

    cw = sorted([(count,w) for w,count in counts.iteritems()], reverse=True)
    print('top words and their counts:')
    print('\n'.join(map(str,cw[:20])))

    # print some stats
    total_words = sum(counts.itervalues())
    print('total words:', total_words)
    bad_words = [w for w,n in counts.iteritems() if n <= count_thr]
    vocab = [w for w,n in counts.iteritems() if n > count_thr]
    bad_count = sum(counts[w] for w in bad_words)
    print('number of bad words: %d/%d = %.2f%%' % (len(bad_words), len(counts), len(bad_words)*100.0/len(counts)))
    print('number of words in vocab would be %d' % (len(vocab), ))
    print('number of UNKs: %d/%d = %.2f%%' % (bad_count, total_words, bad_count*100.0/total_words))
    return vocab


def main(params):
    input_train_json = json.load(open(params['input_train_json'], 'r'))
    print("Load spaCy with GloVe vectors")
    nlp = spacy.load('en', vectors='en_glove_cc_300_1m_vectors')
    words_to_keep = build_vocab(
                        nlp.tokenizer,
                        [img['question'] for img in input_train_json],
                        params['word_count_threshold'])
    vectors = sense2vec.vectors.VectorMap(nlp.vocab.vectors_length)
    for string in words_to_keep:
        word = nlp.vocab[string]
        vectors.borrow(word.orth_, 1, numpy.ascontiguousarray(word.vector))
    replaced = 0
    paraphrases = []
    for i, word in enumerate(nlp.vocab):
        if word.orth_ in words_to_keep:
            word.norm_ = word.orth_
        elif word.lower_ in words_to_keep:
            word.norm_ = word.lower_
        elif word.is_alpha and word.has_vector:
            vector = numpy.ascontiguousarray(word.vector, dtype='float32')
            synonyms, scores = vectors.most_similar(vector, 1)
            word.norm_ = synonyms[0]
            paraphrases.append((word.orth_, word.norm_))
        else:
            word.norm_ = word.shape_
        if i and i % 10000 == 0:
            print(i, 'words processed. Example: %s --> %s' % random.choice(paraphrases))
    print('%d vector-based paraphrases' % len(paraphrases))
    if not os.path.exists(params['spacy_data']):
        os.mkdir(params['spacy_data'])
    if not os.path.exists(os.path.join(params['spacy_data'], 'vocab')):
        os.mkdir(os.path.join(params['spacy_data'], 'vocab'))
    if not os.path.exists(os.path.join(params['spacy_data'], 'tokenizer')):
        os.mkdir(os.path.join(params['spacy_data'], 'tokenizer'))

    nlp.vocab.dump(os.path.join(params['spacy_data'], 'vocab', 'lexemes.bin'))
    with io.open(os.path.join(params['spacy_data'], 'vocab', 'strings.json'), 'w',
            encoding='utf8') as file_:
        nlp.vocab.strings.dump(file_)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_train_json', required=True, help='input json file to build vocab from')
    parser.add_argument('--word_count_threshold', default=5, help='minimum frequency threshold')
    parser.add_argument('--spacy_data', default='spacy_data', help='location of spaCy data')
    args = parser.parse_args()
    params = vars(args) # convert to ordinary dict
    print('parsed input parameters:')
    print(json.dumps(params, indent = 2))
    main(params)
