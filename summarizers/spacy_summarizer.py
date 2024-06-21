import spacy
from heapq import nlargest
from collections import Counter
from string import punctuation

nlp = spacy.load('en_core_web_sm')

def spacy_summarizer(text):
    doc = nlp(text)
    keyword = []
    pos_tag = ['PROPN', 'ADJ', 'NOUN', 'VERB']

    for token in doc:
        if token.text in nlp.Defaults.stop_words or token.text in punctuation:
            continue
        if token.pos_ in pos_tag:
            keyword.append(token.lemma_)

    # Frequency Analysis (Bag of Words)
    freq_word = Counter(keyword)
    max_freq = Counter(keyword).most_common(1)[0][1]

    for word in freq_word.keys():
        freq_word[word] = (freq_word[word] / max_freq)

    # Sentence Scoring
    sent_strength = {}
    for sent in doc.sents:
        for word in sent:
            if word.lemma_ in freq_word.keys():
                if sent in sent_strength:
                    sent_strength[sent] += freq_word[word.lemma_]
                else:
                    sent_strength[sent] = freq_word[word.lemma_]

    # Summarization
    summarized_sentences = nlargest(3, sent_strength, key=sent_strength.get)
    final_sentences = [w.text for w in summarized_sentences]
    summary = ' '.join(final_sentences)
    return summary
