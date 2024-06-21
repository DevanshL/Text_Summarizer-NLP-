import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from string import punctuation
from collections import Counter
from heapq import nlargest

nltk.download('punkt')
nltk.download('stopwords')


def nltk_summarizer(text):
    # Tokenization and Stopwords Removal
    stop_words = set(stopwords.words('english') + list(punctuation))
    words = word_tokenize(text.lower())
    filtered_words = [word for word in words if word not in stop_words]

    # Stemming
    ps = nltk.PorterStemmer()
    stemmed_words = [ps.stem(word) for word in filtered_words]

    # Frequency Analysis (Bag of Words)
    word_freq = Counter(stemmed_words)
    max_freq = word_freq.most_common(1)[0][1]

    for word in word_freq.keys():
        word_freq[word] = (word_freq[word] / max_freq)

    # Sentence Scoring
    sentences = sent_tokenize(text)
    sentence_scores = {}

    for sent in sentences:
        for word in word_tokenize(sent.lower()):
            if word in word_freq.keys():
                if sent not in sentence_scores.keys():
                    sentence_scores[sent] = word_freq[word]
                else:
                    sentence_scores[sent] += word_freq[word]

    # Summarization
    summary_sentences = nlargest(3, sentence_scores, key=sentence_scores.get)
    summary = ' '.join(summary_sentences)
    return summary
