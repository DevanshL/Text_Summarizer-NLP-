from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer
from sumy.summarizers.luhn import LuhnSummarizer
from sumy.summarizers.edmundson import EdmundsonSummarizer
from sumy.nlp.stemmers import Stemmer
from sumy.utils import get_stop_words

LANGUAGE = "english"
SENTENCES_COUNT = 3

def sumy_summarizer(text, method='lsa'):
    # Parsing the text
    parser = PlaintextParser.from_string(text, Tokenizer(LANGUAGE))
    stemmer = Stemmer(LANGUAGE)

    # Selecting the summarizer
    if method == 'lsa':
        summarizer = LsaSummarizer(stemmer)
    elif method == 'luhn':
        summarizer = LuhnSummarizer(stemmer)
    elif method == 'edmundson':
        summarizer = EdmundsonSummarizer(stemmer)
    else:
        summarizer = LsaSummarizer(stemmer)

    summarizer.stop_words = get_stop_words(LANGUAGE)

    # Summarization
    summary = summarizer(parser.document, SENTENCES_COUNT)
    return ' '.join([str(sentence) for sentence in summary])