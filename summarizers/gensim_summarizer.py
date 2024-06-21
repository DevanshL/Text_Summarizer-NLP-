from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from gensim.corpora import Dictionary
from gensim.models import TfidfModel
import nltk

# Download NLTK stopwords and punkt tokenizer
nltk.download('stopwords')
nltk.download('punkt')

def gensim_summarizer(text):
    try:
        # Tokenize text into sentences
        sentences = sent_tokenize(text)

        # Tokenize each sentence into words and remove stopwords
        stop_words = set(stopwords.words('english'))
        tokenized_sentences = []
        for sentence in sentences:
            words = word_tokenize(sentence)
            filtered_words = [word.lower() for word in words if word.lower() not in stop_words and word.isalnum()]
            tokenized_sentences.append(filtered_words)

        # Create a dictionary and corpus for TF-IDF
        dictionary = Dictionary(tokenized_sentences)
        bow_corpus = [dictionary.doc2bow(sentence) for sentence in tokenized_sentences]
        tfidf = TfidfModel(bow_corpus)

        # Score each sentence based on TF-IDF
        sentence_scores = []
        for i, sentence in enumerate(bow_corpus):
            score = sum([tfidf[sentence][j][1] for j in range(len(tfidf[sentence]))])
            sentence_scores.append((score, i))

        # Sort sentences by score in descending order and select top sentences
        sentence_scores.sort(reverse=True, key=lambda x: x[0])
        num_sentences = min(5, len(sentence_scores))  # You can adjust the number of sentences in the summary
        selected_sentences = sorted([sentences[sentence_scores[i][1]] for i in range(num_sentences)])

        # Join the selected sentences to form the summary
        summary = ' '.join(selected_sentences)
        return summary  # Return summary as string
    except Exception as e:
        return f"Error generating summary: {str(e)}"