import collections
if not hasattr(collections, 'Mapping'):
    import collections.abc
    collections.Mapping = collections.abc.Mapping

import streamlit as st
from summarizers.spacy_summarizer import spacy_summarizer
from summarizers.gensim_summarizer import gensim_summarizer
from summarizers.nltk_summarizer import nltk_summarizer
from summarizers.sumy_summarizer import sumy_summarizer

from bs4 import BeautifulSoup
from urllib.request import urlopen
import time

def from_url(url):
    page = urlopen(url)
    soup = BeautifulSoup(page, 'html.parser')
    fetched_text = ' '.join(map(lambda p: p.text, soup.find_all('p')))
    return fetched_text

# Initialize session state for text area
if 'text' not in st.session_state:
    st.session_state['text'] = ''

st.title('Summarizer App')
choose = st.sidebar.radio("Choose Task", ["Summarize", "Compare Summarizers"])
model = st.sidebar.selectbox("Choose Model", ["NLTK", "Spacy", "Gensim", "Sumy"])

# File upload
upload = st.file_uploader("Upload a file", type=['txt', 'pdf'])
url = st.text_input("Enter a URL")

summaries = []

if upload:
    text = str(upload.read(), 'utf-8')
    st.write(text)
elif url:
    text = from_url(url)
    st.write(text)
else:
    text = st.text_area('Enter text to summarize', value=st.session_state['text'], key='text')

if choose == 'Summarize' and st.button('Summarize'):
    if model:
        start = time.time()
        if model == 'Spacy':
            summaries.append(('Spacy', spacy_summarizer(text)))
        elif model == 'Gensim':
            summaries.append(('Gensim', gensim_summarizer(text)))
        elif model == 'NLTK':
            summaries.append(('NLTK', nltk_summarizer(text)))
        elif model == 'Sumy':
            summaries.append(('Sumy', sumy_summarizer(text)))
        end = time.time()

        st.write(f"Time taken: {end - start:.2f} seconds")

        for name, summary in summaries:
            st.write(f"### {name} Summary")
            st.write(summary)

        # Call download button here
        for name, summary in summaries:
            st.download_button(f"Download {name} Summary", summary, f"{name}_summary.txt")

elif choose == "Compare Summarizers" and st.button("Compare"):
    start = time.time()
    spacy_summary = spacy_summarizer(text)
    gensim_summary = gensim_summarizer(text)
    nltk_summary = nltk_summarizer(text)
    sumy_summary = sumy_summarizer(text)
    end = time.time()

    summaries = [
        ("Spacy", spacy_summary),
        ("Gensim", gensim_summary),
        ("NLTK", nltk_summary),
        ("Sumy", sumy_summary)
    ]

    st.write(f"Time taken: {end - start:.2f} seconds")

    st.write("### Spacy Summary")
    st.write(spacy_summary)

    st.write("### Gensim Summary")
    st.write(gensim_summary)

    st.write("### NLTK Summary")
    st.write(nltk_summary)

    st.write("### Sumy Summary")
    st.write(sumy_summary)

    # Call download button here
    for name, summary in summaries:
        st.download_button(f"Download {name} Summary", summary, f"{name}_summary.txt")

