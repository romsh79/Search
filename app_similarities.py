import pandas as pd
import numpy as np
import streamlit as st
import gensim
#import time

# Load the data and models
# dictionary based on all having news
dictionary = gensim.corpora.Dictionary.load('data/newsdict.dict')
# corpus created from the all having news
corpus = gensim.corpora.MmCorpus("data/corpus.mm")
# the sentences derived from news
sentences_orig = pd.read_csv('data/True_news_sentences.csv')
# the news itself
df = pd.read_csv('data/True_news.csv')

# Header    
st.title('Search engine')
st.write("""Here we will find the most relevant news to the user request""")
# User input for search
q = st.text_area("User query",value="Russian operation in Syria")
vec = dictionary.doc2bow(q.lower().split())
tfidf = gensim.models.TfidfModel(corpus)
vec = tfidf[vec]

# Tune the side panel
st.sidebar.title("About")
st.sidebar.info(
    """
    This dashboard realizes ML model of search engine.
    """
)
#st.sidebar.info("Some text [link in web](https://github.com/yuliianikolaenko/COVID_dashboard_proglib).")
st.sidebar.image("grandma.jpg")

# Start engine
# How many news (in %) to use in the search
corpus_len_perc = st.sidebar.slider("Which percentage of the whole news to use", min_value=1, max_value=100, value=100)
corpus_len = int(corpus_len_perc*len(corpus)/100)
result = st.button('Start search')
if result:
	#start_time = time.time()
	for strt in range(0,corpus_len,10000):
		index = 0
		fin = min([strt+10000,corpus_len])
		index = gensim.similarities.MatrixSimilarity(tfidf[corpus[strt:fin]])
		if strt == 0:
			sims = index[vec]
		else:
			sims = np.concatenate((sims,index[vec]), axis=0)
	sims = sorted(enumerate(sims), key=lambda item: -item[1])
	# the time of execution
	# exectime = round((time.time() - start_time)/60.0,1)
	#prntstr = 'The search is finished in ' + str(exectime) + " minutes"
	st.write('The search is finished')

# How many search results to show
showdata = st.sidebar.slider("How many search results to show", min_value=1, max_value=10, value=3)

# Create checkbox on the side panel to show the full text of the news
show_news = st.sidebar.checkbox('Show the full news')
#if show_data == True:
    #st.subheader('Raw data')
    #st.markdown(
    #    "#### Data on twitter records")
    #st.write(df.head(showdata))
	
if result:
	st.write('**Results of search**')
	for i in sims[:showdata]:
		str_prnt = "sentence " + str(i[0]) + ": " + sentences_orig.sentence.iloc[i[0]] + "\n\n cosine distance: " + str(i[1])
		st.write(str_prnt)
		if show_news:
			sms = df[df.index == sentences_orig.news_index.iloc[i[0]]].text.iloc[0]
			st.write(sms)
