######################################
# Import libraries
######################################

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

import seaborn as sns
import os, re, random, string
from collections import defaultdict
import nltk
import contractions
from nltk.tokenize import TweetTokenizer

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import euclidean_distances

# Gensim
import gensim
from gensim.models import Phrases

pd.set_option('display.max_colwidth',None)

######################################
def add_to_topic(df):
######################################    
    '''
    Add tweets to "topics", based on selected keywords. 
    This will be useful for finding insights in plots in later notebooks. 

    For example, 
    - any tweet mentioning movie-related terms (e.g. "watch_night", "star_treck", "movi", etc.) -> category: "movies"
    - tweets mentioning politics-related terms (e.g. "north_korea", "obama", "pelosi", "bush", etc.) ->  category : "politics"
    - tweets mentionning sports-related terms (e.g. "lebron", "laker", "basebal", "fifa", etc.)  ->  category : "sports"
    
    And so on.
    '''

    # Define keywords to look for
    selected_keywords = ['gm',                                                  # general motors (car industry)
                         'nikon','canon',                                       # DSLR cameras
                         'time_warner','time-warn','comcast','espn',            # cable-tv
                         'night_museum','star_trek','see_star','movi',          # movies
                         'watch_night','see_night',                             # movies
                         'dentist','tooth',                                     # dentist
                         'lebron','laker','basebal','basketbal','fifa','ncaa',  # sports
                         'sport','yanke','roger',                               # sports
                         'nike',                                                # Nike
                         'twitter_api','twitter','tweet',                       # Twitter
                         'phone','android','iphon','tether',                    # mobile devices
                         'latex','jqueri','linux','wolfram',                    # IT (Information technology)
                         'lambda', 'classif',                                   # IT (Information technology)
                         'north_korea','obama','pelosi','bush','china',         # politics
                         'india','iran','irancrazi','us',                       # politics
                         'eat','ate','food','mcdonald','safeway',               # food
                         'san_francisco','montreal','east_palo',                # cities
                         'book','malcolm_gladwel','kindl',                       # books
                         'blog', 'new_blog',                                    # blogging
                         'bobbi_flay',                                          # Chef
                         'buffet','warren','warren_buffet'                      # Warren Buffett (American investor)
                        ]

    # Define topics
    categories_dict = {'gm':'car industry',                                     # general motors (car industry)

                       'nikon':'DSLR cameras',                                  # electronic devices
                       'canon':'DSLR cameras',                            # electronic devices

                       'time_warner':'cable TV','time-warn':'cable TV',         # cable-tv
                       'comcast':'cable TV','espn':'cable TV',                  # cable-tv

                       'night_museum':'movies','star_trek':'movies',            # movies
                       'see_star':'movies','movi':'movies',                     # movies
                       'watch_night':'movies','see_night':'movies',             # movies

                       'dentist':'dentist','tooth':'dentist',                   # dentist

                       'lebron':'sports','laker':'sports',                      # sports
                       'basebal':'sports','basketbal':'sports',                 # sports
                       'fifa':'sports','ncaa':'sports',                         # sports
                       'sport':'sports','yanke':'sports','roger':'sports',      # sports

                       'nike': 'Nike',                                          # Nike

                       'twitter_api':'Twitter','twitter':'Twitter',             # Twitter
                       'tweet':'Twitter',                                       # Twitter

                       'phone':'mobile devices','android':'mobile devices',     # mobile devices
                       'iphon':'mobile devices','tether':'mobile devices',      # mobile devices

                       'latex':'IT','jqueri':'IT','classif':'IT',               # IT (Information technology)
                       'linux':'IT','wolfram':'IT',                             # IT (Information technology)
                       'lambda':'IT',                                           # IT (Information technology)

                       'north_korea':'politics','obama':'politics',             # politics
                       'pelosi':'politics','bush':'politics',                   # politics
                       'china':'politics', 'india':'politics',                  # politics
                       'iran':'politics','irancrazi':'politics',                # politics
                       'us':'politics',                                         # politics

                       'eat':'food','ate':'food','food':'food',                 # food
                       'mcdonald':'food','safeway':'food',                      # food

                       'san_francisco':'cities','montreal':'cities',            # cities
                       'east_palo':'cities',                                    # cities

                       'kindl':'books',                                         # books
                       'book':'books','malcolm_gladwel':'books',                # books
                       'blog':'blogging', 'new_blog':'blogging',                # blogging
                       'bobbi_flay':"Bobby Flay",                               # Chef

                       'buffet':'Warren Buffett','warren':'Warren Buffett',     # Warren Buffett (American investor)
                       'warren_buffet':'Warren Buffett'                         # Warren Buffett (American investor)

                       }

    # Look for these keywords in tweets
    keywords = []

    for tweet in df['processed_tweet']:
        cat = 'unlabeled'
        for keyword in selected_keywords:
            if keyword in tweet.split():
                cat = keyword
        keywords.append(cat)

    # Add keywords to DataFrame
    df['keyword'] = keywords
    
    # Add tweet to topic based on found keyword
    df['topic'] = df['keyword'].replace(categories_dict)
    
    # Remove column 'keyword'
    df.drop('keyword',axis=1,inplace=True)
    
    return df

##############################################################################
def plot_most_frequent_terms(frequency_dict, terms_to_plot,add_to_title=None):
##############################################################################
    '''
    Plots the most frequent collocations in corpus
    
    INPUTS:
    - frequency_dict:  dictionary, a dictionary mapping terms to raw counts
    - terms_to_plot :  integer, number of (most frequent) collocations to plot
    - add_to_title  :  string, complementary string for the plot title  
                       The title defaults to: "Top X terms" -> use add_to_title to complete
                       the title string (optional)
    
    OUTPUT:
    - Barplot of most frequent terms and their respective frequency
    
    '''
    
    # Barplot and font specifications
    barplot_specs = {'color':'mediumpurple','alpha':0.7,'edgecolor':'grey'}
    title_specs = {'fontsize':14, "fontweight": "bold", "y": 1.2}
    label_specs = {'fontsize':13}
    ticks_specs = {'fontsize':12}
    
    title = 'Top '+str(terms_to_plot)+' terms'
    
    if add_to_title == None:
        title = title
    else:
        title = title + ' '+ str(add_to_title)
        
    ylabel = 'Counts'

    # Plot top terms and their frequency
    plt.figure(figsize=(18,3))
    sns.barplot(x = list(frequency_dict.keys())[0:terms_to_plot], y = list(frequency_dict.values())[0:terms_to_plot],**barplot_specs)
    plt.ylabel(ylabel,**label_specs)
    plt.title(title,**title_specs)
    plt.xticks(rotation=90,**ticks_specs)
    plt.yticks(**ticks_specs); 
    plt.yscale('log');
    
    
####################################
def clean_tweet_plot(tweet):
####################################    
#    import nltk, contractions

    # Import tokenizer
#    from nltk.tokenize import TweetTokenizer

    # Create an instance  of the tokenizer
    tokenizer = TweetTokenizer(reduce_len=True, strip_handles=True)
    
    '''
    INPUT: 
    - tweet: raw text
    
    OUTPUT:
    - clean_tweet: cleaned text
    '''

    # Remove RT
    clean_tweet = re.sub(r'RT','',tweet)

    # Remove URL
    clean_tweet = re.sub(r'https?:\/\/[^\s]+','',clean_tweet)

    # Remove hash #
    clean_tweet = re.sub(r'#','',clean_tweet)
        
    # Remove twitter username
    clean_tweet = re.sub(r'@[A-Za-z]+','',clean_tweet)
    
    # Remove punctuation repetions (that are not removed by TweetTokenizer)
    clean_tweet = re.sub(r'([._]){2,}','',clean_tweet)
    
    # Case conversion
    clean_tweet = clean_tweet.lower()
    
    # Remove non-ascii chars
    clean_tweet = ''.join([c for c in str(clean_tweet) if ord(c) < 128])

    # Expand contractions
    clean_tweet = contractions.fix(clean_tweet)
    
    # Tokenize tweet
    tokens = tokenizer.tokenize(clean_tweet)

    # Join tokens in a single string to recreate the tweet
    clean_tweet = ' '.join([tok for tok in tokens])
    
    clean_tweet = re.sub(r'\s\.','.',clean_tweet)
    clean_tweet = re.sub(r'\s,',',',clean_tweet)
    clean_tweet = re.sub(r'\s!','!',clean_tweet)
    clean_tweet = re.sub(r'\s\?','?',clean_tweet)
    clean_tweet = re.sub(r'\$','',clean_tweet)
    clean_tweet = re.sub(r'\s+',' ',clean_tweet)
    
    clean_tweet = clean_tweet.strip()
    clean_tweet = re.sub(r'^[:-]','',clean_tweet)
    clean_tweet = clean_tweet.strip()
    
    short_text = clean_tweet.split()
    
    return ' '.join(short_text[:5])+'...'

#########################
def normalize_vector(v):
#######################
    if len(v.shape) == 2:
        return v/np.linalg.norm(v,axis=1).reshape(-1,1)
    else:
        return v/np.linalg.norm(v)
    
#######################
# Plot cosine similarity using heatmaps

def plot_similarity(features):
#################################
    plt.figure(figsize=(20,20))
    corr = features 
    mask = np.triu(np.ones_like(corr, dtype=bool))

    g = sns.heatmap(
        corr,
        vmin=0,
        vmax=features.max().max(),
        cmap= "YlOrRd"
    )
    g.set_title("Semantic Textual Similarity")

##############################################
def visualize_bow_embeddings(X1,X2,df,label):
################################################   
    title_specs = {'fontsize':16} #,'fontweight':'bold'}
    label_specs = {'fontsize':14}
    ticks_specs = {'fontsize':13}

    fig, axes = plt.subplots(1,2,figsize=(13,6))

    idx = df['topic'] == label

    axes[0].scatter(X1[idx,0],X1[idx,1],color="none", edgecolor='m',label=label);
    axes[0].scatter(X1[~idx,0],X1[~idx,1],color="none", edgecolor='grey',alpha=0.3,label=None);
    axes[0].set_xlabel('TSNE 1',**label_specs)
    axes[0].set_ylabel('TSNE 2',**label_specs)
    axes[0].set_title('Bag-of-words',**title_specs)

    axes[1].scatter(X2[idx,0],X2[idx,1],color="none", edgecolor='b',label=label);
    axes[1].scatter(X2[~idx,0],X2[~idx,1],color="none", edgecolor='grey',alpha=0.3,label=None);
    axes[1].set_title('Embeddings',**title_specs)
    axes[1].set_xlabel('TSNE 1',**label_specs)
    axes[1].set_ylabel('TSNE 2',**label_specs)
    
    if label == 'movies':
        add_text(X1,idx=51,text='Star Trek', ax = axes[0]) #x=0.1)
        add_text(X1,idx=124,text='Night at the museum', ax = axes[0]) #,y=0.2)
        
        add_text(X2,idx=51,text='Star Trek', ax = axes[1], y=0.8)
        add_text(X2,idx=124,text='Night at the museum', ax = axes[1]) #,y=0.2)
        
    elif label == 'electronic devices':
        add_text(X1, 76,text='Canon EOS', ax = axes[0])
        add_text(X1, 267,text='Canon 40D', ax = axes[0])

        add_text(X2, 76,text='Canon EOS', ax = axes[1])
        add_text(X2, 267,text='Canon 40D', ax = axes[1])
        
    elif label == 'politics':
        add_text(X1, 375,text='Obama', ax = axes[0]) 
        add_text(X1, 86,text='North Korea', ax = axes[0])
        add_text(X1, 157,text='China', ax = axes[0]) #,y=0.1)
        add_text(X1, 247,text='Clinton', ax = axes[0]) #,x=0.2,y=-0.2)
        add_text(X1, 484,text='Iran', ax = axes[0]) #,x=-0.7)
        
        add_text(X2, 375,text='Obama', ax = axes[1]) 
        add_text(X2, 86,text='North Korea', ax = axes[1])
        add_text(X2, 157,text='China', ax = axes[1],y=-0.5)
        add_text(X2, 247,text='Clinton', ax = axes[1]) #,x=0.2,y=-0.2)
        add_text(X2, 484,text='Iran', ax = axes[1]) #,x=-0.7)
        
    elif label == 'Nike':
        add_text(X1, 73,text='Nike', ax = axes[0])
        add_text(X2, 73,text='Nike', ax = axes[1])
        
    elif label == 'mobile devices':
        add_text(X1, 227,text='iPhone', ax = axes[0])
        add_text(X1, 23,text='iPhone app', ax = axes[0]) 
        
        add_text(X2, 227,text='iPhone', ax = axes[1])
        add_text(X2, 23,text='iPhone app', ax = axes[1]) 
        
    elif label == 'Twitter':
        add_text(X1, 464,text='tweet', ax = axes[0])
        add_text(X1, 8,text='Twitter', ax = axes[0])
        add_text(X1, 60,text='Twitter API', ax = axes[0])

        add_text(X2, 464,text='tweet', ax = axes[1])
        add_text(X2, 8,text='Twitter', ax = axes[1])
        add_text(X2, 60,text='Twitter API', ax = axes[1])
        
    elif label == 'sports':
        add_text(X1, 19,text='Lebron', ax = axes[0])
        add_text(X1, 119,text='Lakers', ax = axes[0])
        add_text(X1, 171,text='NCAA', ax = axes[0])
        add_text(X1, 207,text='All-Star basket', ax = axes[0])
        add_text(X1, 404,text='NY Yankees', ax = axes[0])
        
        add_text(X2, 19,text='Lebron', ax = axes[1])
        add_text(X2, 119,text='Lakers', ax = axes[1])
        add_text(X2, 171,text='NCAA', ax = axes[1])
        add_text(X2, 207,text='All-Star basket', ax = axes[1], y = -1.4)
        add_text(X2, 404,text='NY Yankees', ax = axes[1])
        
    elif label == 'IT': 
        add_text(X1, 7,text='Jquery', ax = axes[0])
        add_text(X1, 480,text='LaTeX', ax = axes[0])
        add_text(X1, 319,text='λ-calculus', ax = axes[0])
        
        add_text(X2, 7,text='Jquery', ax = axes[1])
        add_text(X2, 480,text='LaTeX', ax = axes[1])
        add_text(X2, 319,text='λ-calculus', ax = axes[1])
        
    elif label == 'books':
        add_text(X1, 1,text='kindle2', ax = axes[0])
        add_text(X1, 0,text='kindle2', ax = axes[0])
        add_text(X1, 57,text='malcolm gladwell book', ax = axes[0])
        add_text(X1, 106,text='jQuery book', ax = axes[0])      
        
        add_text(X2, 1,text='kindle2', ax = axes[1])
        add_text(X2, 0,text='kindle2', ax = axes[1], y=-1.)
        add_text(X2, 57,text='malcolm gladwell book', ax = axes[1])
        add_text(X2, 106,text='jQuery book', ax = axes[1])


    plt.legend()
    plt.tight_layout();
    
    
#################################################
def add_text(embedding,idx,text,ax,x=0.0,y=0.0):
#################################################
    ax.annotate(text,(embedding[idx,0]+x,embedding[idx,1]+y),fontsize=12)
    
##################################################################################### 
def plot_most_frequent_words(frequency_dict, terms_to_plot, doc_idx, y_vals = 'raw'):
#####################################################################################
    '''
    Plots the most frequent terms in document
    
    INPUTS:
    - frequency_dict:  a dictionary mapping terms to counts (dictionary)
    - terms_to_plot :  number of (most frequent) terms to plot (integer)
    - doc_idx       :  document index (integer)
    - y_vals        :  Specify if plotting raw counts or weighted frequencies
                       along y-axis (string) : 'raw' or 'tfidf'
    
    OUTPUT:
    - Barplot of most frequent terms and their respective frequency
    
    '''
    
    # Barplot and font specifications
    barplot_specs = {'color':'mediumpurple','alpha':0.7,'edgecolor':'grey'}
    title_specs = {'fontsize':16} #,'fontweight':'bold'}
    label_specs = {'fontsize':14}
    ticks_specs = {'fontsize':13}
    
    title = '{} most frequent terms in document {}'.format(str(terms_to_plot), doc_idx)
    
    if y_vals == 'raw':
        ylabel = 'Counts'
    elif y_vals == 'tfidf':
        ylabel = 'Tf-idf weights'

    # Plot top terms and their frequency
    plt.figure(figsize=(18,2.8))
    sns.barplot(x = list(frequency_dict.keys())[0:terms_to_plot], y = list(frequency_dict.values())[0:terms_to_plot],**barplot_specs)
    plt.ylabel(ylabel,**label_specs)
    plt.title(title,**title_specs)
    plt.xticks(rotation=80,**ticks_specs)
    plt.yticks(**ticks_specs); 

##########################################    
def sort_words_by_weight(X,vocab,doc_idx):
##########################################    
    '''
    Sort words in a given document in descending order based on tf-idf weights.
    
    INPUTS:
    - X       : Tf-idf vectors of corpus (numpy array)
    - vocab   : Tf-idf vocabulary (list)
    - doc_idx : document index (integer)
    
    OUTPUTS:
    - sorted_words    : sorted words in descending order based on tf-idf weights
    - sorted_weights  : sorted weights in descending order based on tf-idf weights
    
    '''
    
    weights = X[doc_idx].toarray().squeeze()
    words = vocab

    sorted_indexes   = np.argsort(weights)[::-1]  # Descending order
    sorted_words     = np.array(words)[sorted_indexes]
    sorted_weights   = weights[sorted_indexes]
    
    return (sorted_words,sorted_weights)

#############################################################################  
def print_top_terms_in_topic(topic_nbr, terms_to_print, pca_model, vocab):
#############################################################################   
    '''
    Prints the top most contributing terms to a given topic.
    
    INPUTS:
    - topic_nbr      : Topic (principal component) index (integer)
    - pca_model      : fitted pca model (model)
    - vocab          : Tf-idf vocabulary (list)
    - terms_to_print : number of contributing terms to print
    
    OUTPUT:
    - positive_contribs : most contributing (pos) terms to specified topic ()
    - negative_contribs : most contributing (neg) terms to specified topic ()

    '''

    coefficients    = pca_model.components_[topic_nbr]
    words           = np.array(vocab)
    sorted_indexes  = np.argsort(np.abs(coefficients))[::-1][0:terms_to_print]  # Descending order / Top-N
    sorted_words    = words[sorted_indexes].tolist()
    sorted_coefs    = coefficients[sorted_indexes].tolist()

    positive_contribs = []   # Positive contributions to specified topic
    negative_contribs = []   # Negative contributions to specified topic
    
    for coef, word in zip(sorted_coefs,sorted_words):
        if coef < 0:
            negative_contribs.append('{}*\"{}\"'.format(np.round(coef,3),word))
        else:
            positive_contribs.append('{}*\"{}\"'.format(np.round(coef,3),word))
    
    print(positive_contribs)
    #print('\n','-'*120,'\n')
    #print(negative_contribs)
    print('\n')

#############################################################################
def find_top_topics_in_doc(doc_idx,df,X,topics_nbr,pca_model,vocab,num_words):
#############################################################################  
   
    '''
    Prints the top most important topics in a given document.
    
    INPUTS:
    - pca_model      : fitted pca model (model)
    - vocab          : Tf-idf vocabulary (list)
    - num_words      : number of contributing terms to print
    
    OUTPUT:

    '''
    
    print('Document number: {} - Publication date: {}\n'.format(doc_idx,df['pub_date'].iloc[doc_idx]))
    print(df.iloc[doc_idx,1],'\n')
    
    text_sample = df['text'].iloc[doc_idx]
    print('Document:\n')
    print(text_sample[0:2000])
    
    sorted_topic_idx  = np.argsort(X[doc_idx])[::-1]
    sorted_weights = X[doc_idx][sorted_topic_idx]

    #print('Dominant topics (top {}):\t\t{}'.format(topics_nbr, ['Topic '+str(i) for i in sorted_topic_idx[0:topics_nbr]]))
    #print('Contribution of dominant topics:\t{}\n'.format(sorted_weights[0:topics_nbr]))
    # Plot dominant topics in document

    # Plot specs
    barplot_specs   = {"color": '#BFBFBF', "alpha": 0.7, "edgecolor": "grey"}
    title_specs     = {"fontsize": 14, "fontweight": "bold", "y": 1.1}
    label_specs     = {"fontsize": 13}

    # y-labels
    y_pos = ['Topic '+str(i) for i in sorted_topic_idx[0:topics_nbr]]
    y_label_text = ['{}: {:5.3f}%'.format(y_pos[i], sorted_weights[i]) for i in np.arange(len(sorted_weights[0:topics_nbr]))]

    # Plot
    plt.title('Dominant topics in document {}'.format(doc_idx),**title_specs)
    plt.barh(np.arange(len(y_pos)),sorted_weights[0:topics_nbr], height = 0.6, **barplot_specs);
    plt.yticks(np.arange(len(y_pos)),y_label_text, **label_specs);
    plt.xlabel('Topic contribution',**label_specs);
    plt.gca().invert_yaxis()
    plt.show()

    # What are they are about:
    for pc in sorted_topic_idx[0:topics_nbr]:
        print('Topic {}'.format(pc))
        print('_'*len('Topic {}'.format(pc)),'\n')
        print_top_terms_in_topic(pc, num_words, pca_model, vocab)
                    

#############################################################################
# Dictionary used to anchor the colormap for plotting the topic embeddings
#############################################################################
vmin_dict = {'Twitter':0.09,
             'politics':0.087,
             'Nike':0.093,
             'IT':0.092,
             'sports':0.089,
             'movies':0.098,
             'cable TV':0.09,
             'mobile devices':0.092,
             'DSLR cameras':0.093,
             'unlabeled':0.08,
             'car industry':0.088,
             'books':0.092,
             'cities':0.09,
             'dentist':0.09,
             'blogging':0.09,
             'food':0.09,
             'Warren Buffett':0.09, 
             'Bobby Flay':0.09}

#############################################################################
def find_neighbohrs(idx, X, data):
#############################################################################
   
    '''
    This function takes a tweet id as an argument and returns the most similar
    tweets according to cosine similarity.
    
    INPUTS:
        - idx  : integer, tweet id (range: 0 - len(df)-1)
        - X    : numpy array, vector representation of text (default: None)
        - data : dataframe, text data
    
    OUTPUT:
        - Top 10 most similar tweets according to cosine similarity.
    '''
    
    neighbohrs_df = pd.DataFrame()
    
    df_cos  = pd.DataFrame(cosine_similarity(X))
    df_dist = pd.DataFrame(euclidean_distances(normalize_vector(X)))
    
    # Sort neighbors with respect to cosine similarity
    neighborhs = np.argsort(df_cos.iloc[idx,:])[::-1]
    
    neighbohrs_df['processed_tweet'] = data.iloc[neighborhs,1]
    neighbohrs_df['cosine_similarity'] = df_cos.iloc[idx,neighborhs]
    neighbohrs_df['euclidean_distance'] = df_dist.iloc[idx,neighborhs]
    neighbohrs_df['label'] = data.iloc[neighborhs,-2]
    neighbohrs_df['topic'] = data.iloc[neighborhs,-1]

    return neighbohrs_df.head(10)