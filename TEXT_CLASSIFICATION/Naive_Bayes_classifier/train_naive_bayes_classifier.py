from nltk.corpus import stopwords
import numpy as np
import pandas as pd
import nltk
import string
import re
import spacy
from sklearn.model_selection import train_test_split

# python -m spacy download en_core_web_sm

nltk.download('stopwords')

class naive_bayes_classifier:
  
    """
    INIT ARGS:
    input_data_path: path to your csv file; note that there expected to be only two columns 'text' and 'label'
    output_dur: directory to output the trained classifier
    export_bool: whether to export the trained model and the files associated with it
    
    """

    def __init__(self, input_data_path: str, output_dir: str, export_bool:bool = True, load_clf:bool = False, verbose:bool = True):
        
        self.input_data_path = input_data_path
        self.outpit_dur = output_dir
        self.export_bool = export_bool
        
        
        if verbose: print('\nLOADING DATA...') 
        
        df = pd.read_csv(input_data_path, encoding='utf-8')
        
        texts = df['text'].to_list() # Use only text column
        labels = df['label'].to_list()

        train_texts, self.test_texts, train_labels, self.test_labels = train_test_split(texts, labels, test_size=0.2, random_state=42)
         
        self.test_labels = np.array(self.test_labels)
        
        self.nlp = spacy.load("en_core_web_sm")

        def process_tweet(tweet):
            """Process tweet function.
            Input:
                tweet: a string containing a tweet
            Output:
                tweets_clean: a list of words containing the processed tweet

            """
            # stemmer = PorterStemmer()
            stopwords_english = stopwords.words('english')
            # remove stock market tickers like $GE
            tweet = re.sub(r'\$\w*', '', tweet)
            # remove old style retweet text "RT"
            tweet = re.sub(r'^RT[\s]+', '', tweet)
            # remove hyperlinks
            tweet = re.sub(r'https?:\/\/.*[\r\n]*', '', tweet)
            # remove hashtags
            # only removing the hash # sign from the word
            tweet = re.sub(r'#', '', tweet)
            # tokenize tweets
            # tokenizer = TweetTokenizer(preserve_case=False, strip_handles=True,
            #                            reduce_len=False)
            # tweet_tokens = tokenizer.tokenize(tweet)

            doc = self.nlp(tweet)
            tweet_tokens = [token.lemma_ for token in doc]

            tweets_clean = []
            for word in tweet_tokens:
                if (word not in stopwords_english and  # remove stopwords
                        word not in string.punctuation):  # remove punctuation
                    tweets_clean.append(word)
                    # tweets_clean.append(word.lower())
                    # stem_word = stemmer.stem(word)  # stemming word
                    # tweets_clean.append(stem_word)

            return tweets_clean
        
        def build_freqs(tweets, ys):
            """Build frequencies.
            Input:
                tweets: a list of tweets
                ys: an m x 1 array with the sentiment label of each tweet
                    (either 0 or 1)
            Output:
                freqs: a dictionary mapping each (word, sentiment) pair to its
                frequency
            """
            # Convert np array to list since zip needs an iterable.
            # The squeeze is necessary or the list ends up with one element.
            # Also note that this is just a NOP if ys is already a list.
            yslist = np.squeeze(ys).tolist()

            # Start with an empty dictionary and populate it by looping over all tweets
            # and over all processed words in each tweet.
            freqs = {}
            for y, tweet in zip(yslist, tweets):
                for word in process_tweet(tweet):
                    pair = (word, y)
                    if pair in freqs:
                        freqs[pair] += 1
                    else:
                        freqs[pair] = 1

            return freqs
        def count_tweets(result, tweets, ys):
            '''
            Input:
                result: a dictionary that will be used to map each pair to its frequency
                tweets: a list of tweets
                ys: a list corresponding to the sentiment of each tweet (either 0 or 1)
            Output:
                result: a dictionary mapping each pair to its frequency
            '''
            for y, tweet in zip(ys, tweets):
                for word in process_tweet(tweet):
                    # define the key, which is the word and label tuple
                    pair = (word, y)

                    # if the key exists in the dictionary, increment the count
                    if pair in result:
                        result[pair] += 1

                    # else, if the key is new, add it to the dictionary and set the count to 1
                    else:
                        result[pair] = 1

            return result
        
        train_labels = np.array(train_labels)

        if verbose: print('\nTRAINING STARTED...')

        self.freqs = count_tweets({}, train_texts, train_labels)

        def train_naive_bayes(freqs, train_y):
            '''
            Input:
                freqs: dictionary from (word, label) to how often the word appears
                train_y: a list of labels correponding to the tweets (0,1)
            Output:
                logprior: the log prior.
                loglikelihood: the log likelihood of Naive bayes equation.
            '''
            loglikelihood = {}
            logprior = 0

            # calculate V, the number of unique words in the vocabulary
            vocab = set([pair[0] for pair in freqs.keys()])
            V = len(vocab)

            # calculate N_pos and N_neg
            N_pos = N_neg = 0
            for pair in freqs.keys():
                # if the label is positive (greater than zero)
                if pair[1] > 0:

                    # Increment the number of positive words by the count for this (word, label) pair
                    N_pos += freqs[pair]

                # else, the label is negative
                else:

                    # increment the number of negative words by the count for this (word,label) pair
                    N_neg += freqs[pair]

            # Calculate D, the number of documents
            D = len(train_y)

            # Calculate D_pos, the number of positive documents (*hint: use sum(<np_array>))
            D_pos = len([i for i in train_y if i == 1])

            # Calculate D_neg, the number of negative documents (*hint: compute using D and D_pos)
            D_neg = len([i for i in train_y if i == 0])

            # Calculate logprior
            logprior = np.log(np.asarray(D_pos)) - np.log(np.asarray(D_neg))

            # For each word in the vocabulary...
            for word in vocab:
                # get the positive and negative frequency of the word
                freq_pos = freqs.get((word, 1), 0) 
                freq_neg = freqs.get((word, 0), 0) 

                # calculate the probability that each word is positive, and negative
                p_w_pos = (freq_pos + 1) / (N_pos + V)
                p_w_neg = (freq_neg + 1) / (N_neg + V)

                # calculate the log likelihood of the word
                loglikelihood[word] = np.log(np.asarray(p_w_pos)) - np.log(np.asarray(p_w_neg))

            return logprior, loglikelihood
                
        self.logprior, self.loglikelihood = train_naive_bayes(self.freqs, train_labels)

        if verbose: print('\nTRAINING FINISHED!')
    def process_tweet(self, tweet):
        """Process tweet function.
        Input:
            tweet: a string containing a tweet
        Output:
            tweets_clean: a list of words containing the processed tweet

        """
        # stemmer = PorterStemmer()
        stopwords_english = stopwords.words('english')
        # remove stock market tickers like $GE
        tweet = re.sub(r'\$\w*', '', tweet)
        # remove old style retweet text "RT"
        tweet = re.sub(r'^RT[\s]+', '', tweet)
        # remove hyperlinks
        tweet = re.sub(r'https?:\/\/.*[\r\n]*', '', tweet)
        # remove hashtags
        # only removing the hash # sign from the word
        tweet = re.sub(r'#', '', tweet)
        # tokenize tweets
        # tokenizer = TweetTokenizer(preserve_case=False, strip_handles=True,
        #                            reduce_len=False)
        # tweet_tokens = tokenizer.tokenize(tweet)

        doc = self.nlp(tweet)
        tweet_tokens = [token.lemma_ for token in doc]

        tweets_clean = []
        for word in tweet_tokens:
            if (word not in stopwords_english and  # remove stopwords
                    word not in string.punctuation):  # remove punctuation
                tweets_clean.append(word)
                # tweets_clean.append(word.lower())
                # stem_word = stemmer.stem(word)  # stemming word
                # tweets_clean.append(stem_word)

        return tweets_clean
    def naive_bayes_predict(self, tweet):
        '''
        Input:
            tweet: a string
        Output:
            p: the sum of all the logliklihoods of each word in the tweet (if found in the dictionary) + logprior (a number)

        '''
        # process the tweet to get a list of words
        word_l = self.process_tweet(tweet)

        # initialize probability to zero
        p = 0

        # add the logprior
        p += self.logprior

        for word in word_l:

            # check if the word exists in the loglikelihood dictionary
            if word in self.loglikelihood:
                # add the log likelihood of that word to the probability
                p += self.loglikelihood[word]

        return p
    def test_naive_bayes(self, verbose:bool = True):
        """
        Input:

        Output:
            accuracy: (# of tweets classified correctly)/(total # of tweets)
        """
        if verbose: print('\nRUNNING TEST EVALUATION...')

        accuracy = 0 

        y_hats = []
        for tweet in self.test_texts:
            # if the prediction is > 0
            if self.naive_bayes_predict(tweet) > 0:
                # the predicted class is 1
                y_hat_i = 1
            else:
                # otherwise the predicted class is 0
                y_hat_i = 0

            # append the predicted class to the list y_hats
            y_hats.append(y_hat_i)

        # error is the average of the absolute values of the differences between y_hats and test_y
        error = np.mean(np.absolute(y_hats-self.test_labels))

        # Accuracy is 1 minus the error
        accuracy = 1 - error
        
        print(f"\nNaive Bayes test set accuracy = {accuracy:.4f}")
        
        pass
