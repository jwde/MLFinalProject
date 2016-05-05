# Notes for paper

# Sentiment Analysis of Political Speeches

<h2>Background</h2>
The goal of political sentiment analysis is to gain insight into political opinions using natural language processing. In the context of this paper, we will consider political speeches, particularly state of the union addresses, and train probabilistic classifiers to predict political party affiliations. We will discuss the problem of selecting a set of features which is indicative of political affiliation, as well as training and classification algorithms.

<h2>Related Work</h2>
There exists research in the area of political sentiment analysis using data from social media sources, with the motivation of improving or replacing current polling systems (Bakliwal et al. 2013). In those cases, classification techniques including Naive Bayes and Support Vector Machines have been used with some success. Notable difficulties in political sentiment analysis on social media include recognizing sarcasm (Bakliwal et al. 2013) and spelling errors. It is reasonable to expect fewer of these kinds of problems when analyzing state of the union addresses, however the motivation of better polling no longer makes sense when analyzing speeches. If we can produce a successful probabilistic classifier for political speech, it can be used to objectively rank U.S. political speeches by party leaning, which would be interesting in order to track how individual politicians change views over time or to compare candidates in election cycles.

<h2>Method</h2>
<h3>Preprocessing</h3>
Following the standard used for preprocessing by Kummer & Savoy (2012), we tokenize the sentences and words in every speech, then replace every word occuring fewer than 4 times with an unknown token. We pad the start and end of every sentence with start and end tokens so that n-grams can represent words and phrases being at the start or end of a sentence.

<h3>Feature Generation</h3>

<h3>Feature Selection</h3>

<h3>Classification</h3>


<h2>Results</h2>

<h2>Conclusion and future work</h2>

<h2>References</h2>
"State Of The Union Addresses 1790-2016". State Of The Union. N.p., 2016. Web. 6 Mar. 2016.

Bakliwal, Akshat et al. "Sentiment Analysis Of Political Tweets: Towards An Accurate Classifier". Proceedings of the Workshop on Language in Social Media (LASM 2013) (2013): 49-58. Print.

Kummer, Olena, and Jacques Savoy. "Feature Selection In Sentiment Analysis". CORIA (2012): 273-284. Print.

Resnik, Philip, and Eric Hardisty. "Gibbs Sampling For The Uninitiated". (2010): 1-22. Print.
