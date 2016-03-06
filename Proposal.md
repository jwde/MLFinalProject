# Political Sentiment Analysis using State of the Union Transcripts

<h2>Introduction</h2>
Political Sentiment analysis has mostly used data from social media sources to analyze political leanings. This paper will focus on rhetoric from political speeches. It will consider the transcripts of State of the Union addresses to build a model to classify political speech as either Democrat-leaning or Republican-leaning. An effective model for political leaning could provide a more objective measure of the neutrality or divisiveness of political speech and would be particularly interesting giving the upcoming presidential election.
<h2>Classification</h2>
The paper will consider a Naive Bayes classifier. Naive Bayes classifiers have been used with success in other text classification problems such as spam detection as well as more general sentiment analysis. It may consider and compare other models.
<h2>Data</h2>
In order to be effective, there must be a sufficiently large number of speeches, but they must also be recent enough to be representative of the current state of the Democratic and Republican parties. We will consider all State of the Union addresses between now and 1934. Full transcripts are available at <a href='stateoftheunion.onetwothree.net/texts/index.html'>here</a>. State of the Union addresses are a good source for training since they are unlikely to contain sarcasm, bad spelling, or grammatical issues.
<h2>Features</h2>
We will consider n-gram, tf-idf, and bag of words features.
<h2>Experiments</h2>
We will use 10-fold stratified cross-validation on the set of speech transcripts. Features will be selected based on Z-score.
<h2>References</h2>

"State Of The Union Addresses 1790-2016". State Of The Union. N.p., 2016. Web. 6 Mar. 2016.

Bakliwal, Akshat et al. "Sentiment Analysis Of Political Tweets: Towards An Accurate Classifier". Proceedings of the Workshop on Language in Social Media (LASM 2013) (2013): 49-58. Print.

Kummer, Olena, and Jacques Savoy. "Feature Selection In Sentiment Analysis". CORIA (2012): 273-284. Print.

Resnik, Philip, and Eric Hardisty. "Gibbs Sampling For The Uninitiated". (2010): 1-22. Print.
