# Natural Language Processing
## 'Sentiment Analyses of Movie Reviews: A Classification Problem'

Improving sentiment analyses using objectivity classification. We intend to explore the relationship between objectivity and sentiment analysis by pre-sanitizing our review dataset with nltk's Naive Bayes Classification model which was trained to define objectivity using the 'rotten_imdb' dataset. We then trained an LSTM model using different training sets and tested on IMDB's Large Movie Review dataset to see if pre-santizing data has an impact on sentiment analysis.

> Authored by Rajan Bharaj and Sangeyl Lee 

## Instructions for use
### Enviroment 

- Using Python version 3.7.9, 64-bit (for tensorflow version 1.x)
- Using Tensorflow version 1.15.0 (will not work with version 2.x)
- Make sure you have installed all relevant python libraries to run programs [nltk, tensorflow, scikit-learn, ...]

### Instructions

- Download 'nltk-naive_bayes_classification.py', 'rating-automation.py' programs and 'rotten_imdb', 'aclImdb' datasets
- Make sure all the files/folders are in the same parent location
- Open & run 'nltk-naive_bayes_classification.py' on your local machine to train the classification model
- Open 'rating-automation.py' on your local machine
- Starting on line 4: Update folder paths of 'rating-automation.py' to match the location on your local machine
- Run 'rating-automation.py'

## References

1. Andrew L. Maas, Raymond E. Daly, Peter T. Pham, Dan Huang, Andrew Y. Ng, and Christopher Potts. 2011. Learning Word Vectors for Sentiment Analysis. In Proceedings of the 49th Annual Meeting of the Association for Computational Linguistics: Human Language Technologies, pages 142–150, Portland, Oregon, USA. Association for Computational Linguistics.

2. “Understanding LSTM Networks.” Understanding LSTM Networks -- Colah's Blog, https://colah.github.io/posts/2015-08-Understanding-LSTMs/. 
