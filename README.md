# AirBnB
AirBnB Dataset Analysis

Overview

This project uses AirBnB Dataset that contains the following files: 
calendar.csv
listings.csv
reviews.csv

Analysis:

The analysis can be found in this blog post:
https://junwuwriting.medium.com/airbnb-data-analysis-be7a491be905

Data Science Questions asked:
1) Are there any price spikes for peak months? 
2) What are the most frequented neighborhoods in Seattle
3) Can we predict the price using number_of_reviews and review_score_rating?
4) Are the review comments reflective of the review_score_ratings? 

Code:

- AirBnB.ipynb jupyter notebook contains the first part of the analysis that includes the prediction price with linear model and logistic regression. 
- AirBnB.py contains the helper functions used for all of the analysis. 
- AirBnB_Sentiment.ipynb jupyter notebook contains the sentiment analysis using tf-df and random forest model. 

Libraries:

numpy - library used for matrix math
pandas - library used for dataframe handling
re - library used for regular expression
nltk - library used for natural language processing
sklearn - library used for modeling 
matplotlib - library used for graphing

All libraries can be installed using: pip install [library]

Acknowledgements:

Sentiment Analysis: First Steps With Python's NLTK library
https://realpython.com/python-nltk-sentiment-analysis/

Tf-IDF Documentation:
https://www.rdocumentation.org/packages/superml/versions/0.5.3/topics/TfIdfVectorizer

Re Documentation:
https://www.w3schools.com/python/python_regex.asp



