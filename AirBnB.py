import pandas as pd
import numpy as np
from collections import defaultdict
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt
from nltk.corpus import wordnet
import string
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.tokenize import WhitespaceTokenizer
from nltk.stem import WordNetLemmatizer
import re  
import nltk 
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer 
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Putting all the helper functions in one file. 


# Read file function
def read_file(file_path):
    '''
    INPUT
    file_path - where the file is located.
    OUTPUT
    df - dataframe of the file. 
    '''
    df = pd.read_csv(file_path)
    return df

# Get first five lines of the file. 

def get_five_lines(df):
    '''
    INPUT
    df - dataframe of some file. 
    OUTPUT
    five_lines - first five lines of the file. 
    '''
    five_lines=df.head()
    return five_lines

# outputting files from analysis

def output_to_csv(df, fpath, fname):
    '''
    INPUT
    df - dataframe of some file. 
    fpath - the path of the output directory.
    fname - name of the output file. 
    '''
    df.to_csv(fpath+fname)
    return

# Checking for missing columns

def get_no_missing_cols(df):
    '''
    INPUT
    df - dataframe of some file. 
    fpath - the path of the output directory.
    fname - name of the output file. 
    '''
    no_nulls_cols=set(df.columns[df.isnull().mean()==0])
    return no_nulls_cols

# Checking for nulls in columns

def check_cols_null(df, colname):
    '''
    INPUT
    df - dataframe of some file. 
    colname - column name of the column you want to check. 
    OUTPUT
    missing_mean - returns the percentage of missing data
    '''
    missing_mean=df[colname].isnull().mean()
    return missing_mean

# Function to graph and find optimal linear model

def find_optimal_lm_mod(X, y, cutoffs, test_size = .30, random_state=42, plot=True):
    '''
    INPUT
    X - pandas dataframe, X matrix
    y - pandas dataframe, response variable
    cutoffs - list of ints, cutoff for number of non-zero values in dummy categorical vars
    test_size - float between 0 and 1, default 0.3, determines the proportion of data as test data
    random_state - int, default 42, controls random state for train_test_split
    plot - boolean, default 0.3, True to plot result

    OUTPUT
    r2_scores_test - list of floats of r2 scores on the test data
    r2_scores_train - list of floats of r2 scores on the train data
    lm_model - model object from sklearn
    X_train, X_test, y_train, y_test - output from sklearn train test split used for optimal model
    '''
    r2_scores_test, r2_scores_train, num_feats, results = [], [], [], dict()
    for cutoff in cutoffs:

        #reduce X matrix
        reduce_X = X.iloc[:, np.where((X.sum() > cutoff) == True)[0]]
        num_feats.append(reduce_X.shape[1])

        #split the data into train and test
        X_train, X_test, y_train, y_test = train_test_split(reduce_X, y, test_size = test_size, random_state=random_state)

        #fit the model and obtain pred response
        lm_model = LinearRegression(normalize=True)
        lm_model.fit(X_train, y_train)
        y_test_preds = lm_model.predict(X_test)
        y_train_preds = lm_model.predict(X_train)

        #append the r2 value from the test set
        r2_scores_test.append(r2_score(y_test, y_test_preds))
        r2_scores_train.append(r2_score(y_train, y_train_preds))
        results[str(cutoff)] = r2_score(y_test, y_test_preds)

    if plot:
        plt.plot(num_feats, r2_scores_test, label="Test", alpha=.5)
        plt.plot(num_feats, r2_scores_train, label="Train", alpha=.5)
        plt.xlabel('Number of Features')
        plt.ylabel('Rsquared')
        plt.title('Rsquared by Number of Features')
        plt.legend(loc=1)
        plt.show()

    best_cutoff = max(results, key=results.get)

    #reduce X matrix
    reduce_X = X.iloc[:, np.where((X.sum() > int(best_cutoff)) == True)[0]]
    num_feats.append(reduce_X.shape[1])

    #split the data into train and test
    X_train, X_test, y_train, y_test = train_test_split(reduce_X, y, test_size = test_size, random_state=random_state)

    #fit the model
    lm_model = LinearRegression(normalize=True)
    lm_model.fit(X_train, y_train)

    return r2_scores_test, r2_scores_train, lm_model, X_train, X_test, y_train, y_test

# Creating dummy columns for categorical variable analysis

def create_dummy_df(df, cat_cols, dummy_na):
    '''
    INPUT:
    df - pandas dataframe with categorical variables you want to dummy
    cat_cols - list of strings that are associated with names of the categorical columns
    dummy_na - Bool holding whether you want to dummy NA vals of categorical columns or not
    
    OUTPUT:
    df - a new dataframe that has the following characteristics:
            1. contains all columns that were not specified as categorical
            2. removes all the original columns in cat_cols
            3. dummy columns for each of the categorical columns in cat_cols
            4. if dummy_na is True - it also contains dummy columns for the NaN values
            5. Use a prefix of the column name with an underscore (_) for separating 
    '''
    for col in  cat_cols:
        try:
            # for each cat add dummy var, drop original column
            df = pd.concat([df.drop(col, axis=1), pd.get_dummies(df[col], prefix=col, prefix_sep='_', drop_first=True, dummy_na=dummy_na)], axis=1)
        except:
            continue
    return df


# A Linear Model for the variables

def clean_fit_linear_mod(df, response_col, cat_cols, dummy_na, test_size=.3, rand_state=42):
    
    '''
    INPUT:
    df - a dataframe holding all the variables of interest
    response_col - a string holding the name of the column 
    cat_cols - list of strings that are associated with names of the categorical columns
    dummy_na - Bool holding whether you want to dummy NA vals of categorical columns or not
    test_size - a float between [0,1] about what proportion of data should be in the test dataset
    rand_state - an int that is provided as the random state for splitting the data into training and test 
    
    OUTPUT:
    test_score - float - r2 score on the test data
    train_score - float - r2 score on the test data
    lm_model - model object from sklearn
    X_train, X_test, y_train, y_test - output from sklearn train test split used for optimal model
    
    '''
    
    df.dropna(subset=[response_col])
    df= df.dropna(how='all', axis=1)
    
    if dummy_na:
        df = create_dummy_df(df, cat_cols, dummy_na)
    
    # Mean function
    fill_mean = lambda col: col.fillna(col.mean())
    # Fill the mean
    df = df.apply(fill_mean, axis=0)

    #Split into explanatory and response variables
    X = df.drop(response_col, axis=1)
    y = df[response_col]

    #Split into train and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=rand_state)

    lm_model = LinearRegression(normalize=True) # Instantiate
    lm_model.fit(X_train, y_train) #Fit

    #Predict using your model
    y_test_preds = lm_model.predict(X_test)
    y_train_preds = lm_model.predict(X_train)

    #Score using your model
    test_score = r2_score(y_test, y_test_preds)
    train_score = r2_score(y_train, y_train_preds)

    return test_score, train_score, lm_model, X_train, X_test, y_train, y_test, y_test_preds

# Wordnet tags processing
def get_wordnet_pos(pos_tag):
    '''
    INPUT
    pos_tag - wordnet tags 
    OUTPUT
    tag - wordnet tags for nouns, adj, etc.. 
    '''
    if pos_tag.startswith('J'):
        return wordnet.ADJ
    elif pos_tag.startswith('V'):
        return wordnet.VERB
    elif pos_tag.startswith('N'):
        return wordnet.NOUN
    elif pos_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN
    
# Cleaning texts for NLP

def clean_text(text):
    '''
    INPUT
    text - text document to clean
    OUTPUT
    clean text - text that's been cleaned. 
    '''
    # lower text
    text = text.lower()
    # tokenize text and remove puncutation
    text = [word.strip(string.punctuation) for word in text.split(" ")]
    # remove words that contain numbers
    text = [word for word in text if not any(c.isdigit() for c in word)]
    # remove stop words
    stop = stopwords.words('english')
    text = [x for x in text if x not in stop]
    # remove empty tokens
    text = [t for t in text if len(t) > 0]
    # pos tag text
    pos_tags = pos_tag(text)
    # lemmatize text
    text = [WordNetLemmatizer().lemmatize(t[0], get_wordnet_pos(t[1])) for t in pos_tags]
    # remove words with only one letter
    text = [t for t in text if len(t) > 1]
    # join all
    text = " ".join(text)
    return(text)

# Fitting TfDf Random Forest model

def clean_fit_tfdf_random_forest(df, X_col, y_col, test_size, rand_state, n_estimators, max_features, min_df, max_df):
    
    '''
    INPUT:
    df - a dataframe holding all the variables of interest
    X_col - a string holding the name of the column for X
    y_col - a string holding the name of the column for y
    test_size - a float between [0,1] about what proportion of data should be in the test dataset
    rand_state - an int that is provided as the random state for splitting the data into training and test 
    max_feature - an int that specifies maximum features allowed. 
    min_df - an int that is min df 
    max_df - an int that is max_df
    
    OUTPUT:
    predictions - an object of predictions of Y
    confusion_matrix - an object of confusion matrix
    classification_report - an object of classification report
    accuracy_score - an object of accuracy score
    '''
    #
    tfidfconverter = TfidfVectorizer(max_features=max_features, min_df=min_df, max_df=max_df, stop_words=stopwords.words('english'))  
    X = tfidfconverter.fit_transform(df[X_col]).toarray()
    y=df[y_col]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=rand_state)
    
    text_classifier = RandomForestClassifier(n_estimators=n_estimators, random_state=rand_state)  
    text_classifier.fit(X_train, y_train)
    
    predictions = text_classifier.predict(X_test)
    
    conf_matrix=confusion_matrix(y_test,predictions)
    class_report=classification_report(y_test,predictions) 
    accu_score=accuracy_score(y_test, predictions)
    
    return predictions, conf_matrix, class_report, accu_score