"""
Trains and tests the logistic regression on CoLA data. This provides the
validation accuracy on the development set.
"""

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import seaborn as sn
import math

def import_data(filename):
    """
    Imports ccglink confidence scores on CoLA data from a file. The input file
    is a tab-separated file with 5 columns: the label for the sentence
    (1/0 for grammatical/ungrammatical), the length of the sentence, whether
    the sentence was parsed as a sentence, Missing Link's confidence score, and
    whether Missing Link retrained itself on the sentence (1/0). The result is
    returned as a pandas DataFrame.
    """

    grammaticality = []
    length = []
    parsed = []
    confidence = []
    retrain = []

    with open(filename, 'r') as infile:
        for line in infile:
            values = line.split('\t')
            conf = float(values[3])
            if math.isinf(conf)
            if math.isinf(conf) and values[0] == "0":
                conf = 500.0
            elif math.isinf(conf) and values[0] == "1":
                conf = 0.0
            grammaticality.append(int(values[0]))
            length.append(int(values[1]))
            parsed.append(int(values[2]))
            confidence.append(conf)
            retrain.append(int(values[4]))

    samples = {
        'grammaticality': grammaticality, 
        'length': length, 
        'parsed': parsed,
        'confidence': confidence,
        'retrain': retrain,
    }

    return pd.DataFrame(samples, columns=['grammaticality', 'length', 'parsed',
        'confidence', 'retrain'])

def main():

    print("Importing training data...")
    training = import_data('analysis/in_domain_scores.tsv')
    print("Done.")

    X_train = training[['length', 'parsed', 'confidence', 'retrain']]
    y_train = training['grammaticality']

    print("Importing testing data...")
    testing = import_data('analysis/out_of_domain_scores.tsv')
    print("Done.")

    X_test = testing[['length', 'parsed', 'confidence', 'retrain']]
    y_test = testing['grammaticality']

    print("Training regression.")
    regression = LogisticRegression(solver='lbfgs')
    regression.fit(X_train, y_train)
    print("Done.")

    print("Testing regression.")
    y_pred = regression.predict(X_test)
    print("Done.")

    print('MCC: ', metrics.matthews_corrcoef(y_test.values, y_pred))

if __name__ == "__main__":
    main()
