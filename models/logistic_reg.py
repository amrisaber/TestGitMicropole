from sklearn.linear_model import LogisticRegression

def regression_log():
    """build regression model
    """
    return(LogisticRegression(solver='liblinear', multi_class='ovr'))