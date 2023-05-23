from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold

from models.split_data import look_splits

def evaluate(models,url,names):
    """_summary_

    Args:
        models (_type_): _description_
        url (_type_): _description_
        names (_type_): _description_
    """
    for name, model in models:
        X_train, X_validation, Y_train, Y_validation = look_splits(url,names)
        kfold = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)
        cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring='accuracy')
        print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))
        