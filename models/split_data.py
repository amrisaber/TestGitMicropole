from sklearn.model_selection import train_test_split
from models.read_data import read

def look_splits(url,names):
    """build training and data test

    Args:
        url (_type_): _description_
        names (_type_): _description_
    """
    array = read(url,names).values
    X = array[:,0:4]
    y = array[:,4]
    X_train, X_validation, Y_train, Y_validation = train_test_split(X, y, test_size=0.20, random_state=1, shuffle=True)
    return(X_train, X_validation, Y_train, Y_validation)