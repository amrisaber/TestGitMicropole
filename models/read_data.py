from pandas import read_csv

def read(url, names):
    """Load dataset

    Args:
        url (_type_): _description_
        names (_type_): model names
    """
    dataset = read_csv(url, names=names)   
    return(dataset)
    

