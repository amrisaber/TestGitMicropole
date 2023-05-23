from models.evaluate_model import evaluate
from models.read_data import read
from models.logistic_reg import regression_log
from models.decision_tree import tree
from models.svm_reg import svm
from models.split_data import look_splits

def main():
    """1- First update cagnotte
    """
    url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv"
    names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
    dataS = read(url,names)
    models = [('Reg',regression_log()), ('Cart',tree()), ('SVM',svm())]
    return(evaluate(models,url,names))

    
if __name__ == '__main__':
    main()