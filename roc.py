from sklearn.metrics import roc_curve, auc
from sklearn import datasets
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

def ROC_Curve(X,y):
    y = label_binarize(y, classes=[0,1,2,3,4,5,6,7,8,9])
    n_classes = 10

    X_train, X_test, y_train, y_test =\
        train_test_split(X, y, test_size=0.33, random_state=0)


    clf = OneVsRestClassifier(LinearSVC(random_state=0))
    y_score = clf.fit(X_train, y_train).decision_function(X_test)

    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Plot of a ROC curve for a specific class
    for i in range(n_classes):
        plt.figure(i)
        plt.plot(fpr[i], tpr[i], label='ROC curve (area = %0.2f)' % roc_auc[i])
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate ')
        plt.ylabel('True Positive Rate ')
        plt.title('ROC class %i'%i)
        plt.legend(loc="lower right")
        plt.show()
