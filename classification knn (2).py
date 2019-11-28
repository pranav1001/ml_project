import pandas as pd
from sklearn import neighbors, metrics
import matplotlib.pyplot as plt
import seaborn as sns
import timeit

start = timeit.default_timer()

result=pd.read_csv(r"C:\Users\ashmit\Downloads\AI and ML\AI and ML\Syllabus\dataset-project.csv",sep=',')


train_x = result.iloc[1:7336,0:7].values
train_y = result.iloc[1:7336,7].values

test_x = result.iloc[7336:10217,0:7].values
expected = result.iloc[7336:10217,7].values

n_neighbors=3 #variable for storing number of nearest neighbors

weights='distance' #or weights='distance'

#creating instance of a classifier
clf = neighbors.KNeighborsClassifier(n_neighbors, weights) 

#other argument is metric (default is 'minkowski'. There are many other possibilities including 'euclidean')

#Other useful argument is p which controls power of minkowski distance. Default is 2

clf.fit(train_x, train_y)
prediction = clf.predict(test_x) #predict using the learnt classifier

stop = timeit.default_timer()

print('Time: ', stop - start)

print("############### Predictions #################")
print(prediction)
print("#############################################")

print("Accuracy={0}".format(metrics.accuracy_score(expected, prediction)))

#print(metrics.classification_report(expected, prediction))

print(metrics.confusion_matrix(expected, prediction))


def print_confusion_matrix(confusion_matrix, class_names, figsize = (10,7), fontsize=14):  
    df_cm = pd.DataFrame(
        confusion_matrix, index=class_names, columns=class_names, 
    )
    fig = plt.figure(figsize=figsize)
    try:
        heatmap = sns.heatmap(df_cm, annot=True, fmt="d")
    except ValueError:
        raise ValueError("Confusion matrix values must be integers.")
    heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=fontsize)
    heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=0, ha='right', fontsize=fontsize)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    return fig
print_confusion_matrix(metrics.confusion_matrix(expected, prediction),[1,2,3,4,5,6,7,8,9,10])