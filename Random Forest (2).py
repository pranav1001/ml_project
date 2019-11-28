import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn import metrics
import timeit
import matplotlib.pyplot as plt
import seaborn as sns


start = timeit.default_timer()


result=pd.read_csv(r"C:\Users\ashmit\Downloads\AI and ML\AI and ML\Syllabus\dataset-project.csv",sep=',')


clf = RandomForestClassifier(n_estimators=100)

# Train on the first 10000 images:
train_x = result.iloc[1:7336,0:7].values
train_y = result.iloc[1:7336,7].values

print("Train model")
clf.fit(train_x, train_y)


# Test on the next 1000 images:
test_x = result.iloc[7336:10217,0:7].values
expected = result.iloc[7336:10217,7].values
expected=expected.tolist()
print("Compute predictions")
predicted = clf.predict(test_x)


stop = timeit.default_timer()

print('Time: ', stop - start)  

print("Accuracy: ", accuracy_score(expected, predicted))
print(metrics.confusion_matrix(expected, predicted))   


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
print_confusion_matrix(metrics.confusion_matrix(expected, predicted),[1,2,3,4,5,6,7,8,9,10])

