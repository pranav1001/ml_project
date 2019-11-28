import matplotlib
import pandas as pd

result=pd.read_csv(r"C:\Users\ashmit\Desktop\accuracies.csv",sep=',',encoding="ISO-8859-1")
accuracy=result.iloc[1:7,1].values
time=result.iloc[1:7,2].values
print(accuracy)
print(time)
matplotlib.pyplot.scatter(time,accuracy,c="red")