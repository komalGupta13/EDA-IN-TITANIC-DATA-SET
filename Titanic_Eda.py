import pandas as pd
import matplotlib.pyplot as mp
import seaborn as se

df=pd.read_csv("https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv")

# anaysis the data
print(df.info())
print(df.describe())
#see the first 5 datasets and last 5 dataset
print(df.head())
print(df.tail())


#Handle the missing Dataset
df["Age"]=df["Age"].fillna(df["Age"].median())
df["Embarked"]=df["Embarked"].fillna(df["Embarked"].mode()[0])
print(df.head())
print(df.tail())

#Remove the Duplicates
df=df.drop_duplicates()

#Filter passenger : First class
First_class=df[df["Pclass"]==1]
print(First_class.head())
print(First_class.tail())

#Visulization of servival rate according to the Pclass
Servival_by_class=df.groupby("Pclass")["Survived"].mean()
mp.bar(Servival_by_class.index,Servival_by_class.values, color="Skyblue" )
mp.title("Servival_rate")

mp.xlabel("Pclass")
mp.ylabel("Servived")
mp.show()

#Histogram for Age distribution
se.histplot(df["Age"],kde="True",bins=30,color="purple",edgecolor="black")
mp.title("Age_Distribution")
mp.xlabel("Age")
mp.ylabel("Frequnecy")


mp.show()

# Scatter plot in Age vs Fare
mp.scatter(df["Age"],df["Fare"], alpha=0.5,color="red",)

mp.title("Age_vs_fear")
mp.xlabel("AGE")
mp.ylabel("FARE")
mp.legend()

mp.show()