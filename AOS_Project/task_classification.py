import pandas as pd
import numpy as np
from numpy.random.mtrand import randint
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
nltk.download('stopwords')
import pickle
from nltk.corpus import stopwords
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.decomposition import PCA
from sklearn.metrics import average_precision_score
import time
#reference https://stackabuse.com/implementing-pca-in-python-with-scikit-learn/
def read_csv(path):
    return pd.read_csv(path)

def model(df):
    # tfidfconverter = TfidfVectorizer(max_features=1, min_df=1, max_df=0.7, stop_words=stopwords.words('english'))
    # job_id = tfidfconverter.fit_transform(df["job_id"]).toarray()
    df.drop(["job_id"],inplace=True,axis=1)
      # [randint(1, 4) for i in range(0,df["file_size"].count())]
    df.drop(["operation"],inplace=True,axis=1)
    where_are_NaNs = np.isnan(df)
    df[where_are_NaNs] = 592
    df2= pd.DataFrame();
    df2["file_size"]=[randint(500, 1000) for i in range(0,80)]
    df2["output_size"]=[randint(5, 10000) for i in range(0,80)]
    df2["file_size"]=[randint(400,1300) for i in range(0,80)]

    df.append(df2,ignore_index=True)


    pso_optimization(df)
    y = df["output"]
    df.drop(["output"],inplace=True,axis=1)

    X =df;

    names = ["Nearest Neighbors", "Linear SVM", "RBF SVM","Gaussian Process",
             "Decision Tree", "Random Forest", "Neural Net", "AdaBoost",
             "Naive Bayes"]
    classifiers = [
        KNeighborsClassifier(3),
        SVC(kernel="linear", C=0.025),
        SVC(gamma=2, C=1),
        GaussianProcessClassifier(1.0 * RBF(1.0)),

        DecisionTreeClassifier(max_depth=5),
        RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
        MLPClassifier(alpha=1, max_iter=1000),
        AdaBoostClassifier(),
        GaussianNB()

        ]

    # Splitting the dataset into the Training set and Test set

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    print("##################Without PCA####################")
    # iterate over classifiers
    score=[]
    time_withoutPCA = []
    for name, clf in zip(names, classifiers):
        start = time.clock()
        clf.fit(X_train, y_train)
        y_score=clf.score(X_test, y_test)
        score.append(y_score)
        time_withoutPCA.append(time.clock() - start)
        # print("time_withoutPCA: ", time_withoutPCA)
        print(name,y_score)

    print("time_withoutPCA: ", time_withoutPCA)

    plot_bar_x(names, score, titlle="Various Algorithm without PCA")
    plot_bar_x_execution_time(names, time_withoutPCA, titlle="Excution time without PCA")

    print("#################WITH PCA#########################")

    #########PCA 1#############
    # pca = PCA()
    # X_train = pca.fit_transform(X_train)
    # X_test = pca.transform(X_test)
    ##############PCA 2###########
    pca = PCA(n_components=3)
    X_train = pca.fit_transform(X_train)
    X_test = pca.transform(X_test)
    score_withPCA = []
    time_withPCA=[]
    for name, clf in zip(names, classifiers):
        start=time.clock()
        clf.fit(X_train, y_train)
        y_hat= clf.predict(X_test)
        y_score_pca=clf.score(X_test, y_test)
        # plt.plot(list(y_hat),'ro',linewidth=7.0)
        # plt.plot(list(y_test), 'g^', linewidth=7.0)
        # plt.show()
        score_withPCA.append(y_score_pca)
        print(name,y_score_pca)
        time_withPCA.append(time.clock() - start)
    print("time_withPCA: ", time_withPCA)

    plot_bar_x(names, score_withPCA, titlle="Various Algorithm with PCA")
    plot_bar_x_execution_time(names, time_withPCA, titlle="Excution time with PCA")
    # classifier = RandomForestClassifier(max_depth=2, random_state=0)
    # classifier.fit(X_train, y_train)
    #
    # # Predicting the Test set results
    # y_pred = classifier.predict(X_test)
    #
    #
    # cm = confusion_matrix(y_test, y_pred)
    # print(cm)
    # print('Accuracy :' , accuracy_score(y_test, y_pred))
count=0;
def plot_bar_x(label,no_movies,titlle):
    # this is for plotting purpose
    index = np.arange(len(label))
    plt.bar(index, no_movies,color=['#D2691E', 'red', 'green', 'blue', 'cyan',"#D2691E","#DC143C","#008B8B","#A9A9A9","#483D8B"])
    plt.xlabel('model', fontsize=5)
    plt.ylabel('accuracy', fontsize=10)
    plt.xticks(index, label, fontsize=10, rotation=30)
    plt.title(titlle)
    plt.savefig(titlle+".png")
    plt.show()

def plot_bar_x_execution_time(label, no_movies, titlle):
    # this is for plotting purpose
    index = np.arange(len(label))
    plt.bar(index, no_movies,
            color=['#D2691E', 'red', 'green', 'blue', 'cyan', "#D2691E", "#DC143C", "#008B8B", "#A9A9A9",
                   "#483D8B"])
    plt.xlabel('model', fontsize=5)
    plt.ylabel('execution time', fontsize=10)
    plt.xticks(index, label, fontsize=10, rotation=30)
    plt.title(titlle)
    plt.savefig(titlle + ".png")
    plt.show()

def pso_optimization(df):
    #reference https://www.geeksforgeeks.org/multidimensional-data-analysis-in-python/
    from sklearn.cluster import KMeans
    clusters = 3
    kmeans = KMeans(n_clusters=clusters)
    kmeans.fit(df)
    df["output"]=kmeans.labels_;
    print(kmeans.labels_)
if __name__ == '__main__':
    df=read_csv("data_set/hadoop_task_logs.csv")
    # pso_optimization(df)
    model(df)

    # print(df)
