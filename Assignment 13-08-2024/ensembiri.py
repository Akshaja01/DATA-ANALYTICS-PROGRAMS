import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn import metrics as mat
import seaborn as sns
import plotly.express as px
from sklearn.model_selection import train_test_split as tts
from sklearn.svm import LinearSVC, SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import VotingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC


st.set_page_config(page_title="Iris Data Analysis", page_icon="🌷", layout="wide")
st.title("🌹 Iris Data Analysis 🌹")

iris = pd.read_csv('iris.csv')
st.header('IRIS DATA')
st.table(iris.head())

le = LabelEncoder()
iris['Label'] = le.fit_transform(iris['Species'])
iris.drop(columns=['Id'], axis=1, inplace=True)
st.header('Encoded IRIS DATA')
st.table(iris.head())


x = iris.drop(columns=['Species', 'Label'])
y = iris[['Label']]


c1, c2 = st.columns(2)
c1.subheader("Features set")
c1.table(x.head())

c2.subheader("Labels")
c2.table(y.head())

xtrain, xtest, ytrain, ytest = tts(x, y, test_size=0.2, random_state=10, shuffle=True, stratify=y)


c3, c4, c5, c6 = st.columns(4)

c3.subheader("Training features")
c3.table(xtrain.head())

c4.subheader("Training labels")
c4.table(ytrain.head())

c5.subheader("Testing features")
c5.table(xtest.head())

c6.subheader("Testing labels")
c6.table(ytest.head())


linearsvm = LinearSVC()
linearsvm.fit(xtrain, ytrain)


pickle.dump(linearsvm, open('svcm1.pkl', 'wb'))


ypred = linearsvm.predict(xtest)


st.header("Confusion Matrix of the Model")
cm = mat.confusion_matrix(ytest, ypred)
disp = px.imshow(cm, text_auto=True, labels=dict(x='Predicted Values', y='Actual Values'), x=[0, 1, 2], y=[0, 1, 2])
st.plotly_chart(disp, container_width=True)


st.header("Classification Report")
report = mat.classification_report(ytest, ypred, output_dict=True)
st.write(report)


st.header("Accuracy")
st.subheader(round(report['accuracy'] * 100, 2))


poly_svc = Pipeline([
    ("poly_features", PolynomialFeatures(degree=3)),
    ("scaler", StandardScaler()),
    ("svm_clf", LinearSVC(C=10, loss="hinge"))
])
poly_svc.fit(xtrain, ytrain)
ypred1 = poly_svc.predict(xtest)


pickle.dump(poly_svc, open('polysvc.pkl', 'wb'))

poly_kernel_svm_clf = Pipeline([
    ("scaler", StandardScaler()),
    ("svm_clf", SVC(kernel="poly", degree=3, coef0=1, C=5))
])
poly_kernel_svm_clf.fit(xtrain, ytrain)
ypred2 = poly_kernel_svm_clf.predict(xtest)


pickle.dump(poly_kernel_svm_clf, open('polykernel.pkl', 'wb'))

rbf_kernel_svm_clf = Pipeline([
    ("scaler", StandardScaler()),
    ("svm_clf", SVC(kernel="rbf", gamma=5, C=0.001))
])
rbf_kernel_svm_clf.fit(xtrain, ytrain)
ypred3 = rbf_kernel_svm_clf.predict(xtest)


pickle.dump(rbf_kernel_svm_clf, open('rbfkernel.pkl', 'wb'))


c7, c8, c9 = st.columns(3)
c7.subheader("Accuracy of Polynomial Feature Model")
c7.subheader(round(mat.accuracy_score(ytest, ypred1) * 100, 2))

c8.subheader("Accuracy of Polynomial Kernel Model")
c8.subheader(round(mat.accuracy_score(ytest, ypred2) * 100, 2))

c9.subheader("Accuracy of RBF Kernel Model")
c9.subheader(round(mat.accuracy_score(ytest, ypred3) * 100, 2))


params = {'C': [0.1, 1, 10, 100], 'gamma': [1, 0.1, 0.01, 0.001], 'kernel': ['rbf']}
rbfclf = GridSearchCV(SVC(), param_grid=params, cv=5, n_jobs=-1, verbose=0)

rbfclf.fit(xtrain, ytrain)
ypred4 = rbfclf.predict(xtest)


pickle.dump(rbfclf, open('rbfc1f.pkl', 'wb'))

st.subheader("Accuracy of New RBF Classifier")
st.subheader(round(mat.accuracy_score(ytest, ypred4) * 100, 2))


gnb1 = GaussianNB()
lr1 = LogisticRegression()
svm1 = SVC(C=5, probability=True) 


voting_clf_hard = VotingClassifier(estimators=[('lr', lr1), ('gnb', gnb1), ('svc', svm1)], voting='hard')
voting_clf_soft = VotingClassifier(estimators=[('lr', lr1), ('gnb', gnb1), ('svc', svm1)], voting='soft')

voting_clf_hard.fit(xtrain, ytrain)
voting_clf_soft.fit(xtrain, ytrain)

ypred5 = voting_clf_hard.predict(xtest)
ypred6 = voting_clf_soft.predict(xtest)


c5 = mat.classification_report(ytest, ypred5, output_dict=True)
c6 = mat.classification_report(ytest, ypred6, output_dict=True)


pickle.dump(voting_clf_hard, open('voteclf1.pkl', 'wb'))
pickle.dump(voting_clf_soft, open('voteclf2.pkl', 'wb'))

cl1, cl2 = st.columns(2)

cl1.subheader('Accuracy of Hard Voting')
cl1.subheader(round(c5['accuracy'] * 100, 2))
for clf in (lr1, gnb1, svm1):
    clf.fit(xtrain, ytrain)
    yb2 = clf.predict(xtest)
    cb2 = mat.classification_report(ytest, yb2, output_dict=True)
    cl1.write(round(cb2['accuracy'] * 100, 2))

cl2.subheader('Accuracy of Soft Voting')
cl2.subheader(round(c6['accuracy'] * 100, 2))
for clf in (lr1, gnb1, svm1):
    clf.fit(xtrain, ytrain)
    yb3 = clf.predict(xtest)
    cb3 = mat.classification_report(ytest, yb3, output_dict=True)
    cl2.write(round(cb3['accuracy'] * 100, 2))