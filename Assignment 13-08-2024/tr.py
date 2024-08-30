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
from sklearn.datasets import load_breast_cancer

st.set_page_config(page_title="Breast Cancer Data Analysis", page_icon="️:hospital:", layout="wide")
st.title("🎗️ Breast Cancer Data Analysis 🎗️")


data = load_breast_cancer()
cancer = pd.DataFrame(data.data, columns=data.feature_names)
cancer['diagnosis'] = data.target

st.header('BREAST CANCER DATA')
st.table(cancer.head())

cancer['Label'] = cancer['diagnosis']
cancer.drop(columns=['diagnosis'], axis=1, inplace=True)
st.header('Encoded BREAST CANCER DATA')
st.table(cancer.head())

x = cancer.drop(columns=['Label'])
y = cancer[['Label']]


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


pickle.dump(linearsvm, open('breast_svcm1.pkl', 'wb'))


ypred = linearsvm.predict(xtest)

st.header("Confusion Matrix of the Model")
cm = mat.confusion_matrix(ytest, ypred)
disp = px.imshow(cm, text_auto=True, labels=dict(x='Predicted Values', y='Actual Values'), x=[0, 1], y=[0, 1])
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


pickle.dump(poly_svc, open('breast_polysvc.pkl', 'wb'))


poly_kernel_svm_clf = Pipeline([
    ("scaler", StandardScaler()),
    ("svm_clf", SVC(kernel="poly", degree=3, coef0=1, C=5))
])
poly_kernel_svm_clf.fit(xtrain, ytrain)
ypred2 = poly_kernel_svm_clf.predict(xtest)

pickle.dump(poly_kernel_svm_clf, open('breast_polykernel.pkl', 'wb'))


rbf_kernel_svm_clf = Pipeline([
    ("scaler", StandardScaler()),
    ("svm_clf", SVC(kernel="rbf", gamma=5, C=0.001))
])
rbf_kernel_svm_clf.fit(xtrain, ytrain)
ypred3 = rbf_kernel_svm_clf.predict(xtest)


pickle.dump(rbf_kernel_svm_clf, open('breast_rbfkernel.pkl', 'wb'))

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


pickle.dump(rbfclf, open('breast_rbfc1f.pkl', 'wb'))

st.subheader("Accuracy of New RBF Classifier")
st.subheader(round(mat.accuracy_score(ytest, ypred4) * 100, 2))