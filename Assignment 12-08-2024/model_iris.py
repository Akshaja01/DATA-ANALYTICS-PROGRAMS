st.header("Accuracy")
c=mat.classification_report(ytest,ypred,output_dict=True)
st.subheader(round(c['accuracy']*100,2))

poly_svc=Pipeline(("poly_features",PolynomialFeatures(degree=3)),("scaler",StandardScaler()),("svm_clf",LinearSVC(c=10,loss="hinge")))
poly_svc.fit(xtrain,ytrain)
ypred1=poly_sv.predict(xtest)

c1=mat.classification_report(ytest,ypred1,output_dict=True)
m2=pickle.dump(poly_svc,open('polysvc.pkl','wb'))

poly_kernel_svm_clf=Pipeline((("Scaler"=StandardScaler()),(("svm_clf",svc((kernel="poly",degree=3,coef0=1,c=5))))

poly_kernel_svm_clf.fit(xtrain,ytrain)

ypred2=poly_kernel_svm_clf.predict(xtest)

c2=mat.classification_report(ytest,ypred2,output_dict=True)
m3=pickle.dump(poly_kernel_svm_clf,open('polykernel.pkl','wb'))

rbf_kernel_svm_clf=Pipeline(("Scaler",StandardScaler()),("svm_clf",svc(kernel="rbf",gamma=5,c=0.001))))
rbf_kernel_svm_clf.fit(xtrain,ytrain)
rbf_kernel_svm_clf.fit(xtrain,ytrain)

ypred3=rbf_kernel_svm_clf.predict(xtest)
c3==mat.classification_report(ytest,ypred3,output_dict=True)
m4=pickle.dump(rbf_kernel_svm_clf,open('rbfkernel.pkl','wb'))

c7,c8,c9=st.columns(3)
c7.subheader("Accuracy of polynomial feature model1")
c7.subheader(round(c1.['accuracy']*100,2))

c8.subheader("Accuracy of polynomial kernel model1")
c8.subheader(round(c2.['accuracy']*100,2))

c9.subheader("Accuracy of rbf kernel model1")
c9.subheader(round(c3.['accuracy']*100,2))

params={'c':[0,1,1,10,100,100],'gamma':[1,0,1,0.001,0.0001].'kernel':['rbf']}
rbfclf=GridSearchCV(estimator=svc(),param_grid=params,cv=5,n_jobs=5,verhouse=0)

rbfc1f.fit(xtrain,ytrain)
ypred4=rbfclf.predict(xtest)
c4=mat.classification_report(ytest,ypred4,output_dict=True)
m5=pickle.dump(rbfclf,open('rbfc1f.pkl','wb'))

st.subheader("Accuracy of new rbf classifier")
st.subheader(c4['accuracy']*100,2))
