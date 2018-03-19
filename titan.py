from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.learning_curve import learning_curve
from sklearn import linear_model
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier,BaggingRegressor
from sklearn import cross_validation

from keras.models import Sequential
from keras.layers import Dense,Activation
from keras.optimizers import SGD

def set_missing_ages(df):
	age_df = df[['Age','Fare','Parch','SibSp','Pclass']]
	known_age = age_df[pd.notna(age_df['Age'])]
	unknown_age = age_df[pd.isna(age_df['Age'])]
	y = np.array(known_age['Age'])
	x = np.mat(known_age)[:,1:]

	rfr = RandomForestRegressor(random_state=0,n_estimators=2000,n_jobs=-1)
	rfr.fit(x,y)

	predict_ages = rfr.predict(np.mat(unknown_age)[:,1:])
	print(predict_ages)

	df.loc[pd.isna(df['Age']),'Age'] = predict_ages
	return df


def logistic(df):
	train_df = df[df.columns[1:]]
	train_mat = np.mat(train_df)
	y = np.array(df['Survived'])
	lr = linear_model.LogisticRegression(C=1.0,penalty='l1',tol=1e-6)
	lr.fit(train_mat,y)
	return lr 

def lr_predict(test,lr):
	test_mat = np.mat(test)
	prediction = lr.predict(test_mat)
	return prediction

def cv(df,lr):
	train_df = df[df.columns[1:]]
	train_mat = np.mat(train_df)
	y = np.array(df['Survived'])
	print(cross_validation.cross_val_score(lr,train_mat,y,cv=10))

def rfr(df):
	pass

def adaboost(df):
	train_df = df[df.columns[1:]]
	train_mat = np.mat(train_df)
	y = np.array(df['Survived'])
	a = DecisionTreeClassifier(max_depth=1,min_samples_leaf=1)
	dt = a.fit(train_mat,y)
	ad = AdaBoostClassifier(base_estimator=dt,n_estimators=300)
	ada = ad.fit(train_mat,y)
	return ada

def ada_predict(test,ada):
	test_mat = np.mat(test)
	prediction = ada.predict(test_mat)
	return prediction

def df_res(test,res):
	s = pd.DataFrame({'PassengerId':test.index,'Survived':res.astype(np.int32)})
	return s

def bagging(df,test):
	train_mat = np.mat(df[df.columns[1:]])
	y = np.array(df['Survived'])

	meta_lr = linear_model.LogisticRegression(C=1.0,penalty='l1',tol=1e-6)
	bag = BaggingRegressor(meta_lr,n_estimators=20,max_samples=0.8,max_features=1.0,bootstrap=True,bootstrap_features=False,n_jobs=-1)
	bag_fit = bag.fit(train_mat,y)
	prediction = bag_fit.predict(test)
	end = df_res(test,prediction)
	return bag_fit

def get_diff(res_old,res_new):
	df = pd.DataFrame({'old':res_old['Survived'],'new':res_new['Survived']})
	print(np.shape(df[df['old']!=df['new']]))



def plot_learn_curve(model,train,label,ylim=None,cv=None,n_jobs=1,train_sizes=np.linspace(.05,1.,20),title='learning_curve',verbose=0,plot=True):
	train_sizes,train_scores,test_scores = learning_curve(model,train,label,cv=cv,n_jobs=n_jobs,train_sizes=train_sizes,verbose=verbose)
	train_scores_mean = np.mean(train_scores,axis=1)
	train_scores_std = np.std(train_scores,axis=1)
	test_scores_mean = np.mean(test_scores,axis=1)
	test_scores_std = np.std(test_scores,axis=1)
	
	if plot:
		plt.figure()
		plt.title(title)
		if ylim is not None:
			plt.ylim(*ylim)
		plt.xlabel('data size')
		plt.ylabel('score')
		plt.gca().invert_yaxis()
		plt.grid()

		plt.fill_between(train_sizes,train_scores_mean - train_scores_std,train_scores_mean+train_scores_std,alpha=0.1,color="b")
		plt.fill_between(train_sizes,test_scores_mean - test_scores_std,test_scores_mean+test_scores_std,alpha=0.1,color="r")
		plt.plot(train_sizes,train_scores_mean,'o-',color='b',label='train set score')
		plt.plot(train_sizes,test_scores_mean,'o-',color='r',label='cv set score')

		plt.legend(loc='best')

		plt.draw()
		plt.show()

		midpoint = ((train_scores_mean[-1]+train_scores_std[-1])+(test_scores_mean[-1]-test_scores_std[-1]))/2
		diff = (train_scores_mean[-1]+test_scores_std[-1]) - (test_scores_mean[-1]-test_scores_std[-1])
		return midpoint,diff

#用神经网络试一试
def init_nn():
	model = Sequential()

	model.add(Dense(input_dim=7,units=20,kernel_initializer='he_normal'))
	model.add(Activation('sigmoid'))

	model.add(Dense(units=30,kernel_initializer='he_normal'))
	model.add(Activation('sigmoid'))

	model.add(Dense(units=2))
	model.add(Activation('softmax'))

	return model

def fit_model(model,x_train,y_train):
	model.compile(optimizer=SGD(lr=0.1),loss='mse',metrics=['accuracy']) #adam优化方法
	model.fit(x_train,y_train,epochs=60,validation_split=0.05)  #取训练集的5%作为验证集
	return model
