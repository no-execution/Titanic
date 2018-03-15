import pandas as pd
import numpy as np
import sklearn.preprocessing as preprocessing
from sklearn.ensemble import RandomForestRegressor

def prep(train,test):
	#建立一个总的df用来操作。
	df = pd.concat([train,test],axis=0)
	df.index = df['PassengerId']
	df = df.drop('PassengerId',axis=1)

	#修改test数据集的索引
	#测试集
	test.index = test['PassengerId']
	test = test.drop('PassengerId',axis=1)
	#训练集
	train.index = train['PassengerId']
	train = train.drop('PassengerId',axis=1)

	#把Cabin给序列化
	#测试集
	test.loc[pd.notna(test['Cabin']),'Cabin'] = 1
	test.loc[pd.isna(test['Cabin']),'Cabin'] = 0
	test = test.drop('Cabin',axis=1)
	#训练集
	train.loc[pd.notna(train['Cabin']),'Cabin'] = 1
	train.loc[pd.isna(train['Cabin']),'Cabin'] = 0
	train = train.drop('Cabin',axis=1)

	#把Fare缺失的部分填上平均值
	#该部分''训练集''可以不用
	#测试集
	test.loc[pd.isna(test['Fare']),'Fare'] = np.mean(df['Fare'])
	#训练集
	train.loc[pd.isna(train['Fare']),'Fare'] = np.mean(df['Fare'])

	#补齐缺失年龄
	ages = get_mean_age(df)
	#训练集
	train = fill_na_age(ages,train)
	#测试集
	test = fill_na_age(ages,test)
	df = pd.concat([train,test],axis=0)
	#将年龄分段，先分为5段
	age_band = pd.cut(train['Age'],5)
	train['Age_Range'],test['Age_Range'] = 0,0
	#然后将不同的段分类
	#训练集
	train.loc[train['Age']<=16,'Age_Range'] = 0
	train.loc[(train['Age']>16) & (train['Age']<=32),'Age_Range'] = 1
	train.loc[(train['Age']>32) & (train['Age']<=48),'Age_Range'] = 2
	train.loc[(train['Age']>48) & (train['Age']<=64),'Age_Range'] = 3
	train.loc[train['Age']>64,'Age_Range'] = 4
	#测试集
	test.loc[test['Age']<=16,'Age_Range'] = 0
	test.loc[(test['Age']>16) & (test['Age']<=32),'Age_Range'] = 1
	test.loc[(test['Age']>32) & (test['Age']<=48),'Age_Range'] = 2
	test.loc[(test['Age']>48) & (test['Age']<=64),'Age_Range'] = 3
	test.loc[test['Age']>64,'Age_Range'] = 4

	#性别序列化
	#训练集
	train.loc[train['Sex']=='female','Sex'] = 0
	train.loc[train['Sex']=='male','Sex'] = 1
	#测试集
	test.loc[test['Sex']=='female','Sex'] = 0
	test.loc[test['Sex']=='male','Sex'] = 1

	#登船港口序列化
	#训练集
	train['Embarked_Class'],test['Embarked_Class'] = 0,0
	train.loc[pd.isna(train['Embarked']),'Embarked'] = 'S'
	train.loc[train['Embarked']=='S','Embarked_Class'] = 0
	train.loc[train['Embarked']=='C','Embarked_Class'] = 1
	train.loc[train['Embarked']=='Q','Embarked_Class'] = 2
	#测试集
	test.loc[test['Embarked']=='S','Embarked_Class'] = 0
	test.loc[test['Embarked']=='C','Embarked_Class'] = 1
	test.loc[test['Embarked']=='Q','Embarked_Class'] = 2

	#统计姓名中的称谓，通过称谓进行分类,然后序列化
	#训练集
	train_titles = num_word(train)
	train = title_class(train,train_titles)
	train.loc[train['Title']=='Mr','Title'] = 0
	train.loc[train['Title']=='Miss','Title'] = 1
	train.loc[train['Title']=='Mrs','Title'] = 2
	train.loc[train['Title']=='Master','Title'] = 3
	train.loc[train['Title']=='Rare','Title'] = 4
	#测试集
	test_titles = num_word(test)
	test = title_class(test,test_titles)
	test.loc[test['Title']=='Mr','Title'] = 0
	test.loc[test['Title']=='Miss','Title'] = 1
	test.loc[test['Title']=='Mrs','Title'] = 2
	test.loc[test['Title']=='Master','Title'] = 3
	test.loc[test['Title']=='Rare','Title'] = 4

	#把Fare分段，分成4段
	#df['Fare_Range'] = pd.qcut(df['Fare'],4)
	
	df.loc[df['Fare']<=7.9,'Fare_Range'] = 0
	df.loc[(df['Fare']>7.9) & (df['Fare']<=14.5),'Fare_Range'] = 1
	df.loc[(df['Fare']>14.5) & (df['Fare']<=31.3),'Fare_Range'] = 2
	df.loc[df['Fare']>31.3,'Fare_Range'] = 3
	#print(df[['Fare_Range','Survived']].groupby(['Fare_Range'],as_index=False).mean().sort_values(by='Fare_Range',ascending=True))

	train['Fare_Range'] = df[df.index<=891]['Fare_Range'].astype(np.int32)
	test['Fare_Range'] = df[df.index>891]['Fare_Range'].astype(np.int32)

	return train,test




def get_mean_age(df):
	mid = df[pd.notna(df['Age'])]
	n = len(mid)
	miss = []
	mrs = []
	mr = []
	master = []
	dr = []
	for i in range(n):
		if ('Miss.' in mid['Name'].iloc[i]) or ('Ms.' in mid['Name'].iloc[i]):
			miss.append(i)
		elif 'Mrs.' in mid['Name'].iloc[i]:
			mrs.append(i)
		elif 'Mr.' in mid['Name'].iloc[i]:
			mr.append(i)
		elif 'Master.' in mid['Name'].iloc[i]:
			master.append(i)
		elif 'Dr.' in mid['Name'].iloc[i]:
			dr.append(i)
		else:
			continue
	print(len(miss+mrs+mr+master+dr))
	#取各个分块的平均年龄
	res = {'Miss':np.mean(df.iloc[miss]['Age']),\
		'Mrs':np.mean(df.iloc[mrs]['Age']),\
		'Mr':np.mean(df.iloc[mr]['Age']),\
		'Master':np.mean(df.iloc[master]['Age']),\
		'Dr':np.mean(df.iloc[dr]['Age'])}
	return res

#为缺失年龄的数据填充年龄
def fill_na_age(ages,df):
	n = len(df)
	miss = []
	mrs = []
	mr = []
	master = []
	dr = []
	for i in range(n):
		if pd.isna(df['Age'].iloc[i]):
			if ('Miss.' in df['Name'].iloc[i]) or ('Ms.' in df['Name'].iloc[i]):
				miss.append(i)
			elif 'Mrs.' in df['Name'].iloc[i]:
				mrs.append(i)
			elif 'Mr.' in df['Name'].iloc[i]:
				mr.append(i)
			elif 'Master.' in df['Name'].iloc[i]:
				master.append(i)
			elif 'Dr.' in df['Name'].iloc[i]:
				dr.append(i)
			else:
				continue
		else:
			continue
	print(miss,mrs,mr,master,dr)
	df.loc[miss+df.index[0],'Age'] = ages['Miss']
	df.loc[mrs+df.index[0],'Age'] = ages['Mrs']
	df.loc[mr+df.index[0],'Age'] = ages['Mr']
	df.loc[master+df.index[0],'Age'] = ages['Master']
	df.loc[dr+df.index[0],'Age'] = ages['Dr']
	return df


def set_missing_ages(df):
	age_df = df[['Age','Fare','Parch','SibSp','Pclass']]
	known_age = age_df[pd.notna(age_df['Age'])]
	unknown_age = age_df[pd.isna(age_df['Age'])]
	y = np.array(known_age['Age'])
	x = np.mat(known_age)[:,1:]

	rfr = RandomForestRegressor(random_state=0,n_estimators=2000,n_jobs=-1)
	rfr.fit(x,y)
	predict_ages = rfr.predict(np.mat(unknown_age)[:,1:])
	return predict_ages

#用正则表达式将人名当中的称谓提取出来
def num_word(df):
	words = df['Name']
	titles = words.str.extract('([A-Za-z]+)\.',expand=False)
	return titles

#把称谓归类
def title_class(df,titles):
	df['Title'] = titles
	df['Title'] = df['Title'].replace([\
			'Capt',\
			'Col',\
			'Countess',\
			'Don',\
			'Dr',\
			'Jonkheer',\
			'Lady',\
			'Major',\
			'Rev',\
			'Sir',\
			'Dona'],'Rare')
	df['Title'] = df['Title'].replace(['Mlle','Ms'],'Miss')
	df['Title'] = df['Title'].replace('Mme','Mrs')
	return df