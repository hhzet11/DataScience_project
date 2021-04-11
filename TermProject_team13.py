#!/usr/bin/env python
# coding: utf-8

# In[6]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder

data=pd.read_csv("vehicles (1).csv", encoding='utf-8', index_col=0)
print(data.columns)
#drop columns that is not used to precidt condition
data=data.drop(['id', 'url','region', 'region_url', 'model','fuel', 
                'title_status','transmission', 'vin', 'drive', 'size', 'type', 
                'paint_color', 'image_url', 'description', 'country', 
                'state', 'lat', 'long'],axis=1)
print(data.columns)
#drop missing values in condition coulmn
data.dropna(subset=['condition'], how='all', inplace = True)

#Get statistics on the dataset
print("\n-----Description of dataset-----")
print(data.describe())

# print (data['manufacturer'].unique())
# print (data['cylinders'].unique())
# print (data['condition'].unique())

#(categorical features)manufacturer, cylinders, conditions are all not dirty

data['year'].replace((0, np.NaN), inplace=True)

###########price boxplot###########
data['price']=data['price']/1000
# data = data[data['price']<300]
data['price'].replace((0, np.NaN), inplace=True)

sns.boxplot(x="condition", y="price", data=data)
plt.title("Box Plot_Before Remove")
plt.ylabel("price(thousand dollors)")
plt.show()

#IQR 
for i in data['condition'].unique():   
    fraud_column_data=data[data['condition']==i]['price']
    quan_25=np.percentile(fraud_column_data.values,25)
    quan_75=np.percentile(fraud_column_data.values,75)
    iqr=quan_75 - quan_25
    iqr=iqr*1.8
    lowest=quan_25
    highest=quan_75+iqr
    outlier_index=fraud_column_data[(fraud_column_data<lowest)|(fraud_column_data>highest)].index
    #이상치 row 모두 삭제(이상치 비율이 너무 많아서)
    data.drop(outlier_index,axis=0, inplace=True)
    
#이상치 제거후 박스 플롯
sns.boxplot(x="condition", y="price", data=data)
plt.ylabel("price(thousand dollors)")
plt.title("Box Plot_After Remove")
plt.show()

#Get statistics on the dataset
print("\n-----Description of dataset-----")
data.info()

###########odometer boxplot###########
data['odometer'].replace((0, np.NaN), inplace=True)

sns.boxplot(x="condition", y="odometer", data=data)
plt.title("Box Plot_Before Remove")
plt.show()

# IQR 
for i in data['condition'].unique():   
    fraud_column_data_o=data[data['condition']==i]['odometer']
    quan_25_o=np.percentile(fraud_column_data_o.values,25)
    quan_75_o=np.percentile(fraud_column_data_o.values,75)
    iqr=quan_75_o - quan_25_o
    iqr=iqr*1.5
    lowest=quan_25_o
    highest=quan_75_o+iqr
    outlier_index_o=fraud_column_data_o[(fraud_column_data_o<lowest)|(fraud_column_data_o>highest)].index
    data.drop(outlier_index_o,axis=0, inplace=True)

#이상치 제거후 박스 플롯
sns.boxplot(x="condition", y="odometer", data=data)
plt.title("Box Plot_After Remove")
plt.show()

#Get statistics on the dataset
print("\n-----Description of dataset-----")
data.info()

#If there has missing for space, change it to NaN
data.replace({"": np.nan}, inplace=True)

#Get statistics on Missing values
print("\n-----Description of Missing values [Before]-----")
print(data.isna().sum())

######### Fill Missing values - Cylinder
#cylinders에서 숫자만 분류해 저장 -> float변환
data['cylinders'] = data.cylinders.str.split(' ').str[0]
data['cylinders'] = pd.to_numeric(data['cylinders'],errors = 'coerce')

#cylinders의 datatype 변화 확인 object -> float
print("\nTo check the type change")
data.info()

#condition별 평균 구해서 fill
mean=data.groupby("condition")["cylinders"].mean()
print("\nMean for each condition")
print(mean)
data["cylinders"].fillna(value = round(mean),inplace=True)

######### Fill Missing values - Manufacturer
#calculate mode value
most_freq=data['manufacturer'].value_counts(dropna=True).idxmax()
data['manufacturer'].fillna(most_freq,limit=2,inplace=True)

#After limit=2 -> drop left NaN
data.dropna(axis=0,inplace=True)

print("\n-----Description of Missing values [After]-----")
print(data.isna().sum())

print("\nTo check the value type ")
print(data.info())
data_org1 = data.copy()
data_org = data.copy()

#min-max scalering
scaler = MinMaxScaler()
scaler.fit(data[["price","year","cylinders","odometer"]])
data_norm = scaler.transform(data[["price","year","cylinders", "odometer"]])

print("\nTo check the normalized data")
print(data_norm)

#data['manufacturer'] : string -> int
labelEncoder = LabelEncoder()
labelEncoder.fit(data['manufacturer'])
data['manufacturer'] = labelEncoder.transform(data['manufacturer'])

print("\nTo check the label encodered data['manufacturer']")
print(data['manufacturer'])

#data['condition'] : string -> int
labelEncoder = LabelEncoder()
labelEncoder.fit(data['condition'])
data['condition'] = labelEncoder.transform(data['condition'])

print("\nTo check the label encodered data['condition']")
print(data['condition'])

# Apply changed data to DataFrame
data[["price","year","cylinders","odometer"]] = data_norm
print(data)


#################################modeling-MLR#################################
from IPython.display import display
#Calculate the correlation matrix
corr=data.corr()
display(corr)
sns.heatmap(corr,xticklabels=corr.columns,yticklabels=corr.columns,cmap='RdBu')
plt.show()

#split train & test set
from sklearn.model_selection import train_test_split
x=data.drop(columns=['condition'])
y=data['condition'].values
x_train,x_test,y_train,y_test = train_test_split(x,y,train_size=0.8,random_state=1)

#Create model
from sklearn.linear_model import LinearRegression
mlr = LinearRegression()
mlr.fit(x_train,y_train)
y_predict=mlr.predict(x_test)

#create OLS model and fit data
import statsmodels.api as sm
x2=sm.add_constant(x)
model=sm.OLS(y,x2)
est=model.fit()

#Check for the normality of the residuals
import pylab
sm.qqplot(est.resid,line='s')
pylab.show()

#Evaluate Accuracy
from sklearn import metrics
print('\nMultiple Linear Regression Evaluation')
print('Root Mean Squared Error= {:.3f}'.format(np.sqrt(metrics.mean_squared_error(y_test,y_predict))))
print("Mean Absolute Error:{:.3f}".format(metrics.mean_absolute_error(y_test,y_predict)))
print("Mean Squared Error:{:.3f}".format(metrics.mean_squared_error(y_test,y_predict)))

#print out the summary
print(est.summary())


####################5-fold cross validation about MLR########################
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
kfold = KFold(n_splits=5)
n_iter = 0 
cv_accuracy = []

X = data.drop(columns=['condition'])
X = X.to_numpy()
scaler = MinMaxScaler()
scaler.fit(X)
X = scaler.transform(X)


y = data['condition'].values
LabelEncoder = LabelEncoder()
LabelEncoder.fit(y)
y = LabelEncoder.transform(y)


for train_index, test_index in kfold.split(data):
    n_iter+=1
    label_train =  data['condition'].iloc[train_index]
    label_test = data['condition'].iloc[test_index]
    print('\n##교차 검증: {0}'.format(n_iter))
    print('학습 레이블 데이터 분포: \n', label_train.value_counts())
    print('검증 레이블 데이터 분포: \n', label_test.value_counts())
    
    #split()으로 반환된 인덱스를 이용해 학습용, 검증용 테스트 데이터 추출


    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    # 학습 및 예측
    mlr.fit(X_train, y_train)
    pred = mlr.predict(X_test)
    # 반복 시마다 정확도 측정
    accuracy=mlr.score(X_test,y_test)
#     accuracy = np.round(accuracy_score(y_test, pred),4)
    train_size = X_train.shape[0]
    test_size = X_test.shape[0]
    print('{0} 교차검증 정확도: {1}, 학습데이터 크기: {2}, 검증데이터 크기: {3}'.format(n_iter, accuracy, train_size, test_size))
    cv_accuracy.append(accuracy)

    
# 교차검증별 정확도 및 평균정확도 계산
print('\n 교차검증별 정확도: ', cv_accuracy)
print('\n 교차검증 평균 정확도: ', np.mean(cv_accuracy))


#################################modeling-KNN#################################
from sklearn.preprocessing import LabelEncoder

X = data.drop(columns=['condition'])
print(X.head())

y = data['condition'].values
print("\noriginal target data['condition']")
print(y[0:3])

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y,random_state = 20)

from sklearn.neighbors import KNeighborsClassifier

k_list = range(1,15)
train_accuracies = []
test_accuracies = []


#Compute accuracy 

for k in k_list:
    classifier = KNeighborsClassifier(n_neighbors = k)
    classifier.fit(X_train, y_train)
    prediction = classifier.predict(X_test)
    train_accuracies.append(classifier.score(X_train,y_train))
    test_accuracies.append((prediction==y_test).mean())


#Test Accuracy & Train Accuracy    
plt.plot(k_list, train_accuracies, label="TRAIN set")
plt.plot(k_list, test_accuracies, label="TEST set")
plt.xlabel("k")
plt.ylabel("Accuracy")
plt.xticks(np.arange(0,16, step=1))
plt.title("Usedcar Classifier Accuracty")

plt.show()

knn = KNeighborsClassifier(n_neighbors = 1)
knn.fit(X_train,y_train)

#Compute Prediction
print("\n(1)predicted target data")
print(knn.predict(X_test[0:3]))
print("-----Prediction Accuracy-----")
print(knn.score(X_test,y_test)) # 정확도 낮음!

#print(data_org) #original data
X = data_org.drop(columns=['condition'])

#print(X.head())

y = data_org['condition'].values

###############user input#################
#user_price = 7993
#uesr_price = user_price/1000
#user_year = 2010
#user_manu = "chevrolet"
#user_cylinder = 8
#odometer = 194050

user = np.array([[7.993,2010,'chevrolet',8,194050]]) #condition 예측을 위해 기타 정보 입력.
X = X.to_numpy()
#print(X)
#print("\noriginal data + user")
X = np.append(X,user, axis=0)
#print(X)

#User Enterd Manufacturer: String -> Float
manu = X[:,2]
#from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
encoder.fit(manu)
manu = encoder.transform(manu)
X[:,2] = manu

#print("\nmanufacurer: string -> float")
#print(X)

# 사용자가 입력한 데이터 포함해서 다시 normalization 
scaler = MinMaxScaler()
scaler.fit(X)
X = scaler.transform(X)
#print("\nMinMaxScaler resurt")
#print(X)

user = X[-1] # 사용자 입력 값(표준화 완료)
print("user",user)
X = X[0:-1]# original data(normalized)

#target data: String -> Float
LabelEncoder = LabelEncoder()
LabelEncoder.fit(y)
y = LabelEncoder.transform(y)
#print(y)

#print("\nwhole data")
X= np.insert(X,3,y,1) # 원래 형태로 만들기 위해.
#print(X)
from math import sqrt 
def euclidean_distance(row1,row2):
    distance = 0.0
    for i in range(len(row1)-1):
        distance += (row1[i]-row2[i])**2
    return sqrt(distance)

def get_neighbors(train,test_row, num_neighbors):
    distances = list()
    for train_row in train:
        dist =euclidean_distance(test_row, train_row)
        distances.append((train_row, dist))
    distances.sort(key=lambda tup:tup[1])
    neighbors = list()
    for i in range(num_neighbors):
        neighbors.append(distances[i][0])
    return neighbors

print("\n----------Neighbors----------")
neighbors = get_neighbors(X, user,3)
for neighbor in neighbors:
    print(neighbor)
    
output_values = [row[3] for row in neighbors]
prediction = max(set(output_values), key=output_values.count)
prediction = int(prediction)
print(LabelEncoder.classes_)
print("\n########## Target Prediction Result ###########")
print('\nCondition: {}.'.format(LabelEncoder.classes_[prediction]))

#############price - manufacturer############
# 전체를 돌리기에는 너무 많아서 가까운 것 100개로 표현.
neighbors = get_neighbors(X, user,100)
for c in range(len(neighbors)):
    if X[c][3] == 0: #excellent
        plt.scatter(neighbors[c][0],neighbors[c][2], color='orange')
    elif X[c][3] == 1: #fair
        plt.scatter(neighbors[c][0],neighbors[c][2], color='b')
    elif X[c][3] == 2: #good
        plt.scatter(neighbors[c][0],neighbors[c][2], color='g')
    elif X[c][3] == 3: #like new
        plt.scatter(neighbors[c][0],neighbors[c][2], color='gray')
    elif X[c][3] == 4: #new
        plt.scatter(neighbors[c][0],neighbors[c][2], color='yellow')
    elif X[c][3] == 5: #salvage
        plt.scatter(neighbors[c][0],neighbors[c][2], color="purple")
plt.title("price - manufacturer")
plt.scatter(user[0],user[2], marker = '*', color = 'red')
plt.show()

#############year-odometre############
# 전체를 돌리기에는 너무 많아서 가까운 것 100개로 표현.
neighbors = get_neighbors(X, user,100)
for c in range(len(neighbors)):
    if X[c][3] == 0: #excellent
        plt.scatter(neighbors[c][1],neighbors[c][5], color='orange')
    elif X[c][3] == 1: #fair
        plt.scatter(neighbors[c][1],neighbors[c][5], color='b')
    elif X[c][3] == 2: #good
        plt.scatter(neighbors[c][1],neighbors[c][5], color='g')
    elif X[c][3] == 3: #like new
        plt.scatter(neighbors[c][1],neighbors[c][5], color='gray')
    elif X[c][3] == 4: #new
        plt.scatter(neighbors[c][1],neighbors[c][5], color='yellow')
    elif X[c][3] == 5: #salvage
        plt.scatter(neighbors[c][1],neighbors[c][5], color='purple')
plt.title("year-odometer")
plt.scatter(user[1],user[4], marker = '*', color = 'red')
plt.show()

#############cylinder-odometre############
# 전체를 돌리기에는 너무 많아서 가까운 것 100개로 표현.
neighbors = get_neighbors(X, user,100)
for c in range(len(neighbors)):
    if X[c][3] == 0: #excellent
        plt.scatter(neighbors[c][4],neighbors[c][5], color='orange')
    elif X[c][3] == 1: #fair
        plt.scatter(neighbors[c][4],neighbors[c][5], color='b')
    elif X[c][3] == 2: #good
        plt.scatter(neighbors[c][4],neighbors[c][5], color='g')
    elif X[c][3] == 3: #like new
        plt.scatter(neighbors[c][4],neighbors[c][5], color='gray')
    elif X[c][3] == 4: #new
        plt.scatter(neighbors[c][4],neighbors[c][5], color='yellow')
    elif X[c][3] == 5: #salvage
        plt.scatter(neighbors[c][4],neighbors[c][5], color='purple')
plt.title("cylinder-odometer")
plt.scatter(user[3],user[4], marker = '*', color = 'red')
plt.show()

#############price-odometre############
# 전체를 돌리기에는 너무 많아서 가까운 것 100개로 표현.
neighbors = get_neighbors(X, user,100)
for c in range(len(neighbors)):
    if X[c][3] == 0: #excellent
        plt.scatter(neighbors[c][0],neighbors[c][5], color='orange')
    elif X[c][3] == 1: #fair
        plt.scatter(neighbors[c][0],neighbors[c][5], color='b')
    elif X[c][3] == 2: #good
        plt.scatter(neighbors[c][0],neighbors[c][5], color='g')
    elif X[c][3] == 3: #like new
        plt.scatter(neighbors[c][0],neighbors[c][5], color='gray')
    elif X[c][3] == 4: #new
        plt.scatter(neighbors[c][0],neighbors[c][5], color='yellow')
    elif X[c][3] == 5: #salvage
        plt.scatter(neighbors[c][0],neighbors[c][5], color='purple')
    
plt.title("price-odometer")
plt.scatter(user[0],user[4], marker = '*', color = 'red')
plt.show()

####################5-fold cross validation about KNN########################
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder

kfold = KFold(n_splits=5)
n_iter = 0 

X = data.drop(columns=['condition'])
X = X.to_numpy()
scaler = MinMaxScaler()
scaler.fit(X)
X = scaler.transform(X)


y = data['condition'].values
LabelEncoder = LabelEncoder()
LabelEncoder.fit(y)
y = LabelEncoder.transform(y)


cv_accuracy = []
for train_index, test_index in kfold.split(data):
    n_iter+=1
    label_train =  data['condition'].iloc[train_index]
    label_test = data['condition'].iloc[test_index]
    print('\n##교차 검증: {0}'.format(n_iter))
    print('학습 레이블 데이터 분포: \n', label_train.value_counts())
    print('검증 레이블 데이터 분포: \n', label_test.value_counts())
    
    
    #split()으로 반환된 인덱스를 이용해 학습용, 검증용 테스트 데이터 추출
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    # 학습 및 예측
    knn.fit(X_train, y_train)
    pred = knn.predict(X_test)
    # 반복 시마다 정확도 측정
    accuracy = np.round(accuracy_score(y_test, pred),4)
    train_size = X_train.shape[0]
    test_size = X_test.shape[0]
    print('{0} 교차검증 정확도: {1}, 학습데이터 크기: {2}, 검증데이터 크기: {3}'.format(n_iter, accuracy, train_size, test_size))
    cv_accuracy.append(accuracy)

    
# 교차검증별 정확도 및 평균정확도 계산
print('\n 교차검증별 정확도: ', cv_accuracy)
print('\n 교차검증 평균 정확도: ', np.mean(cv_accuracy))


###########ensemble learning using KNN model###########
from sklearn.model_selection import train_test_split
x=data.drop(['condition'],axis=1)
y=data['condition']
x_train,x_test,y_train,y_test = train_test_split(x,y,train_size=0.8,random_state=123)
from sklearn.ensemble import BaggingClassifier
ensemble=BaggingClassifier(base_estimator=knn, bootstrap_features=True,max_features=5, n_jobs=30,  random_state=777)
ensemble.fit(x_train, y_train)

print('The accuracy for bagged KNN(ensemble) is:', ensemble.score(x_test, y_test) )
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report 
prediction=ensemble.predict(x_test)
print(classification_report(y_test, prediction))

##########################input and prediction##########################

input={'price': [0.8],
            'year': [0.95],
            'manufacturer': [18],
              'cylinders': [0.333333],
              'odometer': [0.1]}

input=pd.DataFrame(input)

prediction=ensemble.predict(input)

if(prediction==0):
    print("excellent")
elif(prediction==1):
    print("fair")
elif(prediction==2):
    print("good")
elif(prediction==3):
    print("like new")
elif(prediction==4):
    print("new")
elif(prediction==4):
    print("salvage")
    

###################Predict GUI###################################
from math import sqrt 
def euclidean_distance(row1,row2):
    distance = 0.0
    for i in range(len(row1)-1):
        distance += (row1[i]-row2[i])**2
    return sqrt(distance)

def get_neighbors(train,test_row, num_neighbors):
    distances = list()
    for train_row in train:
        dist =euclidean_distance(test_row, train_row)
        distances.append((train_row, dist))
    distances.sort(key=lambda tup:tup[1])
    neighbors = list()
    for i in range(num_neighbors):
        neighbors.append(distances[i][0])
    return neighbors

#function to predict condition
def predict_multiple():
    user_price=int(entry_price.get())/1000
    user_year=entry_year.get()
    user_manu=entry_manu.get()
    user_cylinders=entry_cylinders.get()
    user_odometer=entry_odometer.get()
        
    #condition 예측을 위해 기타 정보 입력.
    user = np.array([[user_price,user_year,user_manu,user_cylinders,user_odometer]])
    X = data_org1.drop(columns=['condition'])
    X = X.to_numpy()
    print(X)
    print("\noriginal data + user")
    X = np.append(X,user, axis=0)
    print(X)

    #User Enterd Manufacturer: String -> Float
    manu = X[:,2]
    from sklearn.preprocessing import LabelEncoder
    encoder = LabelEncoder()
    encoder.fit(manu)
    manu = encoder.transform(manu)
    X[:,2] = manu

    print("\nmanufacurer: string -> float")
    print(X)

    # # 사용자가 입력한 데이터 포함해서 다시 normalization 
    scaler = MinMaxScaler()
    scaler.fit(X)
    X = scaler.transform(X)
    print("\nMinMaxScaler resurt")
    print(X)

    user = X[-1] # 사용자 입력 값(표준화 완료)
    print("user",user)
    X = X[0:-1]# original data(normalized)

    #target data: String -> Float
    y = data_org['condition'].values
    LabelEncoder = LabelEncoder()
    LabelEncoder.fit(y)
    y = LabelEncoder.transform(y)
    print(y)

    X= np.insert(X,3,y,1) # 원래 형태로 만들기 위해.

    """from sklearn.preprocessing import LabelEncoder
    encoder = LabelEncoder()
    encoder.fit(user[2])
    user = encoder.transform(user)
    print("\nmanufacurer: string -> float")
    print(user[2])"""

    
    # 사용자가 입력한 데이터 포함해서 다시 normalization 
    User = user.reshape(1,-1)
    print("\nMinMaxScaler resurt")
    print(User)
    prediction=ensemble.predict(User)

    if(prediction==0):
        result="excellent"
    elif(prediction==1):
        result="fair"
    elif(prediction==2):
        result="good"
    elif(prediction==3):
        result="like new"
    elif(prediction==4):
        result="new"
    elif(prediction==5):
        result="salvage"
    
    #Set text at GUI
    label_final.config(text="\ncondition : "+result)
    
from tkinter import *
#setting window
root = Tk()
root.title("Predict condition")
root.geometry("300x400-500+140")
root.resizable(False, False)

#make text
label = Label(root, text = "\n\n< Type value >\n")
label.pack()

#price
label_price = Label(root, text = "Price : ")
label_price.pack()
entry_price = Entry(root, width=30)
entry_price.bind("<Return>", predict_multiple)
entry_price.pack()

#year
label_year = Label(root, text = "Year : ")
label_year.pack()
entry_year = Entry(root, width=30)
entry_year.bind("<Return>", predict_multiple)
entry_year.pack()

#manufacturer
label_manufacturer = Label(root, text = "Manufacturer : ")
label_manufacturer.pack()
entry_manu = Entry(root, width=30)
entry_manu.bind("<Return>", predict_multiple)
entry_manu.pack()

#cylinders
label_cylinders = Label(root, text = "Cylinders : ")
label_cylinders.pack()
entry_cylinders = Entry(root, width=30)
entry_cylinders.bind("<Return>", predict_multiple)
entry_cylinders.pack()

#odometer
label_odometer = Label(root, text = "Odometer : ")
label_odometer.pack()
entry_odometer = Entry(root, width=30)
entry_odometer.bind("<Return>", predict_multiple)
entry_odometer.pack()

#make Button
button = Button(root, width=10, text="show",overrelief="solid",command=predict_multiple)
button.pack()

#make text
label_final = Label(root,text=" ")
label_final.pack()

#set on the window
root.mainloop()


# In[ ]:




