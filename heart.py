# -*- coding: utf-8 -*-
"""
Created on Fri Apr 22 17:17:49 2022

@author: Zhou N
"""

#读取数据
import numpy as np 
import pandas as pd 
df_heart = pd.read_csv("heart.csv") 




#数据整理
#特征和标签
y = df_heart ['target']
X = df_heart.drop(['target'], axis=1)


#划分测试集和训练集
from sklearn.model_selection import train_test_split #拆分数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                   test_size=0.2, random_state=123)


#构建
import keras 
import tensorflow as tf
from keras.models import Sequential 
from keras.layers import Dense 
from keras.layers import Dropout
ann = Sequential()
ann.add(Dense(units=36, input_dim=13, activation = 'relu')) 
ann.add(Dense(units=48, activation = 'relu')) 
ann.add(Dropout(0.5)) # 添加Dropout
ann.add(Dense(units=60, activation = 'relu')) 
ann.add(Dense(units=82, activation = 'relu')) 
ann.add(Dropout(0.5)) # 添加Dropout
ann.add(Dense(units=100, activation = 'relu')) 
ann.add(Dense(units=1, activation = 'sigmoid')) 
ann.summary() 

#编译
ann.compile(optimizer = 'adam',loss = 'binary_crossentropy',metrics = ['acc'])

#训练
history = ann.fit(X_train,y_train,epochs=30,batch_size = 64, validation_data=(X_test,y_test))


import streamlit as st

#加标题
st.title('heart predict')

#构建数据收集UI
age=st.number_input('请输入年龄')
gender=st.selectbox('sex',options=('男','女'))
if gender=='男':
    sex=1
else:
    sex=0
       
cp=st.selectbox('cp',options=(0,1,2,3))
trestbps=st.number_input('insert trestbps')
chol=st.number_input('insert chol')
fbs=st.selectbox('fbs',options=(0,1))
restecg=st.selectbox('restcg',options=(0,1))
thalach=st.number_input('Insert thalach')
exang=st.selectbox('exang',options=(0,1))
oldpeak=st.number_input('Insert oldpeak')
slope=st.selectbox('slope',options=(0,1,2))
ca=st.selectbox('ca',options=(0,1))
thal=st.selectbox('thal',options=(1,2,3))

#八、将所收集数据构成DataFrame
allfactor=[[age,sex, cp,trestbps,chol,fbs,restecg,thalach,
       exang,oldpeak,slope,ca,thal]]

allfactor=pd.DataFrame(allfactor)

#九、调用模型预测
output=ann.predict(allfactor)
    
#十、定义函数解读预测结果
def dis(output):
    if output==1:
        return '您患心脏病风险较高'
    else:
        return '您患心脏病风险较低'

#十一、输出
outcome=dis(output)
st.write('预测结果为',outcome)





