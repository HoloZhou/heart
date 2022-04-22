# -*- coding: utf-8 -*-
"""
Created on Fri Apr 22 20:14:55 2022

@author: Zhou N
"""
import tensorflow as tf
import numpy as np 
import pandas as pd 
from tensorflow.keras.models import load_model

ann = tf.keras.models.load_model('heart.h5') 

import streamlit as st


#加标题
st.set_page_config(layout="wide")
col1, col2, col3, col4, col5=st.columns([1, 1, 1, 1, 1])



with col1:
    st.title('心脏病风险预测')
with col2:
    age=st.number_input('请输入年龄')
    gender=st.selectbox('sex',options=('男','女'))
    if gender=='男':
        sex=1
    else:
        sex=0
with col3:
    cp=st.selectbox('胸痛类型',options=(0,1,2,3))
    trestbps=st.number_input('静息血压')
    chol=st.number_input('insert chol')
    fb=st.selectbox('血糖值是否大于120mg/dl',options=('是','否'))
    if fb=='是':
        fbs=1
    else:
        fbs=0
with col4:   
    restecg=st.selectbox('静息心电图结果',options=(0,1))
    
    thalach=st.number_input('Insert thalach')
    exang=st.selectbox('exang',options=(0,1))
    
with col5:
    oldpeak=st.number_input('运动后ST段降低')
    slope=st.selectbox('峰值运动时ST段坡度',options=(0,1,2))
    ca=st.selectbox('ca',options=(0,1))
    tha=st.selectbox('thal',options=('正常','可逆损伤','不可逆损伤'))
    if tha=='正常':
        thal=1
    elif tha=='可逆损伤':
        thal=2
    else:
        thal=3



       


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
