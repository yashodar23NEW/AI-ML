import pandas as pd
url="https://raw.githubusercontent.com/selva86/datasets/refs/heads/master/Ionosphere.csv"
df=pd.read_csv(url)
df.head()

x=df.drop(['Class'],axis=1)
y=df['Class']
print (y)


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)


#import dependceies
from keras.models import Sequential,load_model
from keras.layers import Dense,Dropout

#build model

model=Sequential()
model.add(Dense(units=10,activation='relu',input_dim=len(x_train.columns)))
model.add(Dense(units=5,activation='relu'))
model.add(Dense(units=1,activation='sigmoid'))
model.summary()


# Compile the model

model.compile(loss='binary_crossentropy',optimizer='sgd',metrics=['accuracy'])

# train model

model.fit(x_train,y_train,epochs=100,batch_size=32)


# save the model
model.save('weights.h5')
