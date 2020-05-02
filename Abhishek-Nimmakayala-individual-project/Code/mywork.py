#
import numpy as np
import cv2
import os
from keras.callbacks import ModelCheckpoint, EarlyStopping
from PIL import Image
from keras.models import Sequential
from keras.utils import to_categorical
from keras.layers import Dense, Flatten, BatchNormalization, Dropout, Activation, LeakyReLU,Conv2D,MaxPooling2D
import pandas as pd
from keras.optimizers import Adam
from tensorflow.keras.optimizers import SGD
import seaborn as sns
import matplotlib.pyplot as plt
from keras.preprocessing.image import img_to_array, ImageDataGenerator, load_img
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

#
x=[]
y_age=[]
y_gender=[]
train_df=pd.read_csv(os.getcwd() + "/boneage-training-dataset.csv")

print(os.getcwd())

for filename, boneage, gender in train_df[['id','boneage','male']].values:
    x.append(filename)
    y_age.append(boneage)
    y_gender.append(gender)
print(x[0:5])
print(y_age[0:5])
print(type(x))


x_in_paths=[]
for i in range (0,len(x)):
    x_in_paths.append(os.getcwd() + "/boneage-training-dataset/" + str(x[i])+'.png')

y_gender_final=[]
for i in range(0,len(y_gender)):
    # print(y_gender[i])
    if y_gender[i]==True:
        y_gender_final.append(1)
    else:
        y_gender_final.append(0)


print(x_in_paths[0:2])

print(y_gender_final[0:2])

#
#Data visulaization
gender_df_count=train_df.groupby(by='male').count()
gender_df_count.reset_index(inplace=True)

print(gender_df_count)


explode = (0, 0.05)
fig2, ax2 = plt.subplots()
ax2.pie(gender_df_count.boneage,autopct='%1.0f%%',shadow=True, labeldistance=1,startangle=90,explode=explode)
plt.legend(labels=['Girls','Boys'])
plt.title('Percentage of Boys and Girls')
plt.show()


# print(train_df.head(10))
#
gender_df_mean=train_df.groupby(by='male').mean()
gender_df_mean.reset_index(inplace=True)
print(gender_df_mean)
index=['Boys','Girls']
values=[118,135]
plt.bar(index,values)
plt.ylabel('Bone age in months')
plt.title("Average bone age of Boys and Girls")
plt.show()

# print(mean_age)
# plt.bar()
#
#distribution of age within each gender
male = train_df[train_df['male'] == True]
female = train_df[train_df['male'] == False]
fig, ax = plt.subplots(2,1)
ax[0].hist(male['boneage'], color = 'blue')
ax[0].set_ylabel('Number of boys')
ax[1].hist(female['boneage'], color = 'red')
ax[1].set_xlabel('Age in months')
ax[1].set_ylabel('Number of girls')
fig.set_size_inches((10,7))
plt.show()
#
# WAIT HERE

x_in_final = []   ## converts only x_train to images (x_in_bal has only x_train)
for i in range(0,len(x)):
    image = cv2.imread(x_in_paths[i])
    image = cv2.resize(image, (50, 50))
    image = np.array(image)
    x_in_final.append(image)


# image=Image.open('/home/ubuntu/Deep-Learning/boneage-training-dataset/boneage-training-dataset/1377.png','r')
# print(image)
print('done')
print(x_in_final[0])

#


x_in_final=np.array(x_in_final,dtype=np.float32)
y_age=np.array(y_gender_final,dtype=np.float32)
print(x_in_final.shape)
print(y_age.shape)


x_train, x_test, y_train_age, y_test_age = train_test_split(x_in_final, y_age, random_state=7, test_size=0.2)
x_train, x_test = x_train/255, x_test/255

#

generator = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    # shear_range=0.02,
    zoom_range=0.3,
    fill_mode='nearest',
    horizontal_flip=True,
    brightness_range=[0.5, 1],
    data_format='channels_last')


print(x_train.shape)
print(y_train_age.shape)
i=1
for batch in generator.flow(x_train,y_train_age,batch_size=32):
    x_train=np.concatenate((x_train,batch[0]),axis=0)
    y_train_age=np.concatenate((y_train_age,batch[1]),axis=0)
    i+=1
    if i>316:
        break


print(f"new x train shape {x_train.shape}")
print(f"new y train shape {y_train_age.shape}")



#

# Model 1 (Simple MLP)
epochs=100
batch_size = 128
model = Sequential()
model.add(Flatten(input_shape=x_train.shape[1:]))
model.add(Dropout(0.3))
model.add(Dense(512))
model.add(LeakyReLU(alpha=0.05))
model.add(BatchNormalization(axis=-1))
model.add(Dense(256))
model.add(LeakyReLU(alpha=0.05))
model.add(BatchNormalization())
model.add(Dense(128, activation = 'sigmoid'))
model.add(Dropout(0.5))
model.add(BatchNormalization())
model.add(Dense(64, activation = 'sigmoid'))
model.add(BatchNormalization())
model.add(Dense(1, activation = 'linear'))
print(model.summary())
LR = 1e-4

model.compile(optimizer=Adam(lr=LR), loss="mse", metrics=['mse','mae'])


graph=model.fit(x_train, y_train_age, epochs = epochs, batch_size = batch_size,validation_data=(x_test, y_test_age))



#

from sklearn.metrics import cohen_kappa_score, f1_score
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'



# Loss plot
plt.subplot(211)
plt.title(f"Loss, epochs={epochs},BS={batch_size}")
plt.plot(graph.history['loss'], label='train')
plt.plot(graph.history['val_loss'], label='validation')
plt.legend()
# Accuracy plot
print('\n')
plt.subplot(212)
plt.title('Mean Squared error')
plt.plot(graph.history['mae'], label='train')
plt.plot(graph.history['val_mse'], label='validation')
plt.legend()
plt.show()

# y_pred=model.predict(x_test)

#

# Model 2 (CNN Model)
epochs=200
batch_size = 128
LR2=1e-4
model = Sequential()
model.add(Conv2D(32, (3, 3), padding='same', input_shape=(x_train.shape[1:]), activation='relu'))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))
model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))
model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='linear'))
sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='mse', optimizer=Adam(lr=LR2), metrics=['mse','mae'])
print('done 2')
print(model.summary())

graph2=model.fit(x_train,y_train_age,batch_size=batch_size,epochs=epochs,validation_data=(x_test,y_test_age))
#
plt.subplot(211)
plt.title(f"Loss, epochs={epochs},BS={batch_size}")
plt.plot(graph2.history['loss'], label='train')
plt.plot(graph2.history['val_loss'], label='validation')
plt.legend()
# Accuracy plot
print('\n')
plt.subplot(212)
plt.title('MSE')
plt.plot(graph2.history['mse'], label='train')
plt.plot(graph2.history['val_mae'], label='validation')
plt.legend()
plt.show()