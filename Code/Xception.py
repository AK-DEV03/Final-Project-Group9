import cv2
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.metrics import mean_absolute_error
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.xception import preprocess_input
from tensorflow.keras.layers import GlobalMaxPooling2D, Dense
from tensorflow.keras.callbacks import ModelCheckpoint,EarlyStopping,ReduceLROnPlateau
from tensorflow.keras.applications.xception import Xception
from tensorflow.keras.optimizers import Adam


class BoneAge(tf.keras.Model):
    """ initializing variables """
    def __init__(self):
        super(BoneAge, self).__init__()
        self.df = None
        self.seed = 42
        self.image_paths = []
        self.batch_size = 32
        self.img_size = (256, 256)
        self.mean_bone_age = None
        self.std_bone_age = None
        self.x_train = []
        self.y_train = []
        self.x_val = []
        self.y_val = []
        self.x_test = []
        self.y_test = []
        self.train_datagen = ImageDataGenerator(rotation_range=20, width_shift_range=0.2, height_shift_range=0.2,
                                                 shear_range=0.02, zoom_range=0.3, fill_mode='nearest',
                                                 horizontal_flip=True, brightness_range=[0.5, 1],
                                                 data_format='channels_last', preprocessing_function=preprocess_input)
        self.val_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
        self.test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
        self.data_preparation()

    def read_images(self, df):
        """ Reading the images and resizing it and converting it numpy arrays """
        x = []
        y = []
        df = df.reset_index(drop=True)
        for i in range(len(df['image_paths'])):
            image_path = df.loc[i, 'image_paths']
            image = cv2.resize(cv2.imread(image_path), (256, 256))
            x.append(image)
            y.append(df.loc[i, 'bone_age_z'])
        x = np.array(x, dtype=np.float32)
        y = np.array(y, dtype=np.float32)
        return x, y

    def data_preparation(self):
        """ Preparing the data for modelling """
        self.df = pd.read_csv(os.getcwd() + '/boneage-training-dataset.csv')
        # Creating a new column age class which will used for stratified sampling during dataset split
        self.df['age_class'] = pd.cut(self.df['boneage'], 20)
        self.df['id'] = self.df['id'].apply(lambda x: str(x) + '.png')
        # Adding new column image_paths in the dataframe
        image_ids = self.df['id'].tolist()
        for image_id in image_ids:
            self.image_paths.append(os.path.join(os.getcwd(), 'boneage-training-dataset', image_id))
        self.df['image_paths'] = self.image_paths
        # Standardizing values in boneage column
        self.mean_bone_age = self.df['boneage'].mean()
        self.std_bone_age = self.df['boneage'].std()
        self.df['bone_age_z'] = (self.df['boneage'] - self.mean_bone_age) / self.std_bone_age

        # plotting left hand x-rays in each category
        bone_categories = self.df.age_class.unique().tolist()
        rows = 1
        f, ax = plt.subplots(nrows=rows, ncols=len(bone_categories), figsize=(50, 20 * 2))
        j = -1
        for category in bone_categories:
            j = j + 1
            samples = self.df[self.df['age_class'] == category]['image_paths'].sample(rows).values
            file_id = samples[0]
            im = cv2.imread(file_id)
            ax[j].imshow(im)
            ax[j].set_title(bone_categories[j], fontsize=12)
        plt.tight_layout()
        plt.show()

        # Splitting the dataset into train, validation and test
        train_df, test_df = train_test_split(self.df, test_size=0.2, random_state=self.seed,
                                             stratify=self.df['age_class'])
        train_df, valid_df = train_test_split(train_df, test_size=0.2, random_state=self.seed,
                                             stratify=train_df['age_class'])
        self.x_train, self.y_train = self.read_images(train_df)
        self.x_val, self.y_val = self.read_images(valid_df)
        self.x_test, self.y_test = self.read_images(test_df)
        print("Reading images and converting to numpy array completed")
        self.train_model()

    def plot_it(self, history):
        """ Plotting Mean absolute error """
        fig, ax = plt.subplots(figsize=(20, 10))
        ax.plot(history.history['mae_in_months'])
        ax.plot(history.history['val_mae_in_months'])
        plt.title('Model Error')
        plt.ylabel('error')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Val'], loc='upper right')
        ax.grid(color='black')
        plt.show()

    def mae_in_months(self, x_p, y_p):
        return mean_absolute_error((self.std_bone_age * x_p + self.mean_bone_age), (self.std_bone_age * y_p + self.mean_bone_age))

    def train_model(self):
        """ Training the model """
        print("Training the model")
        LR = 1e-3
        epochs = 200
        callbacks = [EarlyStopping(monitor='val_loss', min_delta=0, patience=30, verbose=0, mode='auto'),
                     ModelCheckpoint('model.h5', monitor='val_loss', mode='min', save_best_only=True),
                     ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, verbose=0, mode='auto',
                                       min_delta=0.0001, cooldown=0, min_lr=0)]
        # Pre trained model Xception without fully connected layers
        base_model = Xception(input_shape=(self.img_size[0], self.img_size[1], 3), include_top=False, weights='imagenet')
        # Unfreeze the layers
        base_model.trainable = True
        x = GlobalMaxPooling2D()(base_model.output)
        x = Dense(512, activation='relu')(x)
        x = Dense(10, activation='relu')(x)
        output = Dense(1, activation='linear')(x)
        model = Model(inputs=base_model.input, outputs=output)
        model.compile(loss='mse', optimizer=Adam(learning_rate=LR), metrics=[self.mae_in_months])
        print(base_model.summary())
        print(model.summary())
        history = model.fit_generator(self.train_datagen.flow(self.x_train, self.y_train, batch_size=self.batch_size),
                                      steps_per_epoch=len(self.x_train)/self.batch_size,
                                      validation_data=self.val_datagen.flow(self.x_val, self.y_val, batch_size=self.batch_size),
                                      validation_steps=len(self.x_val)/self.batch_size,
                                      callbacks=callbacks,
                                      epochs=epochs,
                                      verbose=1)
        self.plot_it(history)
        model.load_weights('model.h5')
        pred = self.mean_bone_age + self.std_bone_age * (model.predict(self.x_val, batch_size=self.batch_size, verbose=True))
        actual = self.mean_bone_age + self.std_bone_age * (self.y_val)

model = BoneAge()
