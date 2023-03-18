#!/usr/bin/env python
# coding: utf-8

# In[1]:


from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
import matplotlib.pyplot as plt

df = pd.read_csv('/kaggle/input/digit-recognizer/train.csv')

df.head(5)


# In[2]:


# Extract targets

y = df['label']
del df['label']

# Splitting data on train and test

x_train, x_test, y_train, y_test = train_test_split(
    df, y, test_size = 0.2, random_state = 13)


# In[3]:


# Transform features data to 2D-array 

x_train = x_train.values.reshape((-1, 28, 28, 1))
x_train = x_train.astype("float32") / 255
x_test = x_test.values.reshape((-1, 28, 28, 1))
x_test = x_test.astype("float32") / 255

# Transform target data to binary

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)


# In[4]:


# Model building

model = keras.Sequential([
    layers.Conv2D(32, (3, 3), activation = "relu", input_shape = (28, 28, 1)),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.2),
    layers.Conv2D(64, (3, 3), activation = "relu"),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.1),
    layers.Conv2D(64, (3, 3), activation = "relu"),
    layers.BatchNormalization(),
    layers.Conv2D(64, (3, 3), activation = "relu"),
    layers.Flatten(),
    layers.Dense(64, activation = "relu"),
    layers.BatchNormalization(),
    layers.Dropout(0.1),
    layers.Dense(32, activation = "relu"),
    layers.Dense(10, activation = "softmax")
])

model.compile(optimizer = "Adam",
             loss = "categorical_crossentropy",
             metrics = ["accuracy"])

model.summary()


# In[5]:


# Fit it and save HISTORY

history = model.fit(x_train,
                    y_train,
                    epochs = 10, 
                    batch_size = 64, 
                    validation_data = (x_test, y_test))

history_dict = history.history
history_dict.keys()


# In[6]:


# Visualization of training process, epoch/loss

history_dict = history.history
loss_values = history_dict["loss"]
val_loss_values = history_dict["val_loss"]
epochs = range(1, len(loss_values) + 1)
plt.plot(epochs, loss_values, "bo", label = "Training loss")
plt.plot(epochs, val_loss_values, "b", label = "Validation loss")
plt.title("Training and validation loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()

# Visualization of training process, epoch/accuracy

plt.clf()
acc = history_dict["accuracy"]
val_acc = history_dict["val_accuracy"]
plt.plot(epochs, acc, "bo", label = "Training acc")
plt.plot(epochs, val_acc, "b", label = "Validation acc")
plt.title("Training and validation accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.show()


# In[7]:


# Prediction test

test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"test_acc: {test_acc}")


# In[8]:


# Preparing test data

test = pd.read_csv('/kaggle/input/digit-recognizer/test.csv')

test = test.values.reshape((-1, 28, 28, 1))
test = test.astype("float32") / 255

test.shape


# In[9]:


# Predict, concat Label and ImageId columns

Label = model.predict(test)

ImageId = np.arange(1, 28001)
Label = np.argmax(Label, axis = 1)
Label = Label.reshape(-1, 1)
ImageId = ImageId.reshape(-1, 1)

submission = np.concatenate((ImageId, Label), axis = 1)


# In[10]:


# Write submission file

submission = pd.DataFrame(submission).apply(np.int64)
submission = submission.rename(columns = {0: "ImageId", 1: "Label"})
submission.to_csv('/kaggle/working/submission.csv', index=False)
submission.head(10)

