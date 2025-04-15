import pandas as pd
import numpy as np

# Replace with the actual path to your CSV
df = pd.read_csv('C:\\Users\\Sir\\Desktop\\FINAL TRY\\fer+ck+augmentedfer.csv')

print(df.head())
print(df.shape)  # Check the number of rows and columns
print(df['emotion'].unique())  # Check unique emotion labels
print(df[' Usage'].unique())    # Check the usage splits
train_df = df[df[' Usage'] == 'Training']
val_df   = df[df[' Usage'] == 'PublicTest']   # or 'Validation' in some datasets
test_df  = df[df[' Usage'] == 'PrivateTest']  # or 'Test' in some datasets
def preprocess_pixels(pixel_string):
    # Split string by space -> list of pixel values
    pixels = pixel_string.split()
    # Convert to array of type float32
    pixels = np.array(pixels, dtype='float32')
    # Reshape to 48x48 (if that’s the size) and add channel dimension
    pixels = pixels.reshape((48, 48, 1))
    return pixels
x_train = np.stack(train_df[' pixels'].apply(preprocess_pixels).values)
y_train = train_df['emotion'].values
x_val = np.stack(val_df[' pixels'].apply(preprocess_pixels).values)
y_val = val_df['emotion'].values
x_test = np.stack(test_df[' pixels'].apply(preprocess_pixels).values)
y_test = test_df['emotion'].values
print(x_train.shape, y_train.shape)
# e.g., (28709, 48, 48, 1), (28709,)
x_train = x_train / 255.0
x_val   = x_val / 255.0
x_test  = x_test / 255.0
from tensorflow.keras.utils import to_categorical
num_classes = len(np.unique(y_train))  # e.g., 7 for typical FER
y_train_cat = to_categorical(y_train, num_classes)
y_val_cat   = to_categorical(y_val, num_classes)
y_test_cat  = to_categorical(y_test, num_classes)
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models

model = models.Sequential()

# Convolutional base
model.add(layers.Conv2D(32, (3,3), activation='relu', input_shape=(48,48,1)))
model.add(layers.BatchNormalization())
model.add(layers.MaxPooling2D((2,2)))

model.add(layers.Conv2D(64, (3,3), activation='relu'))
model.add(layers.BatchNormalization())
model.add(layers.MaxPooling2D((2,2)))

model.add(layers.Conv2D(128, (3,3), activation='relu'))
model.add(layers.BatchNormalization())
model.add(layers.MaxPooling2D((2,2)))

# Flatten & Dense layers
model.add(layers.Flatten())
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.BatchNormalization())
model.add(layers.Dense(num_classes, activation='softmax'))

# Compile the model
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',  # if one-hot
    metrics=['accuracy']
)

model.summary()
epochs = 30
batch_size = 64

history = model.fit(
    x_train,
    y_train_cat,       # or y_train if using sparse_categorical
    validation_data=(x_val, y_val_cat),
    epochs=epochs,
    batch_size=batch_size
)
test_loss, test_acc = model.evaluate(x_test, y_test_cat)
print(f"Test Accuracy: {test_acc:.4f}")
model.save('emotion_detection_model.h5')
import matplotlib.pyplot as plt

# Plot Accuracy
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# Plot Loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()
