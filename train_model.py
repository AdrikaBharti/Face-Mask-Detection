#1  Import required libraries
# import system libs
import os
import time
import shutil
import pathlib
import itertools
from PIL import Image


# import data handling tools
import cv2
import numpy as np
import pandas as pd
import seaborn as sns
import squarify


sns.set_style('darkgrid')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
# import Deep learning Libraries
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam, Adamax
from tensorflow.keras.metrics import categorical_crossentropy
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Activation, Dropout, BatchNormalization
from tensorflow.keras import regularizers
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, GlobalAveragePooling2D

#2 Load the dataset
# Ignore Warnings
import warnings
warnings.filterwarnings("ignore")

print ('modules loaded')

train_dir = 'Face Mask Dataset\\Train'
filepaths = []
labels = []
# Create folder to save charts
CHART_DIR = "charts"
os.makedirs(CHART_DIR, exist_ok=True)
folds = os.listdir(train_dir)
for fold in folds:
    foldpath = os.path.join(train_dir, fold)
    filelist = os.listdir(foldpath)
    for file in filelist:
        fpath = os.path.join(foldpath, file)
        filepaths.append(fpath)
        labels.append(fold)

# Concatenate data paths with labels into one dataframe
Fseries = pd.Series(filepaths, name= 'filepaths')
Lseries = pd.Series(labels, name='labels')
train_df = pd.concat([Fseries, Lseries], axis= 1)
print(train_df)
# Generate  test data paths with labels
test_dir = 'Face Mask Dataset\\Test'
filepaths = []
labels = []

folds = os.listdir(test_dir)
for fold in folds:
    foldpath = os.path.join(test_dir, fold)
    filelist = os.listdir(foldpath)
    for file in filelist:
        fpath = os.path.join(foldpath, file)
        filepaths.append(fpath)
        labels.append(fold)

# Concatenate data paths with labels into one dataframe
Fseries = pd.Series(filepaths, name= 'filepaths')
Lseries = pd.Series(labels, name='labels')
test_df = pd.concat([Fseries, Lseries], axis= 1)
print(test_df)
#3 Data Pre-Processing & Feature selection

def clean_image_folder(folder_path):
    valid_extensions = ['.jpg', '.jpeg', '.png']
    for class_folder in os.listdir(folder_path):
        class_path = os.path.join(folder_path, class_folder)
        for img_file in os.listdir(class_path):
            img_path = os.path.join(class_path, img_file)
            ext = os.path.splitext(img_file)[1].lower()
            if ext not in valid_extensions:
                print(f"Removing unsupported file: {img_path}")
                os.remove(img_path)
                continue
            # Try to load image
            try:
                img = cv2.imread(img_path)
                if img is None:
                    print(f"Removing corrupted image: {img_path}")
                    os.remove(img_path)
            except Exception as e:
                print(f"Error loading {img_path}: {e}")
                os.remove(img_path)

# Run for train and test folders
clean_image_folder(train_dir)
clean_image_folder(test_dir)
#4 Data Visualization


#Bar Chart
plt.figure(figsize=(8,6))
sns.countplot(x='labels', data=train_df, palette='viridis')
plt.title('Class Distribution (Bar Chart)')
plt.xlabel('Classes')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(CHART_DIR, "bar_chart.png"), dpi=300, bbox_inches='tight')
plt.show()

#Heat Map
# Add image width and height to the DataFrame
widths = []
heights = []

for path in train_df['filepaths']:
    with Image.open(path) as img:
        w, h = img.size
        widths.append(w)
        heights.append(h)

train_df['width'] = widths
train_df['height'] = heights


# Correlation matrix
plt.figure(figsize=(6,5))
sns.heatmap(train_df[['width', 'height']].corr(), annot=True, cmap='coolwarm')
plt.title('Heatmap of Image Dimensions')
plt.savefig(os.path.join(CHART_DIR, "heatmap.png"), dpi=300, bbox_inches='tight')
plt.show()

#Histogram
plt.figure(figsize=(12,5))

# Width Histogram
plt.subplot(1, 2, 1)
sns.histplot(train_df['width'], bins=20, kde=True, color='skyblue')
plt.title('Image Width Distribution')

# Height Histogram
plt.subplot(1, 2, 2)
sns.histplot(train_df['height'], bins=20, kde=True, color='salmon')
plt.title('Image Height Distribution')

plt.tight_layout()
plt.savefig(os.path.join(CHART_DIR, "histogram.png"), dpi=300, bbox_inches='tight')
plt.show()

#Pie Chart
class_counts = train_df['labels'].value_counts()  # Changed from 'label' to 'labels'
plt.figure(figsize=(6,6))
plt.pie(class_counts, labels=class_counts.index, autopct='%1.1f%%', startangle=140, colors=sns.color_palette('pastel'))
plt.title('Class Distribution (Pie Chart)')
plt.axis('equal')
plt.savefig(os.path.join(CHART_DIR, "pie_chart.png"), dpi=300, bbox_inches='tight')
plt.show()

#Treemap
plt.figure(figsize=(8,5))
squarify.plot(sizes=class_counts.values, label=class_counts.index, alpha=0.8, color=sns.color_palette('pastel'))
plt.title('Class Distribution (Treemap)')
plt.axis('off')
plt.savefig(os.path.join(CHART_DIR, "Treemap.png"), dpi=300, bbox_inches='tight')
plt.show()
 #5 Splitting and Training the data
# valid and test dataframe
valid_df, test_df = train_test_split(test_df,  train_size= 0.6, shuffle= True, random_state= 123)

# crobed image size
batch_size = 16
img_size = (224, 224)
channels = 3
img_shape = (img_size[0], img_size[1], channels)

# Add data augmentation to the training generator
tr_gen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest'
)
valid_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
train_gen = tr_gen.flow_from_dataframe( train_df, x_col= 'filepaths', y_col= 'labels', target_size= img_size, class_mode= 'categorical',
                                    color_mode= 'rgb', shuffle= True, batch_size= batch_size)

valid_gen =valid_datagen.flow_from_dataframe( valid_df, x_col= 'filepaths', y_col= 'labels', target_size= img_size, class_mode= 'categorical',
                                    color_mode= 'rgb', shuffle= True, batch_size= batch_size)

test_gen = valid_datagen.flow_from_dataframe( test_df, x_col= 'filepaths', y_col= 'labels', target_size= img_size, class_mode= 'categorical',
                                    color_mode= 'rgb', shuffle= False, batch_size= batch_size)

g_dict = train_gen.class_indices      # defines dictionary {'class': index}
classes = list(g_dict.keys())       # defines list of dictionary's kays (classes), classes names : string
images, labels = next(train_gen)      # get a batch size samples from the generator

plt.figure(figsize= (20, 20))

for i in range(16):
    plt.subplot(4, 4, i + 1)
    image = images[i] / 255       # scales data to range (0 - 255)
    plt.imshow(image)
    index = np.argmax(labels[i])  # get image index
    class_name = classes[index]   # get class of image
    plt.title(class_name, color= 'blue', fontsize= 12)
    plt.axis('off')
plt.savefig(os.path.join(CHART_DIR, "sample_images.png"), dpi=300, bbox_inches='tight')
plt.show()
 #6 Vll load the model
# Create Model Structure

# create pre-trained model (you can built on pretrained model such as :  efficientnet, VGG , Resnet )
# we will use efficientnetb3 from EfficientNet family.
# Create Model Structure with EfficientNetB0 as base
base_model = tf.keras.applications.efficientnet.EfficientNetB0(
    include_top=False, 
    weights="imagenet", 
    input_shape=img_shape, 
    pooling='max'
)
base_model.trainable = False  # Freeze base model initially

model = Sequential([
    base_model,
    BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001),
    Dense(256, activation='relu'),
    Dropout(rate=0.45, seed=123),
    Dense(len(classes), activation='softmax')
    
])

# Use a lower learning rate
optimizer = Adamax(learning_rate=0.0005)
model.compile(optimizer=optimizer, 
              loss='categorical_crossentropy', 
              metrics=['accuracy'])

model.summary()

batch_size = 20   # set batch size for training
epochs =  20  # number of all epochs in training

# Add callbacks
early_stop = EarlyStopping(
    monitor='val_accuracy', 
    patience=2, 
    restore_best_weights=True
)
reduce_lr = ReduceLROnPlateau(
    monitor='val_loss', 
    factor=0.5, 
    patience=1, 
    min_lr=0.00001
)

# Train for only 5 epochs
history = model.fit(
    x=train_gen,
    epochs=epochs,   # now uses 20
    verbose=1,
    validation_data=valid_gen,
    callbacks=[early_stop, reduce_lr]
)


# Define needed variables
tr_acc = history.history['accuracy']
tr_loss = history.history['loss']
val_acc = history.history['val_accuracy']
val_loss = history.history['val_loss']
index_loss = np.argmin(val_loss)
val_lowest = val_loss[index_loss]
index_acc = np.argmax(val_acc)
acc_highest = val_acc[index_acc]
Epochs = [i+1 for i in range(len(tr_acc))]
loss_label = f'best epoch= {str(index_loss + 1)}'
acc_label = f'best epoch= {str(index_acc + 1)}'


# Plot training history
plt.figure(figsize= (20, 8))
plt.style.use('fivethirtyeight')
plt.subplot(1, 2, 1)
plt.plot(Epochs, tr_loss, 'r', label= 'Training loss')
plt.plot(Epochs, val_loss, 'g', label= 'Validation loss')
plt.scatter(index_loss + 1, val_lowest, s= 150, c= 'blue', label= loss_label)
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.subplot(1, 2, 2)
plt.plot(Epochs, tr_acc, 'r', label= 'Training Accuracy')
plt.plot(Epochs, val_acc, 'g', label= 'Validation Accuracy')
plt.scatter(index_acc + 1 , acc_highest, s= 150, c= 'blue', label= acc_label)
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(CHART_DIR, "training_history.png"), dpi=300, bbox_inches='tight')
plt.show()

ts_length = len(test_df)
test_batch_size = max(sorted([ts_length // n for n in range(1, ts_length + 1) if ts_length%n == 0 and ts_length/n <= 80]))
test_steps = ts_length // test_batch_size

train_score = model.evaluate(train_gen, steps= test_steps, verbose= 1)
valid_score = model.evaluate(valid_gen, steps= test_steps, verbose= 1)
test_score = model.evaluate(test_gen, steps= test_steps, verbose= 1)

print("Train Loss: ", train_score[0])
print("Train Accuracy: ", train_score[1])
print('-' * 20)
print("Validation Loss: ", valid_score[0])
print("Validation Accuracy: ", valid_score[1])
print('-' * 20)
print("Test Loss: ", test_score[0])
print("Test Accuracy: ", test_score[1])

#7 Evaluating the model
#get prediction
preds = model.predict(test_gen)
y_pred = np.argmax(preds, axis=1)

g_dict = test_gen.class_indices
classes = list(g_dict.keys())

# Confusion matrix
cm = confusion_matrix(test_gen.classes, y_pred)

plt.figure(figsize= (10, 10))
plt.imshow(cm, interpolation= 'nearest', cmap= plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()

tick_marks = np.arange(len(classes))
plt.xticks(tick_marks, classes, rotation= 45)
plt.yticks(tick_marks, classes)
thresh = cm.max() / 2.
for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    plt.text(j, i, cm[i, j], horizontalalignment= 'center', color= 'white' if cm[i, j] > thresh else 'black')

plt.tight_layout()
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.savefig(os.path.join(CHART_DIR, "confusion_matrix.png"), dpi=300, bbox_inches='tight')
plt.show()

# Classification report
print(classification_report(test_gen.classes, y_pred, target_names= classes))
report = classification_report(
    test_gen.classes,
    y_pred,
    target_names=classes
)

with open(os.path.join(CHART_DIR, "classification_report.txt"), "w") as f:
    f.write(report)

# Save the model to current directory
model.save("face_mask_model.h5")

# Verify the file exists before loading
if os.path.exists('face_mask_model.h5'):
    loaded_model = tf.keras.models.load_model('face_mask_model.h5', compile=False)
    loaded_model.compile(Adamax(learning_rate=0.001), 
                        loss='categorical_crossentropy', 
                        metrics=['accuracy'])
    
    # Use a specific test image that exists
    test_image_dir ='Face Mask Dataset\\Test\\WithoutMask'
    if os.path.exists(test_image_dir):
        # Get first image from directory
        test_images = [f for f in os.listdir(test_image_dir) if f.endswith(('.jpg', '.png'))]
        if test_images:
            image_path = os.path.join(test_image_dir, test_images[0])
            image = Image.open(image_path)
            
            # Preprocess and predict
            img = image.resize((224, 224))
            img_array = tf.keras.preprocessing.image.img_to_array(img)
            img_array = tf.expand_dims(img_array, 0)
            
            predictions = loaded_model.predict(img_array)
            class_labels = classes
            score = predictions[0]
            print(f"Prediction: {class_labels[np.argmax(score)]}")
        else:
            print(f"No test images found in {test_image_dir}")
    else:
        print(f"Test directory not found: {test_image_dir}")
else:


    print("Model file not found - save failed")



