# train.py
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import json

# ====== USER TUNABLES ======
DATA_DIR = r"C:\Users\DELL\Music\AI-ML-DL\Employee_Monitoring\data\train"  # <-- fix this path
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 25
MODEL_OUT = "keras_model.h5"
LABELS_OUT = "labels.txt"
USE_CLASS_WEIGHTS = True         # set False if you want to test without weights
LEARNING_RATE = 1e-4
# ===========================

# 1) build generators
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=12,
    width_shift_range=0.08,
    height_shift_range=0.08,
    shear_range=0.04,
    zoom_range=0.12,
    horizontal_flip=True,
    brightness_range=(0.7, 1.3),
    validation_split=0.15,   # 15% val
)

train_gen = train_datagen.flow_from_directory(
    DATA_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training',
    shuffle=True
)

val_gen = train_datagen.flow_from_directory(
    DATA_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation',
    shuffle=False
)

class_indices = train_gen.class_indices
inv_class = {v: k for k, v in class_indices.items()}
print("Classes:", class_indices)

# save labels.txt in index order (0..N-1)
labels_sorted = [inv_class[i] for i in range(len(inv_class))]
with open(LABELS_OUT, "w") as f:
    for l in labels_sorted:
        f.write(l + "\n")
print("Saved labels ->", LABELS_OUT)

# 2) class weights (optional)
class_weight = None
if USE_CLASS_WEIGHTS:
    counts = np.bincount(train_gen.classes)
    num_classes = len(counts)

    for i, c in enumerate(counts):
        if c == 0:
            raise ValueError(
                f"Class index {i} has 0 images. "
                f"Check your folders under {DATA_DIR}"
            )

    total = counts.sum()
    class_weight = {
        i: total / (num_classes * counts[i])
        for i in range(num_classes)
    }
    print("Class counts:", counts)
    print("Class weights:", class_weight)

# 3) build model (transfer learning)
base = tf.keras.applications.MobileNetV2(
    input_shape=IMG_SIZE + (3,),
    include_top=False,
    weights='imagenet'
)
base.trainable = False   # freeze backbone first

x = layers.GlobalAveragePooling2D()(base.output)
x = layers.Dropout(0.4)(x)
x = layers.Dense(128, activation='relu')(x)
x = layers.Dropout(0.25)(x)
outputs = layers.Dense(len(class_indices), activation='softmax')(x)

model = models.Model(base.input, outputs)
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
model.summary()

# 4) callbacks
cb = [
    callbacks.ModelCheckpoint("best_model.h5", save_best_only=True, monitor="val_accuracy"),
    callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-7),
    callbacks.EarlyStopping(monitor="val_loss", patience=7, restore_best_weights=True)
]

# 5) train (stage 1)
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS,
    class_weight=class_weight,
    callbacks=cb
)

# Optional: fine-tune backbone
base.trainable = True
for layer in base.layers[:-50]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-5),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

history_ft = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=10,
    class_weight=class_weight,
    callbacks=cb
)

# 6) save final model
model.save(MODEL_OUT)
print("Saved model:", MODEL_OUT)

# 7) evaluate on validation set and print confusion matrix
val_gen.reset()
preds = model.predict(val_gen, verbose=1)
y_pred = np.argmax(preds, axis=1)
y_true = val_gen.classes

print("Classification Report:")
print(classification_report(y_true, y_pred, target_names=labels_sorted))

cm = confusion_matrix(y_true, y_pred)
print("Confusion matrix:\n", cm)

with open("confusion_matrix.json", "w") as f:
    json.dump({"labels": labels_sorted, "matrix": cm.tolist()}, f)
print("Saved confusion_matrix.json")
