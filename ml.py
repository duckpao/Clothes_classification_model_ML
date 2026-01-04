# =========================================================
# 0. IMPORT
# =========================================================
import os
import shutil
import random
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

print("TensorFlow:", tf.__version__)

# =========================================================
# 1. CẤU HÌNH
# =========================================================
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 10
SEED = 42

BASE_DIR = "/kaggle/input/fashion-product-images-dataset/fashion-dataset"
IMG_SRC = f"{BASE_DIR}/images"
CSV_PATH = f"{BASE_DIR}/styles.csv"

WORK_DIR = "/kaggle/working/data"
TRAIN_DIR = f"{WORK_DIR}/train"
VAL_DIR   = f"{WORK_DIR}/val"

os.makedirs(TRAIN_DIR, exist_ok=True)
os.makedirs(VAL_DIR, exist_ok=True)

# =========================================================
# 2. LOAD & LÀM SẠCH CSV
# =========================================================
df = pd.read_csv(CSV_PATH, on_bad_lines="skip")

# Chỉ giữ các class phổ biến (để train nhanh & ổn định)
KEEP_CLASSES = [
    "Tshirts", "Tops", "Dresses",
    "Jeans", "Trousers",
    "Shoes", "Sandals"
]

df = df[df["articleType"].isin(KEEP_CLASSES)]
df = df[df["id"].notna()]

print("Số ảnh sau khi lọc:", len(df))
print(df["articleType"].value_counts())

# =========================================================
# 3. CHIA TRAIN / VAL
# =========================================================
train_df, val_df = train_test_split(
    df,
    test_size=0.2,
    random_state=SEED,
    stratify=df["articleType"]
)

# =========================================================
# 4. COPY ẢNH → FOLDER THEO CHUẨN KERAS
# =========================================================
def copy_images(dataframe, split_dir):
    for _, row in dataframe.iterrows():
        label = row["articleType"]
        img_name = str(row["id"]) + ".jpg"

        src_path = os.path.join(IMG_SRC, img_name)
        dst_dir = os.path.join(split_dir, label)
        os.makedirs(dst_dir, exist_ok=True)

        if os.path.exists(src_path):
            shutil.copy(src_path, os.path.join(dst_dir, img_name))

print("Copy train images...")
copy_images(train_df, TRAIN_DIR)

print("Copy val images...")
copy_images(val_df, VAL_DIR)

# =========================================================
# 5. IMAGE DATA GENERATOR (RGB)
# =========================================================
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True
)

val_datagen = ImageDataGenerator(rescale=1./255)

train_data = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical"
)

val_data = val_datagen.flow_from_directory(
    VAL_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False
)

class_names = list(train_data.class_indices.keys())
num_classes = train_data.num_classes

print("Classes:", class_names)

# =========================================================
# 6. MODEL – TRANSFER LEARNING (MobileNetV2)
# =========================================================
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights="imagenet"
)

base_model.trainable = False  # đóng băng backbone

x = base_model.output
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(128, activation="relu")(x)
x = layers.Dropout(0.3)(x)
output = layers.Dense(num_classes, activation="softmax")(x)

model = models.Model(inputs=base_model.input, outputs=output)

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# =========================================================
# 7. CALLBACKS
# =========================================================
callbacks = [
    ModelCheckpoint(
        "/kaggle/working/fashion_rgb_model.h5",
        save_best_only=True,
        monitor="val_loss"
    ),
    EarlyStopping(patience=5, restore_best_weights=True),
    ReduceLROnPlateau(patience=3, factor=0.5)
]

# =========================================================
# 8. TRAIN
# =========================================================
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=EPOCHS,
    callbacks=callbacks
)

# =========================================================
# 9. ĐÁNH GIÁ
# =========================================================
val_loss, val_acc = model.evaluate(val_data)
print(f"Validation Accuracy: {val_acc:.2f}")

# =========================================================
# 10. BIỂU ĐỒ
# =========================================================
df_hist = pd.DataFrame(history.history)

plt.figure(figsize=(10,4))
sns.lineplot(data=df_hist[["loss","val_loss"]])
plt.title("Loss")
plt.show()

plt.figure(figsize=(10,4))
sns.lineplot(data=df_hist[["accuracy","val_accuracy"]])
plt.title("Accuracy")
plt.show()

# =========================================================
# 11. CONFUSION MATRIX
# =========================================================
y_pred = model.predict(val_data)
y_pred_classes = np.argmax(y_pred, axis=1)

print(classification_report(
    val_data.classes,
    y_pred_classes,
    target_names=class_names
))

cm = confusion_matrix(val_data.classes, y_pred_classes)

plt.figure(figsize=(8,6))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    xticklabels=class_names,
    yticklabels=class_names
)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.show()
