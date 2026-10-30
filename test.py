import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

IMG_SIZE = 224
BATCH_SIZE = 32

# Load model
model = tf.keras.models.load_model("model.h5")

# Data generator
test_datagen = ImageDataGenerator(rescale=1./255)

val_data = test_datagen.flow_from_directory(
    "dataset/test",
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

# Evaluate
loss, accuracy = model.evaluate(val_data)

print("\n FINAL RESULTS")
print("Accuracy: {:.2f}%".format(accuracy * 100))

# Predictions
predictions = model.predict(val_data)
y_pred = np.argmax(predictions, axis=1)
y_true = val_data.classes

class_labels = list(val_data.class_indices.keys())

#  Classification Report
print("\n Classification Report:")
print(classification_report(y_true, y_pred, target_names=class_labels))

#  Confusion Matrix
cm = confusion_matrix(y_true, y_pred)

print("\n🧾 Confusion Matrix:")
print(cm)

#  Plot Confusion Matrix
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d',
            xticklabels=class_labels,
            yticklabels=class_labels)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")

# Save image for PPT
plt.savefig("confusion_matrix.png")
plt.show()