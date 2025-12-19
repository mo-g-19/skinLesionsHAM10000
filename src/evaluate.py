#This file creates and saves a confusion matrix for a given model and dataset.
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def save_confusion_matrix(model, dataset, class_names, output_path):
    y_true, y_pred = [], []

    #Goes through the dataset and gets the true and predicted labels
    for batch_images, batch_labels in dataset:
        preds = model.predict(batch_images, verbose=0)
        pred_classes = np.argmax(preds, axis=1)
        true_classes = np.argmax(batch_labels, axis=1)

        y_pred.extend(pred_classes)
        y_true.extend(true_classes)

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    #Makes the confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    print("Confusion matrix:\n", cm)

    #Displays and saves the confusion matrix
    fig, ax = plt.subplots(figsize=(6, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(ax=ax, values_format="d", xticks_rotation=45)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close(fig)
