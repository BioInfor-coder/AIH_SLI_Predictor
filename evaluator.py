# encoding: utf-8
import argparse
import itertools
import os.path
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_curve, auc, confusion_matrix,
    classification_report
)


# Plot ROC curve and print the best threshold (closest to top-left corner)
def plot_roc_curve(y_true, y_scores, sensitivity, specificity, filename='roc_curve.png'):
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)  # Compute ROC curve
    roc_auc = auc(fpr, tpr)  # Compute AUC

    # Find the point closest to the top-left corner
    best_idx = np.argmin(np.sqrt(fpr ** 2 + (1 - tpr) ** 2))
    best_threshold = thresholds[best_idx]
    print("Best threshold: {:.4f}, FPR: {:.4f}, TPR: {:.4f}".format(
        best_threshold, fpr[best_idx], tpr[best_idx]))

    # Plot ROC curve
    plt.figure()
    plt.plot(fpr, tpr, color='blue', label=f"AUC = {roc_auc:.4f}")
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')  # Diagonal line
    plt.title("ROC Curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")
    plt.savefig(filename)
    # plt.show()


# Plot confusion matrix
def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix',
                          cmap=plt.cm.Blues, filename="Confusion_matrix"):
    plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=0)
    plt.yticks(tick_marks, classes)

    # Display values inside the matrix cells
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment='center',
                 color='white' if cm[i, j] > thresh else 'black')

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(filename)
    plt.close()


# Main function to load model, predict and evaluate performance
def evaluate_model(model_path, train_X_path, train_Y_path, output_dir):
    # Load the trained model
    with open(model_path, 'rb') as f:
        model = pickle.load(f)

    # Load training data
    train_X = np.loadtxt(train_X_path)
    train_Y = np.loadtxt(train_Y_path)

    # Make predictions
    y_scores = model.predict_proba(train_X)[:, 1]
    y_preds = model.predict(train_X)

    # Compute confusion matrix and classification report
    cm = confusion_matrix(train_Y, y_preds)
    print("Confusion Matrix:\n", cm)
    print("Classification Report:\n", classification_report(train_Y, y_preds))

    sensitivity = cm[1, 1] / (cm[1, 1] + cm[1, 0])  # True Positive Rate
    specificity = cm[0, 0] / (cm[0, 0] + cm[0, 1])  # True Negative Rate
    print("Sensitivity:", sensitivity)
    print("Specificity:", specificity)

    # Plot evaluation curves
    plot_roc_curve(train_Y, y_scores, sensitivity, specificity,filename=os.path.join(output_dir,'ROC_plot.png'))
    # plot confusion matrix
    class_names = ['low', 'high']
    plot_confusion_matrix(cm, classes=class_names, title='Confusion Matrix',
                          filename=f'{output_dir}/Confusion_Matrix.png')


# Entry point: use command-line arguments to run the evaluation
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate a trained model.')

    parser.add_argument('--model_path', type=str, required=False, default='./weights/RandomForestClassifier_weight.pkl',
                        help='Path to the trained model file (.pkl)')
    parser.add_argument('--input_X', type=str, required=False, default='./data/test_X.txt',
                        help='Path to the scaled training feature data (.txt)')
    parser.add_argument('--input_Y', type=str, required=False, default='./data/test_Y.txt',
                        help='Path to the training label data (.txt)')
    parser.add_argument('--output_dir', type=str, required=False, default='',
                        help='Path to the output the results')

    args = parser.parse_args()

    evaluate_model(
        model_path=args.model_path,
        train_X_path=args.input_X,
        train_Y_path=args.input_Y,
        output_dir=args.output_dir
    )
