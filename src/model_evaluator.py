from Mylib import (
    tf_model_evaluator_classes,
)
from pathlib import Path
from sklearn import metrics
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


def evaluate(model, class_names, val_ds, model_evaluation_on_val_path):
    result_text, val_confusion_matrix = tf_model_evaluator_classes.ClassifierEvaluator(
        model=model,
        class_names=class_names,
        train_ds=val_ds,
    ).evaluate()
    model_result_text += result_text

    val_confusion_matrix_path = Path(
        f"{model_evaluation_on_val_path}/confusion_matrix.png"
    )
    val_confusion_matrix.savefig(
        val_confusion_matrix_path, dpi=None, bbox_inches="tight", format=None
    )

    with open(Path(f"{model_evaluation_on_val_path}/result.txt"), mode="w") as file:
        file.write(model_result_text)


def evaluate_on_test_ds(model, class_names, val_ds):
    test_target_data, test_pred = get_full_target_and_pred_for_softmax_model(
        model, val_ds
    )
    test_pred = [int(item) for item in test_pred]
    test_target_data = [int(item) for item in test_target_data]

    test_accuracy = metrics.accuracy_score(test_target_data, test_pred)

    class_names = np.asarray(class_names)
    named_test_target_data = class_names[test_target_data]
    named_test_pred = class_names[test_pred]

    test_classification_report = metrics.classification_report(
        named_test_target_data, named_test_pred
    )

    test_confusion_matrix = metrics.confusion_matrix(
        named_test_target_data, named_test_pred, labels=class_names
    )
    np.fill_diagonal(test_confusion_matrix, 0)
    test_confusion_matrix = get_heatmap_for_confusion_matrix(
        test_confusion_matrix, class_names
    )

    model_results_text = f"Test accuracy: {test_accuracy}\n"
    model_results_text += (
        f"Test classification_report: \n{test_classification_report}\n"
    )

    return model_results_text, test_confusion_matrix


def get_full_target_and_pred_for_softmax_model(model, ds):
    y_true = []
    y_pred = []

    for feature, true_data in ds:
        predictions = model.predict(feature, verbose=0)

        y_pred_batch = np.argmax(predictions, axis=-1).tolist()
        y_true_batch = true_data.numpy().tolist()
        y_true += y_true_batch
        y_pred += y_pred_batch

    return y_true, y_pred


def get_heatmap_for_confusion_matrix(confusion_matrix, labels):
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        confusion_matrix,
        cbar=True,
        annot=True,
        cmap="YlOrRd",
        ax=ax,
        xticklabels=labels,
        yticklabels=labels,
    )

    return fig
