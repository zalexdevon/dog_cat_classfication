import tensorflow as tf
from Mylib import myfuncs, tf_myclasses, tf_myfuncs, tf_model_evaluator
import os


def load_data(data_transformation_path, class_names_path, model_path):
    batch_size = myfuncs.load_python_object(
        f"{data_transformation_path}/batch_size.pkl"
    )
    image_size = myfuncs.load_python_object(
        f"{data_transformation_path}/image_size.pkl"
    )
    class_names = myfuncs.load_python_object(class_names_path)
    model = tf.keras.models.load_model(model_path)

    return batch_size, image_size, class_names, model


def create_test_ds(test_path, batch_size, image_size, class_names):
    test_ds = tf.keras.preprocessing.image_dataset_from_directory(
        test_path,
        shuffle=True,
        image_size=(image_size, image_size),
        batch_size=batch_size,
        class_names=class_names,
    )
    test_ds = tf_myfuncs.cache_prefetch_tfdataset_2(test_ds)
    return test_ds


def evaluate_model_on_test(test_ds, class_names, model, model_evaluation_on_test_path):
    final_model_results_text = (
        "===============Kết quả đánh giá model==================\n"
    )

    # Đánh giá model trên tập train, val
    model_results_text, test_confusion_matrix = tf_model_evaluator.ClassifierEvaluator(
        model=model,
        class_names=class_names,
        train_ds=test_ds,
    ).evaluate()
    final_model_results_text += model_results_text  # Thêm đoạn đánh giá vào

    # Lưu lại confusion matrix cho tập test
    test_confusion_matrix_path = os.path.join(
        model_evaluation_on_test_path, "test_confusion_matrix.png"
    )
    test_confusion_matrix.savefig(
        test_confusion_matrix_path, dpi=None, bbox_inches="tight", format=None
    )

    # Lưu vào file results.txt
    with open(
        os.path.join(model_evaluation_on_test_path, "result.txt"), mode="w"
    ) as file:
        file.write(final_model_results_text)
