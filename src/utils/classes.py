import pandas as pd
import numpy as np
import os
from Mylib import myfuncs
from datetime import datetime
from src.utils import funcs


class CustomisedModelCheckpoint(tf.keras.callbacks.Callback):
    SCORINGS_PREFER_MININUM = ["loss", "mse", "mae"]
    SCORINGS_PREFER_MAXIMUM = ["accuracy", "roc_auc"]

    def __init__(
        self,
        filepath: str,
        scoring_path: str,
        monitor: str,
        val_scoring_limit_to_save_model: float,
    ):
        """Customized từ class ModelCheckpoint trong tf.keras.callbacks, ở đây thêm logic để save model khi đạt (1) <br>
        (1): val scoring của best model phải tốt hơn val_scoring_limit_to_save_model <br>

        Args:
            filepath (str): đường dẫn lưu model
            scoring_path (str): đường dẫn lưu train, val scoring
            monitor (str): chỉ số
            val_scoring_limit_to_save_model (float): mức cần vượt qua để lưu model
        """
        super().__init__()
        self.filepath = filepath
        self.scoring_path = scoring_path
        self.monitor = monitor
        self.val_scoring_limit_to_save_model = val_scoring_limit_to_save_model

    def on_train_begin(self, logs=None):
        # Nếu thuộc SCORINGS_PREFER_MININUM thì lấy âm đẩy về bài toán tìm giá trị lớn nhất
        self.sign_for_score = None
        if self.monitor in self.SCORINGS_PREFER_MAXIMUM:
            self.sign_for_score = 1
        elif self.monitor in self.SCORINGS_PREFER_MININUM:
            self.sign_for_score = -1
        else:
            raise ValueError(f"Chưa định nghĩa cho {self.monitor}")

        self.train_scorings = []
        self.val_scorings = []
        self.models = []

    def on_epoch_end(self, epoch, logs=None):
        self.train_scorings.append(logs.get(self.monitor) * self.sign_for_score)
        self.val_scorings.append(logs.get(f"val_{self.monitor}") * self.sign_for_score)
        self.models.append(self.model)

    def on_train_end(self, logs=None):
        # Tìm model ứng với val scoring tốt nhất
        index_best_model = np.argmax(self.val_scorings)
        best_val_scoring = self.val_scorings[index_best_model]
        best_train_scoring = self.train_scorings[index_best_model]
        best_model = self.models[index_best_model]

        # Trước đó lấy âm -> lấy trị tuyệt đối
        best_val_scoring = np.abs(best_val_scoring)
        best_train_scoring = np.abs(best_train_scoring)

        print(f"Model tốt nhất ứng với epoch = {index_best_model + 1}")

        # Lưu kết quả model
        myfuncs.save_python_object(
            self.scoring_path, (best_train_scoring, best_val_scoring)
        )

        # Lưu model
        self.save_model(best_val_scoring, best_model)

    def save_model(self, best_val_scoring, best_model):
        do_allow_to_save_model = self.is_val_scoring_better_than_target_scoring(
            best_val_scoring
        )
        if do_allow_to_save_model:
            best_model.save(self.filepath)

    def is_val_scoring_better_than_target_scoring(self, val_scoring):
        if self.monitor in self.SCORINGS_PREFER_MAXIMUM:
            return val_scoring > self.val_scoring_limit_to_save_model
        if self.monitor in self.SCORINGS_PREFER_MININUM:
            return val_scoring < self.val_scoring_limit_to_save_model

        raise ValueError(f"Chưa định nghĩa cho {self.monitor}")


class LoggingDisplayer:
    DATE_FORMAT = "%d-%m-%Y-%H-%M-%S"
    READ_FOLDER_NAME = "artifacts/logs"
    WRITE_FOLDER_NAME = "artifacts/gather_logs"

    # Tạo thư mục
    os.makedirs(WRITE_FOLDER_NAME, exist_ok=True)

    def __init__(self, mode, file_name=None, start_time=None, end_time=None):
        self.mode = mode
        self.file_name = file_name
        self.start_time = start_time
        self.end_time = end_time

        if self.file_name is None:
            self.file_name = f"{datetime.now().strftime(self.DATE_FORMAT)}.log"

    def print_and_save(self):
        file_path = f"{self.WRITE_FOLDER_NAME}/{self.file_name}"

        if self.mode == "all":
            result = self.gather_all_logging_result()
        else:
            result = self.gather_logging_result_from_start_to_end_time()

        print(result)
        print(f"Lưu result tại {file_path}")
        myfuncs.write_content_to_file(result, file_path)

    def gather_all_logging_result(self):
        logs_filenames = self.get_logs_filenames()

        return self.read_from_logs_filenames(logs_filenames)

    def gather_logging_result_from_start_to_end_time(self):
        logs_filenames = pd.Series(self.get_logs_filenames())
        logs_filenames = logs_filenames[
            (logs_filenames > self.start_time) & (logs_filenames < self.end_time)
        ].tolist()

        return self.read_from_logs_filenames(logs_filenames)

    def read_from_logs_filenames(self, logs_filenames):
        result = ""
        for logs_filename in logs_filenames:
            logs_filepath = f"{self.READ_FOLDER_NAME}/{logs_filename}.log"
            content = myfuncs.read_content_from_file_60(logs_filepath)
            result += f"{content}\n\n"

        return result

    def get_logs_filenames(self):
        logs_filenames = os.listdir(self.READ_FOLDER_NAME)
        date_format_in_filename = f"{self.DATE_FORMAT}.log"
        logs_filenames = [
            datetime.strptime(item, date_format_in_filename) for item in logs_filenames
        ]
        logs_filenames = sorted(logs_filenames)  # Sắp xếp theo thời gian tăng dần
        return logs_filenames


class ModelTrainingResultPlotter:
    def __init__(self, max_val_value, target_val_value):
        self.max_val_value = max_val_value
        self.target_val_value = target_val_value

    def plot(self):
        components = funcs.gather_result_from_model_training()
        fig = self.plot_from_components(components)

        fig.show()

    def plot_from_components(self, components):
        model_names, train_scores, val_scores, _, _ = zip(*components)

        for i in range(len(train_scores)):
            if train_scores[i] > self.max_val_value:
                train_scores[i] = self.max_val_value

            if val_scores[i] > self.max_val_value:
                val_scores[i] = self.max_val_value

        # Vẽ biểu đồ
        df = pd.DataFrame(
            {
                "x": model_names,
                "train": train_scores,
                "val": val_scores,
            }
        )

        df_long = df.melt(
            id_vars=["x"],
            value_vars=["train", "val"],
            var_name="Category",
            value_name="y",
        )

        fig = px.line(
            df_long,
            x="x",
            y="y",
            color="Category",
            markers=True,
            color_discrete_map={
                "train": "gray",
                "val": "blue",
            },
            hover_data={"x": False, "y": True, "Category": False},
        )

        fig.add_hline(
            y=self.max_val_value,
            line_dash="solid",
            line_color="black",
            line_width=2,
        )

        fig.add_hline(
            y=self.target_val_value,
            line_dash="dash",
            line_color="green",
            line_width=2,
        )

        fig.update_layout(
            autosize=False,
            width=100 * (len(model_names) + 2) + 30,
            height=400,
            margin=dict(l=30, r=10, t=10, b=0),
            xaxis=dict(
                title="",
                range=[
                    0,
                    len(model_names),
                ],
                tickmode="linear",
            ),
            yaxis=dict(
                title="",
                range=[0, self.max_val_value],
            ),
            showlegend=False,
        )

        return fig
