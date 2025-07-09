from pathlib import Path
from src import fix_param_dict, model_trainer
from Mylib import myfuncs


def do_model_training_stage(
    model_training_name,
    num_models,
    loss,
    patience,
    min_delta,
    epochs,
    param_dict,
):
    data_ingestion_path = Path("artifacts/data_ingestion")
    model_training_path = Path(f"artifacts/model_training/{model_training_name}")
    param_dict = fix_param_dict.fix_param_dict(param_dict)

    myfuncs.create_directories([model_training_path])

    class_names = myfuncs.load_python_object(data_ingestion_path / "class_names.pkl")

    model_trainer.train(
        param_dict,
        model_training_path,
        num_models,
        data_ingestion_path,
        loss,
        patience,
        min_delta,
        epochs,
        class_names,
    )


def do_model_training_stage_customized_train_function(
    model_training_name,
    num_models,
    loss,
    patience,
    min_delta,
    epochs,
    param_dict,
    train_model_function,
):
    data_ingestion_path = Path("artifacts/data_ingestion")
    model_training_path = Path(f"artifacts/model_training/{model_training_name}")
    param_dict = fix_param_dict.fix_param_dict(param_dict)

    myfuncs.create_directories([model_training_path])

    class_names = myfuncs.load_python_object(data_ingestion_path / "class_names.pkl")

    model_trainer.train_customized_train_function(
        param_dict,
        model_training_path,
        num_models,
        data_ingestion_path,
        loss,
        patience,
        min_delta,
        epochs,
        class_names,
        train_model_function,
    )


def do_model_training_stage_for_demoTfFunction(
    model_training_name, num_models, loss, patience, min_delta, epochs, param_dict
):
    print("Thực hiện do_model_training_stage_for_demoTfFunction")
    data_ingestion_path = Path("artifacts/data_ingestion")
    model_training_path = Path(f"artifacts/model_training/{model_training_name}")
    param_dict = fix_param_dict.fix_param_dict(param_dict)

    myfuncs.create_directories([model_training_path])

    class_names = myfuncs.load_python_object(data_ingestion_path / "class_names.pkl")

    model_trainer.train_demoTfFunction(
        param_dict,
        model_training_path,
        num_models,
        data_ingestion_path,
        loss,
        patience,
        min_delta,
        epochs,
        class_names,
    )


def do_model_training_stage_for_testCPUandGPUusage(
    model_training_name, num_models, loss, patience, min_delta, epochs, param_dict
):
    print("Thực hiện do_model_training_stage_for_testCPUandGPUusage")
    data_ingestion_path = Path("artifacts/data_ingestion")
    model_training_path = Path(f"artifacts/model_training/{model_training_name}")
    param_dict = fix_param_dict.fix_param_dict(param_dict)

    myfuncs.create_directories([model_training_path])

    class_names = myfuncs.load_python_object(data_ingestion_path / "class_names.pkl")

    model_trainer.train_testCPUandGPUusage(
        param_dict,
        model_training_path,
        num_models,
        data_ingestion_path,
        loss,
        patience,
        min_delta,
        epochs,
        class_names,
    )
