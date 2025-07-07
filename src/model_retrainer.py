from src import create_model
from src import create_object


def retrain(param, train_ds, epoch, loss):
    model = create_model.create_model(param)

    optimizer = create_object.create_object(param, "optimizer")
    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=["accuracy"],
    )

    model.fit(
        x=train_ds,
        epochs=epoch,
        verbose=1,
    )

    return model
