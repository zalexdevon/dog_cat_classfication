from src.create_model import create_model


def retrain(param, train_ds, epoch, loss):
    # Tạo model
    model = create_model(param)

    # compile
    optimizer = create_model(param, "optimizer")
    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=["accuracy"],
    )

    # Train
    model.fit(
        train_ds,
        epochs=epoch,
        verbose=1,
    )

    return model
