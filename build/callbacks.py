from scripts.checkpoint import DualCheckpoint
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from config.general import CALLBACK_CONFIG


def get_callbacks(
    checkpoint_path: str | None = None,
    bestmodel_path: str | None = None,
    cfg = CALLBACK_CONFIG,
    verbose = True,
    **overrides
):
    destin_cfg = cfg.model_copy(update=overrides)

    checkpoint = DualCheckpoint(
        # filepath1=os.path.join(config.CHECKPOINT_PATH, "model1", "model1_epoch{epoch:02d}_{val_macro_f1_score:.4f}.weights.h5"),
        # filepath2=os.path.join(config.BEST_MODEL, "model1.h5"),
        checkpoint_fpath=checkpoint_path,
        best_fpath=bestmodel_path,
        monitor="val_macro_f1_score",
        mode="max",
        verbose=verbose
    )

    early = EarlyStopping(
        patience=destin_cfg.early_patience,
        restore_best_weights=True,
        monitor="val_macro_f1_score",
        mode="max",
        verbose=verbose
    )
    
    reduce_lr = ReduceLROnPlateau(
        patience=destin_cfg.reduce_lr_patience,
        factor=destin_cfg.reduce_lr_factor,
        min_lr=destin_cfg.min_lr,
        monitor="val_macro_f1_score",
        mode="max",
        verbose=verbose
    )

    return checkpoint, early, reduce_lr