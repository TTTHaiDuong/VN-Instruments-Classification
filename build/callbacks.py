from scripts.checkpoint import DualCheckpoint
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from config.general import CALLBACK_CONFIG


def get_callbacks(
    checkpoint_path: str,
    bestmodel_path: str,
    cfg = CALLBACK_CONFIG,
    verbose = True,
    **overrides
):
    destin_cfg = cfg.model_copy(update=overrides)

    checkpoint = DualCheckpoint(
        # filepath1=os.path.join(config.CHECKPOINT_PATH, "model1", "model1_epoch{epoch:02d}_{val_macro_f1_score:.4f}.weights.h5"),
        # filepath2=os.path.join(config.BEST_MODEL, "model1.h5"),
        filepath1=checkpoint_path,
        filepath2=bestmodel_path,
        monitor="val_macro_f1_score",
        save_best_only=True,
        save_weights_only=True,
        verbose=verbose,
        mode="max"
    )

    early = EarlyStopping(
        monitor="val_macro_f1_score",
        restore_best_weights=True,
        patience=destin_cfg.early_patience,
        verbose=verbose,
        mode="max"
    )
    
    reduce_lr = ReduceLROnPlateau(
        monitor="val_macro_f1_score",
        patience=destin_cfg.reduce_lr_patience,
        min_lr=destin_cfg.min_lr,
        factor=destin_cfg.reduce_lr_factor,
        verbose=verbose,
        mode="max"
    )

    return checkpoint, early, reduce_lr