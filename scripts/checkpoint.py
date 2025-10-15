import numpy as np
import re
from tensorflow.keras.callbacks import Callback
from datetime import datetime
from pathlib import Path


class DualCheckpoint(Callback):
    """
    TensorFlow's built-in Checkpoint only allows saving one checkpoint at a time.
    This custom checkpoint callback saves two versions of a checkpoint in two different locations:
        - One version is for historical records (with timestamp in filename),
        - The other version is the "best model" checkpoint that gets overwritten whenever a better model is found.
    """
    
    def __init__(
        self, 
        checkpoint_fpath: str | None = None, 
        best_fpath: str | None = None,
        monitor = "val_macro_f1", # tất cả các lớp học đều với nhau
        mode = "max",
        metric_pattern: str | None = None,
        verbose = False
    ):
        super().__init__()

        if not checkpoint_fpath and not best_fpath:
            raise ValueError("`checkpoint_fpath` and `best_fpath`, both are not provided.")

        self.checkpoint_fpath = Path(checkpoint_fpath) if checkpoint_fpath else None
        self.best_fpath = Path(best_fpath) if best_fpath else None
        self.monitor = monitor
        self.verbose = verbose

        if mode == "min":
            self.monitor_op = np.less
            self.best = np.inf
        elif mode == "max":
            self.monitor_op = np.greater
            self.best = -np.inf
        else:
            raise ValueError("mode must be 'min' or 'max'")        

        self.metric_pattern = (
            metric_pattern
            or rf"_{re.escape(self.monitor)}_([0-9]*\.?[0-9]+)(?:_\d{{8}}_\d{{6}})?(?:\.weights)?\.h5$"
        )
        
        if self.checkpoint_fpath:
            self.checkpoint_fpath.parent.mkdir(parents=True, exist_ok=True)
            self._update_best_from_existing(self.checkpoint_fpath.parent)
        
        if self.best_fpath:
            self.best_fpath.parent.mkdir(parents=True, exist_ok=True)

    
    def _update_best_from_existing(self, history_dpath: Path):
        """Scan checkpoint directory to find the best recorded monitor value."""
        # Find all checkpoint files matching the naming pattern
        if not history_dpath.exists():
            return
        
        pattern = re.compile(self.metric_pattern)
        best_value = self.best

        fnames = [f.name for f in history_dpath.iterdir() if f.is_file()]
        for f in fnames:
            match = pattern.search(f)
            if match:
                # Extract the monitored metric value from filename, convert to float
                value = float(match.group(1))
                if self.monitor_op(value, best_value):
                    best_value = value

        if best_value != self.best:
            self.best = best_value
            if self.verbose:
                print(f"Initialized best {self.monitor} from existing checkpoints: {self.best:.4f}")


    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        current = logs.get(self.monitor)
        if current is None:
            raise ValueError(f"Monitor '{self.monitor}' not available in logs.")

        # Save model only if monitored metric improves
        if self.monitor_op(current, self.best):
            if self.verbose:
                print(f"\nEpoch {epoch+1}: {self.monitor} improved from {self.best:.4f} to {current:.4f}. Saving model.")
            self.best = current
            self._save_model(epoch=epoch, logs=logs)

        elif self.verbose:
            print(f"\nEpoch {epoch+1}: {self.monitor} did not improve from {self.best:.4f}")


    def _render_fname(self, fpath: Path, epoch: int, logs: dict) -> str:
        """Create a filename with metric name, value, and timestamp."""
        score = float(logs.get(self.monitor, 0))
        timestamp = datetime.now()

        format_args = {
            "epoch": epoch + 1,
            "metric_name": self.monitor,
            "metric_score": score,
            "timestamp": timestamp,
            **logs
        }

        try:
            filename = fpath.name.format(**format_args)
        except KeyError as e:
            raise ValueError(f"Tên file chứa placeholder không hợp lệ: {e}")
        return str(fpath.parent / filename)


    def _save_model(self, epoch, logs):
        if not hasattr(self, "model") or self.model is None:
            raise RuntimeError("Callback chưa được gắn với model.")
        
        checkpoint_fpath = self._render_fname(self.checkpoint_fpath, epoch, logs) if self.checkpoint_fpath else None
        best_fpath = self._render_fname(self.best_fpath, epoch, logs) if self.best_fpath else None

        def _save(path: str):
            if path.endswith(".weights.h5"):
                self.model.save_weights(path)
            elif path.endswith(".h5"):
                self.model.save(path)
            else:
                raise ValueError(f"Định dạng file không hợp lệ: {path}")
            if self.verbose:
                print(f"Đã lưu: {path}")

        if checkpoint_fpath:
            _save(checkpoint_fpath)
        if best_fpath:
            _save(best_fpath)