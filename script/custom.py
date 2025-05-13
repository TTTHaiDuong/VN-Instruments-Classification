from tensorflow.keras.callbacks import Callback
import numpy as np
import os

class DualCheckpoint(Callback):
    def __init__(self, filepath1, filepath2,
                 monitor='val_accuracy', mode='max',
                 save_best_only=True, save_weights_only=False, verbose=1):
        super().__init__()
        self.filepath1 = filepath1
        self.filepath2 = filepath2
        self.monitor = monitor
        self.save_best_only = save_best_only
        self.save_weights_only = save_weights_only  # Chỉ áp dụng cho filepath1
        self.verbose = verbose

        os.makedirs(os.path.dirname(filepath1), exist_ok=True)
        os.makedirs(os.path.dirname(filepath2), exist_ok=True)

        if mode == 'min':
            self.monitor_op = np.less
            self.best = np.inf
        elif mode == 'max':
            self.monitor_op = np.greater
            self.best = -np.inf
        else:
            raise ValueError("mode must be 'min' or 'max'")

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        current = logs.get(self.monitor)
        if current is None:
            if self.verbose:
                print(f"Monitor '{self.monitor}' not available in logs.")
            return

        if self.save_best_only:
            if self.monitor_op(current, self.best):
                if self.verbose:
                    print(f"\nEpoch {epoch+1}: {self.monitor} improved from {self.best:.5f} to {current:.5f}. Saving model.")
                self.best = current
                self._save_model(epoch=epoch, logs=logs)
            elif self.verbose:
                print(f"\nEpoch {epoch+1}: {self.monitor} did not improve from {self.best:.5f}")
        else:
            if self.verbose:
                print(f"\nEpoch {epoch+1}: Saving model (regardless of {self.monitor})")
            self._save_model(epoch=epoch, logs=logs)


    def _save_model(self, epoch=None, logs=None):
        # Format tên file nếu có định dạng động
        if epoch is not None and logs is not None:
            filepath1 = self.filepath1.format(epoch=epoch + 1, **logs)
            filepath2 = self.filepath2.format(epoch=epoch + 1, **logs)
        else:
            filepath1 = self.filepath1
            filepath2 = self.filepath2

        # Lưu weights hoặc full model theo cấu hình
        if self.save_weights_only:
            self.model.save_weights(filepath1)
        else:
            self.model.save(filepath1)

        # Luôn lưu full model vào filepath2
        self.model.save(filepath2)
