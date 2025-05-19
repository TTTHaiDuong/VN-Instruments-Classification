import numpy as np
import os
import re
import sys
from tensorflow.keras.callbacks import Callback
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import *



# Checkpoint có sẵn của Tensorflow chỉ cho phép lưu một checkpoint tại một thời điểm.
# Checkpoint tuỳ chỉnh này giúp lưu hai phiên bản của một checkpoint ở hai nơi khác nhau.
# Một phiên bản để ghi nhận lịch sử, còn lại để lưu mô hình tốt nhất nên sau này nó có thể bị thay thế.
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

        self._update_best_from_existing_checkpoints()



    def _update_best_from_existing_checkpoints(self):
        """Kiểm tra thư mục checkpoints để tìm giá trị tốt nhất của monitor."""
        checkpoint_dir = os.path.dirname(self.filepath1)
        if not os.path.exists(checkpoint_dir):
            return

        # Tìm tất cả các file checkpoint trong thư mục
        pattern = re.compile(CHECKPOINT_REGEX)
        best_value = self.best
        for filename in os.listdir(checkpoint_dir):
            match = pattern.search(filename)
            if match:
                value = float(match.group(1)) / 10000  # Chuyển thành float (ví dụ, 0.8000)
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
            if self.verbose:
                print(f"Monitor '{self.monitor}' not available in logs.")
            return

        if self.save_best_only:
            if self.monitor_op(current, self.best):
                if self.verbose:
                    print(f"\nEpoch {epoch+1}: {self.monitor} improved from {self.best:.4f} to {current:.4f}. Saving model.")
                self.best = current
                self._save_model(epoch=epoch, logs=logs)
            elif self.verbose:
                print(f"\nEpoch {epoch+1}: {self.monitor} did not improve from {self.best:.4f}")
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