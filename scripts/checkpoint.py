import numpy as np
import os
import re
from datetime import datetime
from tensorflow.keras.callbacks import Callback
from config.general import *



# TensorFlow's built-in Checkpoint only allows saving one checkpoint at a time.
# This custom checkpoint callback saves two versions of a checkpoint in two different locations:
# - One version is for historical records (with timestamp in filename),
# - The other version is the "best model" checkpoint that gets overwritten whenever a better model is found.
class DualCheckpoint(Callback):
    
    def __init__(self, filepath1, filepath2,
                 monitor='val_accuracy', mode='max',
                 save_best_only=True, save_weights_only=False, verbose=1):
        super().__init__()
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        # Filepath for checkpoint with timestamp, stored in checkpoint directory (historical records)
        self.filepath1 = self._add_timestamp_to_filepath(filepath1)
        # Filepath for best model checkpoint, always overwritten on improvement
        self.filepath2 = filepath2     
        self.monitor = monitor
        self.save_best_only = save_best_only
        self.save_weights_only = save_weights_only  # Applies only to filepath1
        self.verbose = verbose

        os.makedirs(os.path.dirname(filepath1), exist_ok=True)
        os.makedirs(os.path.dirname(filepath2), exist_ok=True)

        # Set comparison operator for monitoring metric
        if mode == 'min':
            self.monitor_op = np.less
            self.best = np.inf
        elif mode == 'max':
            self.monitor_op = np.greater
            self.best = -np.inf
        else:
            raise ValueError("mode must be 'min' or 'max'")

        self._update_best_from_existing_checkpoints()



    def _add_timestamp_to_filepath(self, filepath):
        """Add timestamp to the filename before the extension."""
        pattern = r"(.+?)(\.weights\.h5|\.h5)$"
        match = re.match(pattern, filepath)
        if not match:
            raise ValueError(f"Filepath must end with .weights.h5 or .h5: {filepath}")
    
        base, ext = match.groups()
        return f"{base}_{self.timestamp}{ext}"

    
    def _update_best_from_existing_checkpoints(self):
        """Scan checkpoint directory to find the best recorded monitor value."""
        checkpoint_dir = os.path.dirname(self.filepath1)
        if not os.path.exists(checkpoint_dir):
            return

        # Find all checkpoint files matching the naming pattern
        pattern = re.compile(CHECKPOINT_REGEX)
        best_value = self.best
        for filename in os.listdir(checkpoint_dir):
            match = pattern.search(filename)
            if match:
                # Extract the monitored metric value from filename, convert to float
                value = float(match.group(1)) / 10000
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
            # Save model only if monitored metric improves
            if self.monitor_op(current, self.best):
                if self.verbose:
                    print(f"\nEpoch {epoch+1}: {self.monitor} improved from {self.best:.4f} to {current:.4f}. Saving model.")
                self.best = current
                self._save_model(epoch=epoch, logs=logs)
            elif self.verbose:
                print(f"\nEpoch {epoch+1}: {self.monitor} did not improve from {self.best:.4f}")
        else:
            # Save model at every epoch regardless of monitored metric
            if self.verbose:
                print(f"\nEpoch {epoch+1}: Saving model (regardless of {self.monitor})")
            self._save_model(epoch=epoch, logs=logs)



    def _save_model(self, epoch=None, logs=None):
        # Format filenames if they include dynamic formatting fields
        if epoch is not None and logs is not None:
            filepath1 = self.filepath1.format(epoch=epoch + 1, **logs)
            filepath2 = self.filepath2.format(epoch=epoch + 1, **logs)
        else:
            filepath1 = self.filepath1
            filepath2 = self.filepath2

        # Save weights or full model according to configuration
        if self.save_weights_only:
            self.model.save_weights(filepath1)
        else:
            self.model.save(filepath1)

        # Always save full model to filepath2 (best model checkpoint)
        self.model.save(filepath2)