import tensorflow as tf



# Tạo thêm chỉ số đánh giá f1-score sử dụng cho metrics khi compile model.
# Vì Tensorflow không có sẵn chỉ số này. Sử dụng cho các callback như Checkpoint, Early Stopping, ReduceLROnPlateau
# để chọn ra mô hình có f1-score tối ưu nhất.
class MacroF1Score(tf.keras.metrics.Metric):
    
    def __init__(self, num_classes, name='macro_f1_score', **kwargs):
        super().__init__(name=name, **kwargs)
        self.num_classes = num_classes
        self.true_positives = self.add_weight(name='tp', shape=(num_classes,), initializer='zeros')
        self.false_positives = self.add_weight(name='fp', shape=(num_classes,), initializer='zeros')
        self.false_negatives = self.add_weight(name='fn', shape=(num_classes,), initializer='zeros')



    def update_state(self, y_true, y_pred, sample_weight=None):
        # Chuyển y_true thành nhãn lớp
        y_true = tf.cast(y_true, tf.float32)
        if y_true.shape[-1] == self.num_classes:  # Nếu y_true là one-hot
            y_true = tf.argmax(y_true, axis=-1)
        y_true = tf.cast(y_true, tf.int32)
        
        # Chuyển y_pred thành nhãn dự đoán
        y_pred = tf.argmax(y_pred, axis=-1)
        y_pred = tf.cast(y_pred, tf.int32)

        # Tạo ma trận nhầm lẫn
        conf_matrix = tf.math.confusion_matrix(
            labels=y_true,
            predictions=y_pred,
            num_classes=self.num_classes,
            dtype=tf.float32
        )

        # True Positives: Đường chéo
        tp = tf.linalg.diag_part(conf_matrix)
        
        # False Positives: Tổng cột trừ TP
        fp = tf.reduce_sum(conf_matrix, axis=0) - tp
        
        # False Negatives: Tổng hàng trừ TP
        fn = tf.reduce_sum(conf_matrix, axis=1) - tp

        # Áp dụng sample_weight nếu có
        if sample_weight is not None:
            sample_weight = tf.cast(sample_weight, tf.float32)
            tp = tf.reduce_sum(tp * sample_weight)
            fp = tf.reduce_sum(fp * sample_weight)
            fn = tf.reduce_sum(fn * sample_weight)

        # Cập nhật trạng thái
        self.true_positives.assign_add(tp)
        self.false_positives.assign_add(fp)
        self.false_negatives.assign_add(fn)



    def result(self):
        precision = tf.where(
            (self.true_positives + self.false_positives) > 0,
            self.true_positives / (self.true_positives + self.false_positives + 1e-10),
            tf.zeros_like(self.true_positives)
        )
        recall = tf.where(
            (self.true_positives + self.false_negatives) > 0,
            self.true_positives / (self.true_positives + self.false_negatives + 1e-10),
            tf.zeros_like(self.true_positives)
        )
        f1 = tf.where(
            (precision + recall) > 0,
            2 * (precision * recall) / (precision + recall + 1e-10),
            tf.zeros_like(precision)
        )
        return tf.reduce_mean(f1)



    def reset_states(self):
        self.true_positives.assign(tf.zeros_like(self.true_positives))
        self.false_positives.assign(tf.zeros_like(self.false_positives))
        self.false_negatives.assign(tf.zeros_like(self.false_negatives))