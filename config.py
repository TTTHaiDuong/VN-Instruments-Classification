import tensorflow as tf
from script.f1_score import MacroF1Score



# Các đường dẫn
AUG_DATASET = "aug_dataset"
AUG_TRAIN = "aug_dataset/train"
AUG_VAL = "aug_dataset/val"
AUG_TEST = "aug_dataset/test"

DATASET_PATH = "dataset"
TRAIN_DATA = "dataset/train"
VAL_DATA = "dataset/val"
TEST_DATA = "dataset/test"

CHECKPOINT_PATH = "checkpoint"
BEST_MODEL = "bestmodel"

TRAIN_HISTORY_PATH = "train_history"



# Danh sách lớp để dự đoán
CLASS_NAMES = ["Đàn bầu", "Đàn nhị", "Đàn tranh", "Sáo"] # Tên các lớp nhạc cụ
CLASS_LABELS = ["danbau", "dannhi", "dantranh", "sao"]



# Tham số xử lý khi lấy file âm thanh
SR = 22050       # Tần số lấy mẫu khi số hoá tín hiệu analog (sampling rate)
DURATION = 5     # Thời gian tối đa của file âm thanh (giây)
N_MELS = 128     # Số lượng Mel bands
N_FFT = 2048
HOP_LENGTH = 512



# Lớp trọng số. Sử dụng khi có sự thiên vị dữ liệu
CLASS_WEIGHT_DICT = {
    0: 1.2,
    1: 2.0,
    2: 1.0,
    3: 1.5
}



# Cấu hình tăng cường dữ liệu hình ảnh
RESCALE = 1./255                # Hệ số điều chỉnh lại giá trị pixel về khoảng [0, 1]
BRIGHTNESS_RANGE = (0.95, 1.05) # Làm mờ
WIDTH_SHIFT_RANGE = 0.02        # Dịch chuyển chiều rộng
HEIGHT_SHIFT_RANGE = 0.02       # Dịch chuyển chiều cao
ZOOM_RANGE = 0.025              # Độ phóng to / thu nhỏ



# Cấu hình tăng cường dữ liệu âm thanh
# Cấu hình tăng cường âm thanh mặc định.
AUDIO_AUGMENTATION = {
    "pitch_shift": (-1, 1),
    "time_stretch": (0.9, 1.1),
    "add_noise": (0, 0.005)
}
# Tăng cường dữ liệu âm thanh cho từng lớp.
# Phải sử dụng save_augment_audios_to_mel_spectrogram trong script\augment.py
# để tạo thủ công aug_dataset.
AUDIO_CLASS_AUGMENTATION = {
    "danbau": {
        "time_stretch": (0.98, 1.02)
    },
    "dannhi": {
        "pitch_shift": (-0.5, 0.5),
        "add_noise": (0, 0.01),
        "time_stretch": (0.95, 1.05)
    },
    "dantranh": {
        "add_noise": (0, 0.005)
    },
    "sao": {
        "pitch_shift": (-0.5, 0.5),
        "add_noise": (0, 0.01),
        "time_stretch": (0.95, 1.05)
    }
}



# Tham số đầu vào của mô hình CNN
INPUT_SHAPE = (128, 128, 1) # Kích thước Mel-spectrogram (mel bands, khung thời gian) 

def get_num_classes():
    """Nhận số lượng lớp dự đoán."""
    return len(CLASS_LABELS)



# Cấu hình model
OPTIMIZER = tf.keras.optimizers.Adam(learning_rate=0.001)
# Các chỉ số đo, chỉ số chính xác (accuracy) trong quá trình huấn luyện.
METRICS = ["accuracy", MacroF1Score(name="macro_f1_score", num_classes=4)]
LOSS = "categorical_crossentropy" # Hàm mất mát cho phân loại đa lớp.
BATCH_SIZE = 64                   # Hoặc 64. Kích thước batch cho huấn luyện.
EPOCHS = 50                       # Số lượng epoch cho huấn luyện.
VALIDATION_BATCH_SIZE = 64        # Hoặc 128. Kích thước batch cho validation.



# Cấu hình checkpoint
CHECKPOINT_MONITOR = "val_accuracy"                        # Tham số theo dõi. Lưu mô hình khi val_accuracy (tỉ lệ validation chính xác) tốt nhất.
CHECKPOINT_REGEX = r"model1_\d+_(\d+\.\d{4})\.weights\.h5" # Biểu thức chính quy để kiểm tra tên các file checkpoint.



# Cấu hình EarlyStopping
EARLY_MONITOR = "val_loss" # Tham số theo dõi. Dừng huấn luyện khi val_loss (tỉ lệ validation mất mát) không cải thiện.
EARLY_PATIENCE = 15        # Số epoch không cải thiện trước khi dừng huấn luyện.
VERBOSE = 1                # Độ chi tiết thông báo. 



# Cấu hình ReduceLROnPlateau, giảm tốc độ học khi không cải thiện chỉ số
REDUCE_LR_PATIENCE = 7 # Số epoch nếu không cải thiện thì sẽ giảm tốc độ học lại.
MIN_LR = 1e-6          # Min learning rate.
LR_FACTOR = 0.5        # Hệ số giảm learning rate.
REDUCE_LR_VERBOSE = 1