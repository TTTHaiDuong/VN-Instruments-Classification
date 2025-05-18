# Các đường dẫn
DATASET_PATH = "dataset"
CHECKPOINT_PATH = "checkpoint"
TRAIN_DATA = "dataset/train"
VAL_DATA = "dataset/val"
TEST_DATA = "dataset/test"
BEST_MODEL = "bestmodel"


# Danh sách lớp để dự đoán
class_list = {0: "danbau", 1: "dannhi", 2: "dantranh", 3: "sao"}
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



# Cấu hình tăng cường dữ liệu
RESCALE = 1./255          # Hệ số điều chỉnh lại giá trị pixel về khoảng [0, 1]
WIDTH_SHIFT_RANGE = 0.02  # Dịch chuyển chiều rộng
HEIGHT_SHIFT_RANGE = 0.02 # Dịch chuyển chiều cao
ZOOM_RANGE = 0.025        # Độ phóng to / thu nhỏ



# Tham số đầu vào của mô hình CNN
INPUT_SHAPE = (128, 128, 1) # Kích thước Mel-spectrogram (mel bands, khung thời gian) 

def get_num_classes():
    """Nhận số lượng lớp dự đoán."""
    return len(class_list)



# Cấu hình model
OPTIMIZER = "adam"                # rmsrop, sgd
METRICS = ["accuracy"]            # Các chỉ số đo, chỉ số chính xác (accuracy) trong quá trình huấn luyện.
LOSS = "categorical_crossentropy" # Hàm mất mát cho phân loại đa lớp.
BATCH_SIZE = 32                   # Hoặc 64. Kích thước batch cho huấn luyện.
EPOCHS = 50                       # Số lượng epoch cho huấn luyện.
VALIDATION_BATCH_SIZE = 64        # Hoặc 128. Kích thước batch cho validation.



# Cấu hình checkpoint
CHECKPOINT_MONITOR = 'val_accuracy'                        # Tham số theo dõi. Lưu mô hình khi val_accuracy (tỉ lệ validation chính xác) tốt nhất.
CHECKPOINT_REGEX = r'model1_\d+_(\d+\.\d{4})\.weights\.h5' # Biểu thức chính quy để kiểm tra tên các file checkpoint.



# Cấu hình EarlyStopping
EARLY_MONITOR = 'val_loss' # Tham số theo dõi. Dừng huấn luyện khi val_loss (tỉ lệ validation mất mát) không cải thiện.
EARLY_PATIENCE = 10        # Số epoch không cải thiện trước khi dừng huấn luyện.
VERBOSE = 1                # Độ chi tiết thông báo. 



# Cấu hình ReduceLROnPlateau, giảm tốc độ học
REDUCE_LR_PATIENCE = 7 # Số epoch nếu không cải thiện thì sẽ giảm tốc độ học lại.
MIN_LR = 1e-6          # Min learning rate.
LR_FACTOR = 0.5        # Hệ số giảm learning rate.