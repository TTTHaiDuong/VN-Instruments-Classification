DATASET_PATH = "dataset"
CHECKPOINT_PATH = "checkpoints"


# Danh sách lớp để dự đoán
class_list = {0: "danbau", 1: "dannhi", 2: "dantranh", 3: "sao"}



# Tham số xử lý âm thanh
SR = 22050 # Tần số lấy mẫu khi số hoá tín hiệu analog (sampling rate)



# Cấu hình tăng cường dữ liệu
RESCALE = 1./255          # Hệ số điều chỉnh lại giá trị pixel về khoảng [0, 1]
WIDTH_SHIFT_RANGE = 0.05  # Dịch chuyển chiều rộng
HEIGHT_SHIFT_RANGE = 0.05 # Dịch chuyển chiều cao
ZOOM_RANGE = 0.025        # Độ phóng to / thu nhỏ



# Tham số đầu vào của mô hình CNN
INPUT_SHAPE = (128, 128) # Kích thước Mel-spectrogram (mel bands, khung thời gian) 

def get_num_classes():
    """Nhận số lượng lớp dự đoán."""
    return len(class_list)


# Cấu hình model
OPTIMIZER = "adam"               # rmsrop, sgd
METRICS = ["accuracy"]           # Chỉ số đo chính xác (accuracy) trong quá trình huấn luyện.
LOSS ='categorical_crossentropy' # Hàm mất mát cho phân loại đa lớp.
BATCH_SIZE = 32                  # Kích thước batch cho huấn luyện và validation.
EPOCHS = 5                       # Số lượng epoch cho huấn luyện.
VALIDATION_BATCH_SIZE = 32       # Kích thước batch cho validation.



# Cấu hình checkpoint
CHECKPOINT_MONITOR = 'val_accuracy' # Tham số theo dõi. Lưu mô hình khi val_accuracy (tỉ lệ validation chính xác) tốt nhất.



# Cấu hình EarlyStopping
EARLY_MONITOR = 'val_loss' # Tham số theo dõi. Dừng huấn luyện khi val_loss (tỉ lệ validation mất mát) không cải thiện.
PATIENCE = 5               # Số epoch không cải thiện trước khi dừng huấn luyện.
VERBOSE = 1                # Độ chi tiết thông báo. 