# Vietnamese-Musical-Instruments-Classification

Làm việc với mel-spectrogram
Hiển thị mel-spectrogram
`python mel_spec.py -d -ip "đường\dẫn\audio\file"`
Lưu 1 file hoặc thư mục âm thanh thành mel-spectrogram (png)
`python mel_spec.py -s -ip "đường\dẫn\nguồn" -op "đường\dẫn\đích"`

Quản lý file, dataset
Tìm các file trùng lập trong thư mục, hỗ trợ audio (wav, mp3), image (png, jpeg, jpg)
`python file_utils.py -dup "đường\dẫn\thư\mục"`