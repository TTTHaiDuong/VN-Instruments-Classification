# Vietnamese-Musical-Instruments-Classification

Làm việc với mel-spectrogram
Hiển thị mel-spectrogram
`python mel_spec.py -d -ip "đường\dẫn\audio\file"`
Lưu 1 file hoặc thư mục âm thanh thành mel-spectrogram (png)
`python mel_spec.py -s -ip "đường\dẫn\nguồn" -op "đường\dẫn\đích"`



Quản lý file, hỗ trợ dataset

Đổi tên các file trong một thư mục
`python file_utils.py -r <tên tiền tố> <số bắt đầu> -i <đường\dẫn\thư\mục> -pd <padding cho số thứ tự>`

Cắt một file âm thanh hoặc tất cả các file trong thư mục thành các đoạn có độ dài cố định
`python file_utils.py -s <Độ dài (s)> -i <đường\dẫn\nguồn> -o <đường\dẫn\thư\mục\gốc> -pd <padding cho số thứ tự tên>`

Di chuyển các file trong thư mục sang vị trí khác
`python file_utils.py -m -i <đường\dẫn\dir\nguồn> -o <đường\dẫn\dir\đích>`

Đếm số lượng file trong một thư mục
`python file_utils.py -c -i <đường\dẫn\thư\mục>`

Chuyển định mp3 sang wav
`python file_utils.py -cv -i <đường\dẫn> -o <đường\dẫn\lưu>`

Cắt một file ở các mốc
`python file_utils.py -p <"1:00"> <"2:00"> <...> -i <đường\dẫn\file> -o <đường\dẫn\lưu>`

Tìm các file trùng lập trong thư mục, hỗ trợ audio (wav, mp3) image (png, jpeg, jpg)
`python file_utils.py -dup <đường\dẫn\thư\mục>`

Xoá các file trùng lập trong một thư mục dựa vào mã hash từng file
`python file_utils.py -rm <đường\dẫn\thư\mục>`



Dependencies
numpy
librosa
tensorflow
matplotlib
pydub
sklearn
seaborn