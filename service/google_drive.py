import os
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

# Phạm vi quyền (scope) cần cấp cho ứng dụng
SCOPES = ['https://www.googleapis.com/auth/drive.file']  # Chỉ quyền tạo & chỉnh sửa file do app tạo
CREDENTIALS = r"C:\Users\tranh\MyProjects\VN-Instruments-Classifier\auth\credentials.json"
TOKEN = r"C:\Users\tranh\MyProjects\VN-Instruments-Classifier\auth\token.json"

def get_service():
    creds = None

    # Nếu đã đăng nhập trước đó, load token từ file token.json
    if os.path.exists(TOKEN):
        creds = Credentials.from_authorized_user_file(TOKEN, SCOPES)

    # Nếu chưa có token hoặc token hết hạn → yêu cầu đăng nhập lại
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                CREDENTIALS, SCOPES)  # credentials.json tải từ Google Cloud
            creds = flow.run_local_server(port=0)
        # Lưu token để lần sau không cần đăng nhập
        with open(TOKEN, 'w') as token:
            token.write(creds.to_json())

    # Tạo service để thao tác với Google Drive
    return build('drive', 'v3', credentials=creds)

def upload_file(service, file_path: str):
    # Đường dẫn file cần upload
    file_metadata = {'name': os.path.basename(file_path)}
    media = MediaFileUpload(file_path, resumable=True)

    # Upload file lên Google Drive
    file = service.files().create(
        body=file_metadata, 
        media_body=media, 
        fields='id, webViewLink, webContentLink'
        ).execute()
    
    print(f"[+] Đã upload: {file_path}")
    print("    File ID:", file.get('id'))
    print("    Link xem:", file.get('webViewLink'))
    print("    Link tải:", file.get('webContentLink'))
    print("-" * 50)

def upload_files(service, file_paths: list[str]):
    """Upload nhiều file."""
    for path in file_paths:
        if os.path.exists(path):
            upload_file(service, path)
        else:
            print(f"[!] Không tìm thấy file: {path}")

if __name__ == '__main__':
    upload_file(get_service(), [r"C:\Users\tranh\Downloads\cycle.svg"])
