import requests
import pandas as pd

# 1. URL API VN30
url = "https://iboard-query.ssi.com.vn/stock/group/VN30"

# 2. Headers (mimic browser, tránh bị chặn)
headers = {
    "Accept": "application/json, text/plain, */*",
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36",
    "Referer": "https://iboard.ssi.com.vn/",
    "Origin": "https://iboard.ssi.com.vn"
}

# 3. Gửi GET request
response = requests.get(url, headers=headers)

# 4. Kiểm tra status
if response.status_code == 200:
    data = response.json()  # dữ liệu trả về dạng JSON
    print("Đã lấy dữ liệu thành công!")
else:
    print("Lỗi khi lấy dữ liệu:", response.status_code)

# 5. Xem thử dữ liệu
import json
print(json.dumps(data, indent=2))  # in đẹp dữ liệu JSON
