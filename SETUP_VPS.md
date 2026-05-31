# VPS Setup Guide (Đơn Giản)

## Trên VPS, chạy 5 lệnh này:

```bash
# 1. Clone code
git clone https://github.com/your-username/cn1-ai-comparep-model.git
cd cn1-ai-comparep-model

# 2. Setup Python environment
python3 -m venv venv
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Upload model files (chạy từ máy local - KHÔNG chạy trên VPS)
# scp -r runs/ user@your-vps-ip:/path/to/cn1-ai-comparep-model/

# 5. Chạy app
python3 web_app.py
```

App sẽ chạy tại: `http://your-vps-ip:5000`

---

## Nếu muốn auto-start khi VPS restart

**Cài thêm gunicorn + tạo systemd service:**

```bash
# Install production server
pip install gunicorn

# Create systemd service
sudo nano /etc/systemd/system/ai-detector.service
```

**Paste vào file:**
```ini
[Unit]
Description=AI Image Detector
After=network.target

[Service]
User=your-user
WorkingDirectory=/home/your-user/cn1-ai-comparep-model
Environment="PATH=/home/your-user/cn1-ai-comparep-model/venv/bin"
ExecStart=/home/your-user/cn1-ai-comparep-model/venv/bin/gunicorn --bind 0.0.0.0:5000 web_app:app
Restart=always

[Install]
WantedBy=multi-user.target
```

```bash
# Enable & start
sudo systemctl daemon-reload
sudo systemctl enable ai-detector
sudo systemctl start ai-detector
```

---

## Important Notes

- ⚠️ **Models folder (`runs/`)** - Không trong git, upload bằng SCP
- 🔧 **web_app.py** - Sửa `debug=False, host=0.0.0.0` ✅ Đã sửa
- 🐍 **Python 3.8+** cần thiết
- 📦 **requirements.txt** - Chứa tất cả dependencies ✅ Đầy đủ

---

**Xong! Lên VPS chỉ cần clone + pip install + copy models là chạy được.**
