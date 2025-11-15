# 🚀 Digital Inspector - Quick Start Guide

## ✅ Status: READY TO USE!

**Application is running on:** http://localhost:5002

---

## 🔐 Login Credentials

Use these demo accounts to log in:

| Username | Password | Role |
|----------|----------|------|
| `inspector` | `demo123` | Digital Inspector |
| `admin` | `admin123` | System Administrator |
| `user` | `user123` | Regular User |

---

## 📱 How to Use

### 1. **Login**
- Open: http://localhost:5002
- Enter: `inspector` / `demo123`
- Click: **Sign In**

### 2. **Upload Document**
- Click: **Upload Document** (sidebar)
- Drag & drop PDF or image file
- Wait for processing (~5-7 seconds per page)
- View results automatically

### 3. **View Documents**
- Click: **Documents** (sidebar)
- See all uploaded documents with statistics
- Click **View** to see details
- Download original or annotated versions

### 4. **Analytics**
- Click: **Reports** (sidebar)
- View charts and statistics
- See detection breakdown

---

## 🎨 Detection Color Codes

When viewing annotated images:

- **🟢 Green** = Signatures
- **🟢 Light Green** = Stamps  
- **🟡 Yellow** = QR Codes (valid)
- **🟠 Orange** = QR Codes (invalid)

---

## 📁 Project Files

```
hacknu/
├── app_inspector.py          # NEW: Main Digital Inspector app
├── database.py                # NEW: Database management
├── digital_inspector.db       # NEW: SQLite database
├── app.py                     # OLD: Simple version
│
├── preprocessing.py           # Image preprocessing
├── postprocessing.py          # Detection post-processing
├── best.pt                    # YOLOv8 model
│
├── templates/                 # NEW: HTML templates
│   ├── base.html             # Base layout with sidebar
│   ├── login.html            # Login page
│   ├── dashboard.html        # Dashboard
│   ├── upload.html           # Upload page
│   ├── documents.html        # Documents list
│   ├── document_details.html # Document details
│   └── reports.html          # Analytics
│
├── uploads/                  # Uploaded documents
├── annotated/                # Annotated images
│
└── README_DIGITAL_INSPECTOR.md  # Full documentation
```

---

## 🔄 Restart Application

If needed, restart the application:

```bash
# Stop all Python apps
pkill -9 -f "python.*app"

# Start Digital Inspector
cd /Users/aidyn/Downloads/hacknu
source venv/bin/activate
python app_inspector.py
```

---

## 📊 Features

### ✅ Completed Features

- [x] User authentication (login/logout)
- [x] Dashboard with statistics
- [x] Document upload (PDF/images)
- [x] YOLOv8 detection (signatures, stamps, QR)
- [x] SQLite database storage
- [x] Annotated image generation
- [x] Color-coded bounding boxes
- [x] Document management
- [x] Download original/annotated files
- [x] Reports & analytics
- [x] Charts (Chart.js)
- [x] CRM-style UI with sidebar
- [x] Green/light green/white color scheme

### 🎯 What You Can Do

1. **Upload** construction documents
2. **Detect** signatures, stamps, QR codes automatically
3. **View** annotated images with bounding boxes
4. **Download** processed results
5. **Analyze** statistics and trends
6. **Filter** documents by detection type
7. **Track** confidence scores

---

## 🌐 Access URLs

| Feature | URL |
|---------|-----|
| **Login** | http://localhost:5002/login |
| **Dashboard** | http://localhost:5002/ |
| **Upload** | http://localhost:5002/upload |
| **Documents** | http://localhost:5002/documents |
| **Reports** | http://localhost:5002/reports |

---

## 🔧 Configuration

### Current Settings:
- **Port**: 5002
- **DPI**: 250 (PDF conversion)
- **YOLO imgsz**: 960
- **Confidence threshold**: 0.15
- **Preprocessing**: FAST mode
- **Database**: SQLite (digital_inspector.db)
- **Max file size**: 50MB

### Adjust Settings:
Edit `app_inspector.py`:
```python
# Line ~44
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024

# Line ~269
yolo_detections = model(preprocessed_img, 
                       conf=0.15,    # Change threshold
                       imgsz=960)    # Change size
```

---

## 🐛 Troubleshooting

### Application not starting?
```bash
# Check if port is in use
lsof -i :5002

# Use different port
# Edit app_inspector.py, line ~437:
app.run(debug=True, host='0.0.0.0', port=5003)
```

### Database errors?
```bash
# Reset database
rm digital_inspector.db
python database.py
```

### Model not found?
```bash
# Check model exists
ls -lh best.pt

# Should show: 21MB file
```

---

## 📖 Documentation

- **Full README**: `README_DIGITAL_INSPECTOR.md`
- **Speed optimization**: `SPEED_OPTIMIZATION.md`
- **Visualization fix**: `VISUALIZATION_FIX.md`
- **Filter thresholds**: `FILTER_THRESHOLDS_FIX.md`
- **Text class filter**: `TEXT_CLASS_FILTER.md`

---

## 🎉 You're All Set!

**Digital Inspector is running and ready to use!**

1. Open: http://localhost:5002
2. Login: `inspector` / `demo123`
3. Upload your first document
4. See the magic happen! ✨

---

**Need help?** Check `README_DIGITAL_INSPECTOR.md` for detailed documentation.

**Happy Inspecting! 🛡️**

