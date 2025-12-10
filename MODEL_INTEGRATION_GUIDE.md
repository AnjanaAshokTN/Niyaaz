# How to Integrate a New .pt Model File

## Current Model Usage

Your application currently uses these models:

1. **`models/best.pt`** - Used by:
   - QueueMonitor (person_class_id=12)
   - DressCodeMonitoring
   - CashDetection
   - SmokingDetection
   - CrowdDetection (person_class_id=12)

2. **`models/yolo11n.pt`** - Used by:
   - QueueMonitor (optional, if configured)

## Steps to Add Your New .pt File

### Step 1: Place the File
Copy your new `.pt` file to the `models/` directory:
```
models/
  ├── best.pt (existing)
  ├── yolo11n.pt (existing)
  └── your_new_model.pt (NEW - your file)
```

### Step 2: Determine Model Classes
You need to know:
- **What classes does your model detect?**
- **What is the Person class ID?** (usually 0 for COCO models, or 12 for custom models)

### Step 3: Update Module Code

#### Option A: Replace `best.pt` (if it's an updated version)
If your new model replaces `best.pt`, just replace the file:
```bash
# Backup old model
mv models/best.pt models/best.pt.backup

# Copy your new model
cp your_new_model.pt models/best.pt
```

#### Option B: Use New Model for Specific Modules
If you want to use the new model for specific modules, update the code:

**For QueueMonitor:**
Edit `modules/queue_monitor.py` line ~73:
```python
self.detector = YOLODetector(
    model_path='models/your_new_model.pt',  # Change here
    confidence_threshold=0.4,
    img_size=640,
    person_class_id=12  # Update if different
)
```

**For DressCodeMonitoring:**
Edit `modules/dress_code_monitoring.py` line ~45:
```python
self.model_weight = "models/your_new_model.pt"  # Change here
```

**For CashDetection:**
Edit `modules/cash_detection.py` - find where model is loaded

**For SmokingDetection:**
Edit `modules/smoking_detection.py` - find where model is loaded

**For CrowdDetection:**
Edit `modules/crowd_detection.py` line ~37:
```python
self.detector = YOLODetector(
    model_path='models/your_new_model.pt',  # Change here
    confidence_threshold=0.4,
    img_size=640,
    person_class_id=12  # Update if different
)
```

## Quick Questions to Help You:

1. **What's the filename of your new .pt file?**
2. **Which modules should use it?** (All? Specific ones?)
3. **What classes does it detect?** (List them)
4. **What is the Person class ID?** (0, 12, or something else?)

## Testing Your New Model

After updating, test with:
```python
from ultralytics import YOLO

model = YOLO('models/your_new_model.pt')
print(model.names)  # Shows all class names and IDs
```

This will show you the class mapping so you can set the correct `person_class_id`.


