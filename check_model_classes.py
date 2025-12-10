"""
Script to check classes in a YOLO .pt model file
Run this to see what classes your model detects
"""
import sys
import os

# Add safety patches for PyTorch
import torch
os.environ['PYTORCH_WEIGHTS_ONLY'] = 'False'
os.environ['TORCH_DISABLE_WEIGHTS_ONLY'] = '1'

# Monkey patch torch.load
original_load = torch.load
def safe_load(*args, **kwargs):
    kwargs['weights_only'] = False
    return original_load(*args, **kwargs)
torch.load = safe_load

try:
    from ultralytics import YOLO
except ImportError:
    print("❌ Error: ultralytics not installed. Run: pip install ultralytics")
    sys.exit(1)

def check_model_classes(model_path):
    """Check and display classes in a YOLO model"""
    
    if not os.path.exists(model_path):
        print(f"❌ Error: Model file not found: {model_path}")
        print(f"\nAvailable models in models/ directory:")
        if os.path.exists('models'):
            for f in os.listdir('models'):
                if f.endswith('.pt'):
                    print(f"  - models/{f}")
        return
    
    print(f"\n{'='*60}")
    print(f"  Checking Model: {model_path}")
    print(f"{'='*60}\n")
    
    try:
        # Load the model
        print("📦 Loading model...")
        model = YOLO(model_path)
        
        # Get class names
        if hasattr(model, 'names'):
            class_names = model.names
            num_classes = len(class_names)
            
            print(f"✅ Model loaded successfully!")
            print(f"\n📊 Total Classes: {num_classes}\n")
            print(f"{'Class ID':<12} {'Class Name':<30}")
            print(f"{'-'*12} {'-'*30}")
            
            # Display all classes
            for class_id, class_name in sorted(class_names.items()):
                print(f"{class_id:<12} {class_name:<30}")
            
            # Find Person class if it exists
            person_classes = {k: v for k, v in class_names.items() if 'person' in v.lower() or 'people' in v.lower()}
            if person_classes:
                print(f"\n👤 Person-related classes found:")
                for class_id, class_name in person_classes.items():
                    print(f"   ID {class_id}: {class_name}")
            else:
                print(f"\n⚠️  No 'Person' class found. Check the class names above.")
            
            # Additional model info
            print(f"\n📋 Model Information:")
            print(f"   Task: {getattr(model.task, 'name', 'detect') if hasattr(model, 'task') else 'detect'}")
            
            if hasattr(model, 'model'):
                if hasattr(model.model, 'yaml'):
                    yaml_path = model.model.yaml
                    if yaml_path and os.path.exists(yaml_path):
                        print(f"   Config: {yaml_path}")
            
        else:
            print("⚠️  Could not retrieve class names from model")
            
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    if len(sys.argv) > 1:
        model_path = sys.argv[1]
    else:
        # Default: check best.pt, or ask user
        model_path = 'models/best.pt'
        if not os.path.exists(model_path):
            print("📝 Usage: python check_model_classes.py <path_to_model.pt>")
            print("\nExample: python check_model_classes.py models/your_new_model.pt")
            print("\nOr specify the model path:")
            model_path = input("Enter model path (e.g., models/your_model.pt): ").strip()
            if not model_path:
                print("❌ No model path provided")
                sys.exit(1)
    
    check_model_classes(model_path)
    
    print(f"\n{'='*60}")
    print("✅ Done! Use this information to update your module configurations.")
    print(f"{'='*60}\n")


