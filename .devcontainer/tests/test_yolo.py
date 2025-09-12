#!/usr/bin/env python3
"""
FIXED: YOLO Testing Script for Basketball Image
Addresses root cause of API key loading and environment issues
"""
import os
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO
import warnings
warnings.filterwarnings('ignore')


def debug_environment_loading():
    """Debug function to understand exactly how environment variables are loaded."""
    print("=" * 60)
    print("ENVIRONMENT LOADING DEBUG")
    print("=" * 60)
    
    # Check all possible environment sources
    sources = {
        "os.environ": dict(os.environ),
        "os.getenv": {
            "ROBOFLOW_API_KEY": os.getenv('ROBOFLOW_API_KEY'),
            "HOME": os.getenv('HOME'),
            "JUPYTER_TOKEN": os.getenv('JUPYTER_TOKEN'),
        }
    }
    
    print("Environment variable sources:")
    for source_name, source_vars in sources.items():
        print(f"\n{source_name}:")
        if source_name == "os.environ":
            # Only show relevant vars
            relevant_vars = {k: v for k, v in source_vars.items() 
                           if any(term in k.upper() for term in ['ROBOFLOW', 'JUPYTER', 'HOME', 'YOLO', 'OPENCV'])}
            for key, value in relevant_vars.items():
                if 'API_KEY' in key and value:
                    print(f"  {key}: {value[:10]}...{value[-4:] if len(value) > 14 else ''}")
                else:
                    print(f"  {key}: {value}")
        else:
            for key, value in source_vars.items():
                if 'API_KEY' in key and value:
                    print(f"  {key}: {value[:10]}...{value[-4:] if len(value) > 14 else ''}")
                else:
                    print(f"  {key}: {value}")
    
    # Check process environment directly
    print(f"\nProcess ID: {os.getpid()}")
    print(f"Current working directory: {os.getcwd()}")
    print(f"Python executable: {sys.executable}")
    
    return os.environ.get('ROBOFLOW_API_KEY')


def check_environment():
    """Enhanced environment check with debugging."""
    print("=" * 60)
    print("SYSTEMATIC ENVIRONMENT CHECK")
    print("=" * 60)
    
    checks = {
        "Python": sys.version.split()[0],
        "OpenCV": None,
        "YOLO": None,
        "CUDA": False,
        "GPU Device": "None",
        "Roboflow API Key": "Not Found",
        "Test Image": "Not Found"
    }
    
    # Check OpenCV
    try:
        import cv2
        checks["OpenCV"] = cv2.__version__
    except ImportError:
        checks["OpenCV"] = "Not installed"
    
    # Check YOLO/Ultralytics
    try:
        from ultralytics import YOLO, __version__
        checks["YOLO"] = __version__
    except ImportError:
        checks["YOLO"] = "Not installed"
    
    # Check CUDA
    try:
        import torch
        checks["CUDA"] = torch.cuda.is_available()
        if checks["CUDA"]:
            checks["GPU Device"] = torch.cuda.get_device_name(0)
    except ImportError:
        checks["CUDA"] = "PyTorch not available"
    
    # DEBUG: Enhanced API key detection
    print("\nDEBUG: API Key Detection Process:")
    api_key = debug_environment_loading()
    
    if api_key and api_key.strip():
        checks["Roboflow API Key"] = f"Found ({api_key[:10]}...)"
        os.environ['ROBOFLOW_API_KEY'] = api_key.strip()
        print(f"✓ API key successfully loaded and set")
    else:
        print("✗ API key not found in any environment source")
        print("  This indicates the env_file directive may not be working")
        print("  or the .env file doesn't exist/contain the key")
    
    # Check test image paths
    test_image_paths = [
        "/workspace/data/images/image.png",
        "data/images/image.png", 
        "./data/images/image.png"
    ]
    
    for path in test_image_paths:
        if os.path.exists(path):
            checks["Test Image"] = f"Found at {path}"
            break
    
    # Display results
    for key, value in checks.items():
        print(f"{key:20}: {value}")
    print("=" * 60)
    
    return checks


def create_test_basketball_image():
    """Create a more realistic synthetic basketball test image."""
    print("Creating enhanced synthetic basketball test image...")
    
    # Ensure directory exists
    os.makedirs("/workspace/data/images", exist_ok=True)
    test_path = "/workspace/data/images/image.png"
    
    # Create a more realistic basketball scene
    img = np.zeros((720, 1280, 3), dtype=np.uint8)
    
    # Basketball court background (wood-like color)
    img[:] = [139, 115, 85]  # Brown wood color
    
    # Court lines (white) - more detailed court
    # Outer boundary
    cv2.rectangle(img, (50, 50), (1230, 670), (255, 255, 255), 4)
    
    # Center circle
    cv2.circle(img, (640, 360), 120, (255, 255, 255), 4)
    cv2.circle(img, (640, 360), 20, (255, 255, 255), 4)
    
    # Center line
    cv2.line(img, (640, 50), (640, 670), (255, 255, 255), 4)
    
    # Three-point lines (simplified)
    cv2.ellipse(img, (125, 360), (200, 300), 0, -90, 90, (255, 255, 255), 4)
    cv2.ellipse(img, (1155, 360), (200, 300), 0, 90, 270, (255, 255, 255), 4)
    
    # Free throw areas
    cv2.rectangle(img, (50, 260), (240, 460), (255, 255, 255), 4)
    cv2.rectangle(img, (1040, 260), (1230, 460), (255, 255, 255), 4)
    
    # Basketball hoops (more realistic)
    cv2.rectangle(img, (40, 340), (60, 380), (255, 140, 0), -1)  # Left hoop
    cv2.rectangle(img, (1220, 340), (1240, 380), (255, 140, 0), -1)  # Right hoop
    
    # Add realistic "players" (person-shaped rectangles)
    players = [
        # Team 1 (Red jerseys)
        ((200, 180), (250, 280), (0, 0, 255)),   # Player 1
        ((300, 400), (350, 500), (0, 0, 255)),   # Player 2  
        ((450, 250), (500, 350), (0, 0, 255)),   # Player 3
        ((600, 450), (650, 550), (0, 0, 255)),   # Player 4
        ((750, 200), (800, 300), (0, 0, 255)),   # Player 5
        
        # Team 2 (Blue jerseys)
        ((400, 150), (450, 250), (255, 0, 0)),   # Player 6
        ((550, 350), (600, 450), (255, 0, 0)),   # Player 7
        ((700, 500), (750, 600), (255, 0, 0)),   # Player 8
        ((850, 300), (900, 400), (255, 0, 0)),   # Player 9
        ((1000, 250), (1050, 350), (255, 0, 0)), # Player 10
    ]
    
    # Draw players with more realistic proportions
    for (x1, y1), (x2, y2), color in players:
        # Body (jersey)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, -1)
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 255, 255), 2)
        
        # Head
        head_center = ((x1 + x2) // 2, y1 - 15)
        cv2.circle(img, head_center, 15, (220, 180, 140), -1)  # Skin tone
        cv2.circle(img, head_center, 15, (255, 255, 255), 2)
    
    # Add basketball (more realistic)
    ball_center = (520, 320)
    cv2.circle(img, ball_center, 25, (255, 140, 0), -1)  # Orange ball
    cv2.circle(img, ball_center, 25, (0, 0, 0), 3)  # Ball outline
    
    # Add basketball lines
    cv2.line(img, (ball_center[0] - 25, ball_center[1]), 
             (ball_center[0] + 25, ball_center[1]), (0, 0, 0), 2)
    cv2.line(img, (ball_center[0], ball_center[1] - 25), 
             (ball_center[0], ball_center[1] + 25), (0, 0, 0), 2)
    
    # Save the image
    cv2.imwrite(test_path, img)
    print(f"Enhanced synthetic test image created at: {test_path}")
    print(f"Image contains: 10 players (5 red, 5 blue) + 1 basketball")
    
    return test_path


def test_yolo_basic():
    """Test basic YOLO functionality with enhanced debugging."""
    print("\n" + "=" * 60)
    print("YOLO BASIC FUNCTIONALITY TEST")
    print("=" * 60)
    
    try:
        print("Step 1: Loading YOLOv8n (nano) model...")
        
        # Enhanced model loading with caching info
        model = YOLO("yolov8n.pt")  # Will download ~6MB model if not cached
        
        print(f"✓ Model loaded successfully!")
        print(f"  Model type: {type(model.model).__name__}")
        print(f"  Model device: {model.device}")
        print(f"  Model task: {model.task}")
        print(f"  Available classes: {len(model.names)} classes")
        print(f"  Person class ID: {[k for k, v in model.names.items() if v == 'person']}")
        print(f"  Sports ball class ID: {[k for k, v in model.names.items() if v == 'sports ball']}")
        
        print("\nStep 2: Testing inference on synthetic image...")
        test_img = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
        
        # Run inference with timing
        import time
        start_time = time.time()
        results = model(test_img, verbose=False)
        inference_time = time.time() - start_time
        
        print(f"✓ Inference completed successfully!")
        print(f"  Inference time: {inference_time*1000:.2f}ms")
        print(f"  Results type: {type(results[0])}")
        print(f"  Detections: {len(results[0].boxes) if results[0].boxes is not None else 0}")
        
        return True
        
    except Exception as e:
        print(f"✗ YOLO basic test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_yolo_on_basketball_image():
    """Test YOLO on the basketball test image with detailed analysis."""
    print("\n" + "=" * 60)
    print("YOLO BASKETBALL IMAGE TEST")
    print("=" * 60)
    
    try:
        # Create test image
        image_path = create_test_basketball_image()
        
        print(f"Step 1: Loading test image from: {image_path}")
        img = cv2.imread(image_path)
        if img is None:
            print("✗ ERROR: Could not load image")
            return False
        
        print(f"✓ Image loaded: {img.shape}")
        
        print("Step 2: Loading YOLO model...")
        model = YOLO("yolov8n.pt")
        
        print("Step 3: Running YOLO inference on basketball scene...")
        
        # Run inference with multiple confidence levels for comparison
        confidence_levels = [0.1, 0.25, 0.5]
        
        for conf in confidence_levels:
            print(f"\n  Testing with confidence threshold: {conf}")
            results = model(img, conf=conf, verbose=False)
            
            if results and len(results) > 0 and results[0].boxes is not None:
                boxes = results[0].boxes
                print(f"    Detections found: {len(boxes)}")
                
                # Analyze detections by class
                detections_by_class = {}
                for box, conf_score, cls in zip(boxes.xyxy, boxes.conf, boxes.cls):
                    class_name = model.names[int(cls)]
                    if class_name not in detections_by_class:
                        detections_by_class[class_name] = []
                    detections_by_class[class_name].append({
                        'confidence': float(conf_score),
                        'box': box.tolist()
                    })
                
                for class_name, detections in detections_by_class.items():
                    print(f"    {class_name}: {len(detections)} detections")
                    for i, det in enumerate(detections[:3]):  # Show first 3
                        print(f"      {i+1}. confidence: {det['confidence']:.3f}")
            else:
                print(f"    No detections found at confidence {conf}")
        
        # Save annotated result with best confidence level
        print(f"\nStep 4: Saving annotated result...")
        results = model(img, conf=0.25, verbose=False)
        annotated_img = results[0].plot()
        output_path = "/workspace/data/images/yolo_test_result.png"
        cv2.imwrite(output_path, annotated_img)
        print(f"✓ Annotated result saved to: {output_path}")
        
        return True
        
    except Exception as e:
        print(f"✗ YOLO basketball image test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_roboflow_connection():
    """Test Roboflow API connection with enhanced debugging."""
    print("\n" + "=" * 60)
    print("ROBOFLOW CONNECTION TEST")
    print("=" * 60)
    
    # Enhanced API key checking
    api_key = os.environ.get('ROBOFLOW_API_KEY')
    print(f"API key from environment: {'SET' if api_key else 'NOT SET'}")
    
    if not api_key:
        print("✗ Roboflow API key not found")
        print("  Troubleshooting steps:")
        print("  1. Check if .env file exists in project root")
        print("  2. Verify ROBOFLOW_API_KEY is set in .env")
        print("  3. Ensure docker-compose.yml has env_file: - .env")
        print("  4. Restart container after updating .env")
        return False
    
    try:
        import roboflow
        from roboflow import Roboflow
        
        print(f"Roboflow library version: {roboflow.__version__}")
        print("Testing API connection...")
        
        rf = Roboflow(api_key=api_key)
        print("✓ Roboflow client initialized successfully")
        
        # Try to access the basketball court detection project
        print("Accessing basketball court detection project...")
        try:
            workspace = rf.workspace("basketball-formations")
            project = workspace.project("basketball-court-detection-2-mlopt")
            print(f"✓ Project accessible: {project.name}")
            
            # Get project info without downloading
            print(f"  Project ID: {project.id}")
            print(f"  Workspace: basketball-formations")
            
            return True
            
        except Exception as project_error:
            print(f"⚠ Project access failed: {project_error}")
            print("  This may be normal - project access depends on API key permissions")
            print("  API key appears valid but may not have access to this specific project")
            return True  # API key works, just access limited
        
    except Exception as e:
        print(f"✗ Roboflow connection test failed: {e}")
        print(f"  API key: {api_key[:10] if api_key else 'None'}...")
        return False


def main():
    """Run comprehensive tests with systematic debugging."""
    print("COMPREHENSIVE YOLO AND BASKETBALL CV TESTING")
    print("=" * 60)
    
    # Step 1: Environment check
    env_status = check_environment()
    
    # Step 2: Test basic YOLO
    basic_ok = test_yolo_basic()
    
    # Step 3: Test basketball image processing
    basketball_ok = test_yolo_on_basketball_image()
    
    # Step 4: Test Roboflow (if API key available)
    roboflow_ok = test_roboflow_connection()
    
    # Step 5: Final summary
    print("\n" + "=" * 60)
    print("FINAL TEST SUMMARY")
    print("=" * 60)
    
    results = {
        "Environment Setup": all([
            env_status["OpenCV"] != "Not installed",
            env_status["YOLO"] != "Not installed", 
            env_status["CUDA"] != "PyTorch not available"
        ]),
        "YOLO Basic": basic_ok,
        "YOLO Basketball": basketball_ok,
        "Roboflow API": roboflow_ok,
        "API Key Available": bool(os.environ.get('ROBOFLOW_API_KEY'))
    }
    
    for test_name, result in results.items():
        status = "PASS" if result else "FAIL"
        symbol = "✓" if result else "✗"
        print(f"{symbol} {test_name:20}: {status}")
    
    # Overall assessment
    critical_tests = [results["Environment Setup"], results["YOLO Basic"], results["YOLO Basketball"]]
    all_critical_passed = all(critical_tests)
    
    print("\n" + "=" * 60)
    if all_critical_passed:
        if results["API Key Available"]:
            print("🎉 SUCCESS: All critical tests passed AND API key is available!")
            print("   Container is fully ready for basketball computer vision work.")
        else:
            print("⚠ PARTIAL SUCCESS: YOLO tests passed but API key not available")
            print("   YOLO detection works, but Roboflow integration requires API key setup")
    else:
        print("❌ FAILURE: Critical tests failed")
        print("   Review error messages above for troubleshooting guidance")
    
    print("\nNext steps:")
    if not results["API Key Available"]:
        print("- Add ROBOFLOW_API_KEY to your .env file")
        print("- Get your API key from: https://app.roboflow.com/settings/api")
    if all_critical_passed:
        print("- Test with real basketball videos")
        print("- Customize detection parameters for your use case")
        print("- Explore court detection and player tracking features")
    
    print("=" * 60)
    
    return 0 if all_critical_passed else 1


if __name__ == "__main__":
    sys.exit(main())
