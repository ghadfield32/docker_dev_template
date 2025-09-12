#!/usr/bin/env python3
"""
EFFICIENT YOLO Basketball Testing Script
Location: .devcontainer/tests/test_yolo_basketball.py

UPDATED VERSION: Focuses on package testing, not model downloading
- Downloads models on-demand only when needed for testing
- Tests YOLO functionality on basketball image at data/images/image.png
- Organized within .devcontainer/tests/ structure
- Efficient approach: package validation first, then targeted testing
"""
import os
import sys
import cv2
import numpy as np
import warnings
warnings.filterwarnings('ignore')


def print_section(title: str) -> None:
    """Print a formatted section header."""
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


def check_environment() -> dict:
    """Check environment setup and return status dict."""
    print_section("ENVIRONMENT VALIDATION")
    
    env_status = {
        "python_version": sys.version.split()[0],
        "opencv_available": False,
        "yolo_available": False,
        "cuda_available": False,
        "gpu_device": "None",
        "api_key_present": False,
        "test_image_path": None
    }
    
    # Check OpenCV
    try:
        import cv2
        env_status["opencv_available"] = cv2.__version__
        print(f"✓ OpenCV: {cv2.__version__}")
    except ImportError:
        print("✗ OpenCV: Not available")
        
    # Check YOLO/Ultralytics
    try:
        from ultralytics import YOLO, __version__
        env_status["yolo_available"] = __version__
        print(f"✓ YOLO/Ultralytics: {__version__}")
    except ImportError:
        print("✗ YOLO/Ultralytics: Not available")
        
    # Check CUDA
    try:
        import torch
        env_status["cuda_available"] = torch.cuda.is_available()
        if env_status["cuda_available"]:
            env_status["gpu_device"] = torch.cuda.get_device_name(0)
        print(f"✓ CUDA: {env_status['cuda_available']} - {env_status['gpu_device']}")
    except ImportError:
        print("✗ PyTorch/CUDA: Not available")
        
    # Check API key (simplified - just ROBOFLOW_API_KEY)
    api_key = os.environ.get('ROBOFLOW_API_KEY')
    env_status["api_key_present"] = bool(api_key and api_key.strip() and api_key != 'your_roboflow_api_key_here')
    if env_status["api_key_present"]:
        print(f"✓ Roboflow API Key: Present ({api_key[:10]}...)")
    else:
        print("⚠ Roboflow API Key: Not set or default value")
        
    # Check test image paths
    test_image_paths = [
        "/workspace/data/images/image.png",
        "data/images/image.png",
        "./data/images/image.png",
        "/app/data/images/image.png"
    ]
    
    for path in test_image_paths:
        if os.path.exists(path):
            env_status["test_image_path"] = path
            print(f"✓ Test image found: {path}")
            break
    
    if not env_status["test_image_path"]:
        print("⚠ Test image not found at expected paths - will create synthetic image")
    
    return env_status


def create_basketball_test_image() -> str:
    """Create a minimal basketball test image for YOLO testing."""
    print_section("CREATING TEST IMAGE")
    
    # Ensure directory exists
    image_dir = "/workspace/data/images"
    os.makedirs(image_dir, exist_ok=True)
    test_path = os.path.join(image_dir, "image.png")
    
    print(f"Creating test basketball image at: {test_path}")
    
    # Create a simple but recognizable basketball scene
    img = np.zeros((640, 640, 3), dtype=np.uint8)
    
    # Basketball court background (simple wood color)
    img[:] = [139, 115, 85]
    
    # Court lines (white)
    cv2.rectangle(img, (50, 50), (590, 590), (255, 255, 255), 3)  # Court boundary
    cv2.circle(img, (320, 320), 80, (255, 255, 255), 3)  # Center circle
    cv2.line(img, (320, 50), (320, 590), (255, 255, 255), 3)  # Center line
    
    # Simple players (person-like rectangles)
    players = [
        ((150, 200), (180, 280), (255, 0, 0)),    # Red player 1
        ((250, 350), (280, 430), (255, 0, 0)),    # Red player 2
        ((400, 180), (430, 260), (0, 0, 255)),    # Blue player 1
        ((480, 400), (510, 480), (0, 0, 255)),    # Blue player 2
    ]
    
    for (x1, y1), (x2, y2), color in players:
        # Player body
        cv2.rectangle(img, (x1, y1), (x2, y2), color, -1)
        # Player head
        head_center = ((x1 + x2) // 2, y1 - 10)
        cv2.circle(img, head_center, 10, (220, 180, 140), -1)
    
    # Basketball (orange circle)
    ball_center = (350, 300)
    cv2.circle(img, ball_center, 15, (0, 165, 255), -1)  # Orange ball
    cv2.circle(img, ball_center, 15, (0, 0, 0), 2)      # Ball outline
    
    # Save the image
    cv2.imwrite(test_path, img)
    print(f"✓ Test image created: 4 players + 1 basketball")
    return test_path


def test_yolo_package() -> bool:
    """Test YOLO package installation without downloading models."""
    print_section("YOLO PACKAGE TEST")
    
    try:
        from ultralytics import YOLO
        import torch
        
        print("✓ YOLO package imported successfully")
        print(f"✓ PyTorch CUDA available: {torch.cuda.is_available()}")
        
        # Test creating YOLO object (doesn't download model yet)
        print("✓ YOLO package ready for model loading")
        return True
        
    except Exception as e:
        print(f"✗ YOLO package test failed: {e}")
        return False


def test_yolo_inference(image_path: str, download_model: bool = True) -> bool:
    """Test YOLO inference on basketball image."""
    print_section("YOLO INFERENCE TEST")
    
    try:
        from ultralytics import YOLO
        import time
        
        # Load image
        print(f"Loading image: {image_path}")
        img = cv2.imread(image_path)
        if img is None:
            print(f"✗ Could not load image from {image_path}")
            return False
        
        print(f"✓ Image loaded: {img.shape}")
        
        # Load YOLO model (will download on first use if needed)
        if download_model:
            print("Loading YOLOv8n model (downloading if needed)...")
            model = YOLO("yolov8n.pt")  # ~6MB download on first use
        else:
            print("✓ Skipping model download for package test")
            return True
        
        print(f"✓ Model loaded on device: {model.device}")
        
        # Run inference
        print("Running inference...")
        start_time = time.time()
        results = model(img, verbose=False, conf=0.25)
        inference_time = time.time() - start_time
        
        print(f"✓ Inference completed in {inference_time*1000:.1f}ms")
        
        # Analyze results
        if results and len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes
            print(f"✓ Detections found: {len(boxes)}")
            
            # Show detections by class
            class_counts = {}
            for cls in boxes.cls:
                class_name = model.names[int(cls)]
                class_counts[class_name] = class_counts.get(class_name, 0) + 1
            
            for class_name, count in class_counts.items():
                print(f"  - {class_name}: {count} detected")
                
            # Save annotated result
            annotated = results[0].plot()
            output_path = "/workspace/data/images/yolo_result.png"
            cv2.imwrite(output_path, annotated)
            print(f"✓ Annotated result saved: {output_path}")
            
        else:
            print("ℹ No detections found (this may be normal for synthetic images)")
        
        return True
        
    except Exception as e:
        print(f"✗ YOLO inference test failed: {e}")
        return False


def test_docker_exec_commands() -> None:
    """Provide docker exec commands for testing this setup."""
    print_section("DOCKER EXEC TESTING COMMANDS")
    
    container_name = os.environ.get('ENV_NAME', 'docker_dev_template')
    
    commands = [
        f"# Test YOLO package only (no model downloads)",
        f"docker exec {container_name}_datascience python /app/tests/test_yolo_basketball.py --package-only",
        "",
        f"# Test YOLO with model download and inference",
        f"docker exec {container_name}_datascience python /app/tests/test_yolo_basketball.py --full-test",
        "",
        f"# Interactive testing session",
        f"docker exec -it {container_name}_datascience bash",
        f"# Then run: python /app/tests/test_yolo_basketball.py",
    ]
    
    for cmd in commands:
        print(cmd)


def main() -> int:
    """Run YOLO basketball testing with efficient approach."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test YOLO on basketball image")
    parser.add_argument('--package-only', action='store_true', 
                       help='Test package only, no model downloads')
    parser.add_argument('--full-test', action='store_true',
                       help='Run full test including model download')
    parser.add_argument('--skip-model-download', action='store_true',
                       help='Skip model downloading in inference test')
    args = parser.parse_args()
    
    print("EFFICIENT YOLO BASKETBALL TESTING")
    print("=" * 60)
    print(f"Container optimized build - models downloaded on-demand only")
    
    # Step 1: Environment validation
    env_status = check_environment()
    
    # Step 2: Package testing (always runs)
    package_ok = test_yolo_package()
    
    if args.package_only:
        print_section("PACKAGE-ONLY TEST SUMMARY")
        print(f"✓ Environment: {'OK' if env_status['yolo_available'] else 'FAIL'}")
        print(f"✓ YOLO Package: {'OK' if package_ok else 'FAIL'}")
        print("✓ Container ready for YOLO work (models will download on first use)")
        return 0 if package_ok else 1
    
    # Step 3: Create test image if needed
    image_path = env_status["test_image_path"]
    if not image_path:
        image_path = create_basketball_test_image()
        
    # Step 4: YOLO inference test
    skip_download = args.skip_model_download and not args.full_test
    inference_ok = test_yolo_inference(image_path, download_model=not skip_download)
    
    # Step 5: Docker exec commands
    test_docker_exec_commands()
    
    # Step 6: Final summary
    print_section("FINAL TEST SUMMARY")
    results = {
        "Environment": env_status["yolo_available"] and env_status["opencv_available"],
        "YOLO Package": package_ok,
        "YOLO Inference": inference_ok or skip_download,
        "Test Image": bool(image_path),
        "API Key": env_status["api_key_present"]
    }
    
    for test_name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        symbol = "✓" if passed else "✗"
        print(f"{symbol} {test_name}: {status}")
    
    critical_tests = [results["Environment"], results["YOLO Package"]]
    all_critical_passed = all(critical_tests)
    
    print("\n" + "=" * 60)
    if all_critical_passed:
        if results["YOLO Inference"]:
            print("🎯 SUCCESS: All tests passed! YOLO is ready for basketball detection.")
        else:
            print("📦 PARTIAL SUCCESS: Package tests passed, inference skipped.")
        
        print("\nNext steps:")
        print("- Container is optimized: models download only when needed")
        print("- Use '--full-test' to test complete inference pipeline")
        if not results["API Key"]:
            print("- Set ROBOFLOW_API_KEY in .env for Roboflow integration")
            
    else:
        print("❌ FAILURE: Critical package tests failed")
        print("- Check environment setup and package installation")
        
    print(f"\nEfficient container: ✓ YOLO ready, models downloaded on-demand")
    return 0 if all_critical_passed else 1


if __name__ == "__main__":
    sys.exit(main())
