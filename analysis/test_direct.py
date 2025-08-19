import sys
import json

# Test if scipy is working
try:
    from scipy.signal import find_peaks
    print("✅ scipy imported successfully")
except ImportError as e:
    print(f"❌ scipy import error: {e}")
    print("Run: pip3 install scipy")
    sys.exit(1)

# Test the segmentation import
try:
    from punch_segmentation import detect_individual_punches
    print("✅ punch_segmentation imported successfully")
except ImportError as e:
    print(f"❌ punch_segmentation import error: {e}")
    sys.exit(1)

# Test basic functionality
print("✅ All imports working")
print("Now testing with a video file...")

if len(sys.argv) > 1:
    from analyze_punch import analyze_punch_video
    video_path = sys.argv[1]
    print(f"Testing with: {video_path}")
    
    try:
        result = analyze_punch_video(video_path)
        print("=== RESULT TYPE ===")
        print(type(result))
        
        print("=== RESULT KEYS ===")
        if isinstance(result, dict):
            print(list(result.keys()))
        
        print("=== JSON TEST ===")
        json_str = json.dumps(result)
        print(f"JSON serialization: SUCCESS (length: {len(json_str)})")
        
        print("=== FIRST 500 CHARS ===")
        print(json_str[:500])
        
    except Exception as e:
        print(f"=== ERROR ===")
        print(f"Error type: {type(e)}")
        print(f"Error message: {e}")
        import traceback
        traceback.print_exc()
else:
    print("Usage: python3 test_direct.py <video_file>")