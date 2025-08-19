import sys
import json
from analyze_punch import analyze_punch_video

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python debug_analyze.py <video_path>")
        sys.exit(1)
    
    video_path = sys.argv[1]
    print("=== DEBUGGING ANALYSIS ===")
    print(f"Analyzing: {video_path}")
    
    try:
        result = analyze_punch_video(video_path)
        print("=== RESULT TYPE ===")
        print(type(result))
        print("=== RESULT KEYS ===")
        if isinstance(result, dict):
            print(result.keys())
        print("=== FULL RESULT ===")
        print(json.dumps(result, indent=2))
    except Exception as e:
        print(f"=== ERROR ===")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()