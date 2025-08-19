import cv2
import mediapipe as mp

print("Testing MediaPipe setup...")

# Test MediaPipe import
try:
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose()
    print("✅ MediaPipe loaded successfully!")
except Exception as e:
    print(f"❌ MediaPipe error: {e}")

# Test OpenCV
try:
    print(f"✅ OpenCV version: {cv2.__version__}")
except Exception as e:
    print(f"❌ OpenCV error: {e}")

print("\n🚀 Ready to analyze videos!")