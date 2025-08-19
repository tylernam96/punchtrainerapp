import sys
import json
import cv2
import mediapipe as mp
import numpy as np
import math

def calculate_angle(a, b, c):
    """Calculate angle between three points"""
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    
    if angle > 180.0:
        angle = 360 - angle
        
    return float(angle)  # Convert to regular Python float

def calculate_distance(point1, point2):
    """Calculate Euclidean distance between two points"""
    return float(math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2))

def convert_numpy_types(obj):
    """Convert numpy types to Python native types for JSON serialization"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    return obj

def analyze_punch_video(video_path):
    # Initialize MediaPipe
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        smooth_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        return {"error": "Could not open video file"}
    
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    frame_data = []
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb_frame)
        
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            
            try:
                # Key body landmarks
                left_shoulder = [float(landmarks[11].x), float(landmarks[11].y)]
                right_shoulder = [float(landmarks[12].x), float(landmarks[12].y)]
                left_elbow = [float(landmarks[13].x), float(landmarks[13].y)]
                right_elbow = [float(landmarks[14].x), float(landmarks[14].y)]
                left_wrist = [float(landmarks[15].x), float(landmarks[15].y)]
                right_wrist = [float(landmarks[16].x), float(landmarks[16].y)]
                
                left_hip = [float(landmarks[23].x), float(landmarks[23].y)]
                right_hip = [float(landmarks[24].x), float(landmarks[24].y)]
                
                left_ankle = [float(landmarks[27].x), float(landmarks[27].y)]
                right_ankle = [float(landmarks[28].x), float(landmarks[28].y)]
                
                nose = [float(landmarks[0].x), float(landmarks[0].y)]
                
                # Calculate metrics
                right_arm_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)
                left_arm_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
                
                # Hip rotation (angle between shoulder line and hip line)
                shoulder_vector = np.array(right_shoulder) - np.array(left_shoulder)
                hip_vector = np.array(right_hip) - np.array(left_hip)
                
                # Prevent division by zero
                shoulder_norm = np.linalg.norm(shoulder_vector)
                hip_norm = np.linalg.norm(hip_vector)
                
                if shoulder_norm > 0 and hip_norm > 0:
                    cos_angle = np.dot(shoulder_vector, hip_vector) / (shoulder_norm * hip_norm)
                    # Clamp cos_angle to valid range [-1, 1]
                    cos_angle = np.clip(cos_angle, -1.0, 1.0)
                    hip_rotation = float(abs(np.degrees(np.arccos(cos_angle))))
                else:
                    hip_rotation = 0.0
                
                # Foot position tracking
                foot_separation = calculate_distance(left_ankle, right_ankle)
                
                # Weight distribution (simplified - based on body center vs foot position)
                body_center_x = (left_shoulder[0] + right_shoulder[0] + left_hip[0] + right_hip[0]) / 4
                foot_center_x = (left_ankle[0] + right_ankle[0]) / 2
                weight_shift = float(abs(body_center_x - foot_center_x))
                
                # Hand extension distance from body
                hand_extension = calculate_distance(right_wrist, right_shoulder)
                
                # Chin protection (shoulder height relative to nose)
                chin_protection = float(right_shoulder[1] - nose[1])  # Negative means shoulder is higher
                
                frame_data.append({
                    'frame': frame_count,
                    'timestamp': float(frame_count / fps),
                    'right_arm_angle': right_arm_angle,
                    'left_arm_angle': left_arm_angle,
                    'hip_rotation': hip_rotation,
                    'foot_separation': foot_separation,
                    'weight_shift': weight_shift,
                    'hand_extension': hand_extension,
                    'chin_protection': chin_protection,
                    'right_wrist': right_wrist,
                    'right_shoulder': right_shoulder,
                    'right_elbow': right_elbow
                })
                
            except (IndexError, AttributeError) as e:
                # Skip frames where landmarks aren't detected properly
                pass
        
        frame_count += 1
    
    cap.release()
    
    # Analyze the collected data
    analysis_results = perform_detailed_analysis(frame_data, fps)
    
    result = {
        "success": True,
        "total_frames": frame_count,
        "analyzed_frames": len(frame_data),
        "fps": float(fps),
        "duration_seconds": float(frame_count / fps),
        "analysis": analysis_results
    }
    
    # Convert all numpy types to Python native types
    return convert_numpy_types(result)

def perform_detailed_analysis(frame_data, fps):
    if not frame_data:
        return {"error": "No valid frames to analyze"}
    
    analysis = {}
    feedback = []
    
    # 1. Extension and Return Timing
    arm_angles = [frame['right_arm_angle'] for frame in frame_data]
    max_extension_frame = int(np.argmax(arm_angles))
    min_angle_frame = int(np.argmin(arm_angles))
    
    extension_time = float((max_extension_frame - min_angle_frame) / fps)
    return_time = float((len(frame_data) - max_extension_frame) / fps)
    
    analysis['timing'] = {
        'extension_time_seconds': extension_time,
        'extension_frames': max_extension_frame - min_angle_frame,
        'return_time_seconds': return_time,
        'return_frames': len(frame_data) - max_extension_frame,
        'max_extension_angle': float(max(arm_angles))
    }
    
    # Timing feedback
    if extension_time > 0.5:  # More than half a second
        feedback.append("⚠️ Your jab extension is too slow - aim for 0.2-0.4 seconds")
    elif extension_time < 0.15:
        feedback.append("⚡ Very fast extension! Good snap on your jab")
    else:
        feedback.append("✅ Good extension timing")
    
    if return_time > 0.4:
        feedback.append("⚠️ Return your jab to guard position faster for better defense")
    else:
        feedback.append("✅ Good return speed to guard position")
    
    # 2. Hip Rotation Analysis
    hip_rotations = [frame['hip_rotation'] for frame in frame_data]
    max_hip_rotation = float(max(hip_rotations))
    avg_hip_rotation = float(np.mean(hip_rotations))
    
    analysis['hip_rotation'] = {
        'max_rotation_degrees': max_hip_rotation,
        'average_rotation': avg_hip_rotation,
        'rotation_range': float(max(hip_rotations) - min(hip_rotations))
    }
    
    if max_hip_rotation < 15:
        feedback.append("🔄 Rotate your hips more for additional power - aim for 25-35 degrees")
    elif max_hip_rotation > 45:
        feedback.append("⚠️ Too much hip rotation - you might be over-rotating")
    else:
        feedback.append("✅ Good hip rotation contributing to punch power")
    
    # 3. Foot Movement Analysis
    foot_separations = [frame['foot_separation'] for frame in frame_data]
    foot_movement = float(max(foot_separations) - min(foot_separations))
    
    analysis['footwork'] = {
        'foot_movement_range': foot_movement,
        'stable_stance': bool(foot_movement < 0.1)  # Convert to bool
    }
    
    if foot_movement > 0.15:
        feedback.append("👣 Excessive foot movement detected - try to maintain stable stance")
    else:
        feedback.append("✅ Good stable stance throughout the punch")
    
    # 4. Weight Distribution
    weight_shifts = [frame['weight_shift'] for frame in frame_data]
    max_weight_shift = float(max(weight_shifts))
    
    analysis['balance'] = {
        'max_weight_shift': max_weight_shift,
        'maintains_balance': bool(max_weight_shift < 0.2)  # Convert to bool
    }
    
    if max_weight_shift > 0.25:
        feedback.append("⚖️ Significant weight shift detected - try to maintain center balance")
    else:
        feedback.append("✅ Good balance maintained")
    
    # 5. Extension Analysis
    hand_extensions = [frame['hand_extension'] for frame in frame_data]
    max_extension = float(max(hand_extensions))
    extension_consistency = float(np.std(hand_extensions))
    
    analysis['extension'] = {
        'max_reach': max_extension,
        'consistency': extension_consistency,
        'full_extension_achieved': bool(max(arm_angles) > 160)  # Convert to bool
    }
    
    if max(arm_angles) < 160:
        feedback.append("💪 Extend your arm more fully for maximum reach and power")
    else:
        feedback.append("✅ Good full arm extension")
    
    # 6. Chin Protection
    chin_protections = [frame['chin_protection'] for frame in frame_data]
    avg_chin_protection = float(np.mean(chin_protections))
    
    analysis['defense'] = {
        'average_chin_protection': avg_chin_protection,
        'shoulder_guards_chin': bool(avg_chin_protection < 0)  # Convert to bool
    }
    
    if avg_chin_protection > 0:
        feedback.append("🛡️ Keep your non-punching shoulder higher to protect your chin")
    else:
        feedback.append("✅ Good chin protection with shoulder position")
    
    # Overall Assessment
    good_feedback = len([f for f in feedback if "✅" in f])
    warning_feedback = len([f for f in feedback if "⚠️" in f])
    
    if good_feedback >= 4:
        feedback.insert(0, "🥊 Excellent technique! Your fundamentals are solid")
    elif warning_feedback >= 3:
        feedback.insert(0, "📚 Several areas for improvement - focus on the basics")
    else:
        feedback.insert(0, "👍 Good punch with room for refinement")
    
    analysis['feedback'] = feedback
    
    # Convert all numpy types in the analysis
    return convert_numpy_types(analysis)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(json.dumps({"error": "Please provide video file path"}))
        sys.exit(1)
    
    video_path = sys.argv[1]
    result = analyze_punch_video(video_path)
    print(json.dumps(result, indent=2))