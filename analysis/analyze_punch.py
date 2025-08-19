import sys
import json
import cv2
import mediapipe as mp
import numpy as np
import math
import os
import io
from punch_segmentation import detect_individual_punches, analyze_individual_punch

# Suppress MediaPipe warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import logging
logging.getLogger('mediapipe').setLevel(logging.ERROR)

# Redirect stderr to null to prevent mixing with JSON output
import io
sys.stderr = io.StringIO()

def calculate_angle(a, b, c):
    """Calculate angle between three points"""
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    
    if angle > 180.0:
        angle = 360 - angle
        
    return float(angle)

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
    
    print(f"Processing video at {fps} FPS...")
    
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
                    cos_angle = np.clip(cos_angle, -1.0, 1.0)
                    hip_rotation = float(abs(np.degrees(np.arccos(cos_angle))))
                else:
                    hip_rotation = 0.0
                
                # Foot position tracking
                foot_separation = calculate_distance(left_ankle, right_ankle)
                
                # Weight distribution (simplified)
                body_center_x = (left_shoulder[0] + right_shoulder[0] + left_hip[0] + right_hip[0]) / 4
                foot_center_x = (left_ankle[0] + right_ankle[0]) / 2
                weight_shift = float(abs(body_center_x - foot_center_x))
                
                # Hand extension distance from body
                hand_extension = calculate_distance(right_wrist, right_shoulder)
                
                # Chin protection
                chin_protection = float(right_shoulder[1] - nose[1])
                
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
                pass
        
        frame_count += 1
    
    cap.release()
    
    print(f"Processed {frame_count} frames, {len(frame_data)} with pose data")
    
    # Perform segmented analysis
    analysis_results = perform_segmented_analysis(frame_data, fps)
    
    result = {
        "success": True,
        "total_frames": frame_count,
        "analyzed_frames": len(frame_data),
        "fps": float(fps),
        "duration_seconds": float(frame_count / fps),
        "analysis": analysis_results
    }
    
    return convert_numpy_types(result)

def perform_segmented_analysis(frame_data, fps):
    """Perform multi-punch segmented analysis"""
    
    if not frame_data:
        return {"error": "No valid frames to analyze"}
    
    print("Starting punch segmentation...")
    
    # Detect individual punches
    punches = detect_individual_punches(frame_data, fps)
    
    if not punches:
        print("No punches detected, falling back to single-sequence analysis")
        return perform_fallback_analysis(frame_data, fps)
    
    print(f"Detected {len(punches)} individual punches")
    
    # Analyze each punch individually
    individual_analyses = []
    quality_scores = []
    
    for punch_segment in punches:
        punch_analysis = analyze_individual_punch(frame_data, punch_segment, fps)
        if punch_analysis:
            individual_analyses.append(punch_analysis)
            quality_scores.append(punch_analysis['quality_score'])
    
    if not individual_analyses:
        return {"error": "Failed to analyze detected punches"}
    
    # Overall analysis
    analysis = {
        'analysis_type': 'multi_punch',
        'summary': {
            'total_punches_detected': len(punches),
            'punches_analyzed': len(individual_analyses),
            'video_duration': float(frame_data[-1]['timestamp']),
            'average_punch_rate': float(len(punches) / frame_data[-1]['timestamp'] * 60),
            'average_quality_score': float(np.mean(quality_scores)),
            'best_punch_score': float(max(quality_scores)),
            'consistency_score': float(100 - (np.std(quality_scores) * 2))  # Higher std = lower consistency
        },
        'individual_punches': individual_analyses,
        'overall_feedback': generate_overall_feedback(individual_analyses, quality_scores),
        'recommendations': generate_recommendations(individual_analyses)
    }
    
    return analysis

def perform_fallback_analysis(frame_data, fps):
    """Fallback analysis for videos where punch segmentation fails"""
    
    # Simple analysis treating the whole video as one sequence
    arm_angles = [frame['right_arm_angle'] for frame in frame_data]
    hip_rotations = [frame['hip_rotation'] for frame in frame_data]
    
    # Find the most extended position (minimum angle)
    most_extended_frame = int(np.argmin(arm_angles))
    max_extension_angle = float(max(arm_angles))
    
    analysis = {
        'analysis_type': 'single_sequence',
        'summary': {
            'total_frames_analyzed': len(frame_data),
            'video_duration': float(frame_data[-1]['timestamp']),
            'max_extension_angle': max_extension_angle,
            'max_hip_rotation': float(max(hip_rotations))
        },
        'feedback': [
            "⚠️ Could not detect individual punches",
            "📊 Analyzed as single sequence",
            f"💪 Maximum arm extension: {max_extension_angle:.1f}°",
            f"🔄 Maximum hip rotation: {max(hip_rotations):.1f}°",
            "💡 Try recording with clearer punch separation for better analysis"
        ]
    }
    
    return analysis

def generate_overall_feedback(individual_analyses, quality_scores):
    """Generate overall feedback across all punches"""
    
    feedback = []
    avg_score = np.mean(quality_scores)
    
    # Overall performance assessment
    if avg_score >= 80:
        feedback.append("🥊 Excellent overall technique! Your fundamentals are very solid")
    elif avg_score >= 65:
        feedback.append("👍 Good technique with room for refinement")
    elif avg_score >= 50:
        feedback.append("📚 Several areas for improvement - focus on the basics")
    else:
        feedback.append("🔧 Significant technique improvements needed")
    
    # Consistency analysis
    score_std = np.std(quality_scores)
    if score_std < 10:
        feedback.append("✅ Very consistent technique across punches")
    elif score_std < 20:
        feedback.append("⚡ Reasonably consistent with some variation")
    else:
        feedback.append("⚠️ Inconsistent technique - focus on repeatable form")
    
    # Best punch identification
    best_punch_idx = np.argmax(quality_scores)
    feedback.append(f"⭐ Your best punch was #{best_punch_idx + 1} (Score: {quality_scores[best_punch_idx]})")
    
    # Common issues analysis
    timing_issues = sum(1 for analysis in individual_analyses 
                       if any("slow" in fb.lower() for fb in analysis['feedback']))
    if timing_issues >= len(individual_analyses) * 0.5:
        feedback.append("⏱️ Focus on punch speed and timing")
    
    balance_issues = sum(1 for analysis in individual_analyses 
                        if not analysis['balance']['maintains_balance'])
    if balance_issues >= len(individual_analyses) * 0.5:
        feedback.append("⚖️ Work on maintaining balance throughout punches")
    
    return feedback

def generate_recommendations(individual_analyses):
    """Generate specific training recommendations"""
    
    recommendations = []
    
    # Analyze common weaknesses
    extension_issues = sum(1 for a in individual_analyses 
                          if a['timing']['max_extension_angle'] < 160)
    if extension_issues >= len(individual_analyses) * 0.5:
        recommendations.append({
            'area': 'Arm Extension',
            'issue': 'Not reaching full extension',
            'suggestion': 'Practice shadow boxing with focus on full arm extension',
            'priority': 'High'
        })
    
    timing_issues = sum(1 for a in individual_analyses 
                       if a['timing']['extension_time'] > 0.4)
    if timing_issues >= len(individual_analyses) * 0.5:
        recommendations.append({
            'area': 'Punch Speed',
            'issue': 'Slow punch execution',
            'suggestion': 'Practice rapid jabs with lighter resistance',
            'priority': 'Medium'
        })
    
    hip_issues = sum(1 for a in individual_analyses 
                    if a['hip_rotation']['max_rotation_degrees'] < 15)
    if hip_issues >= len(individual_analyses) * 0.5:
        recommendations.append({
            'area': 'Hip Rotation',
            'issue': 'Insufficient hip engagement',
            'suggestion': 'Practice hip rotation drills and core strengthening',
            'priority': 'High'
        })
    
    return recommendations

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(json.dumps({"error": "Please provide video file path"}))
        sys.exit(1)
    
    video_path = sys.argv[1]
    result = analyze_punch_video(video_path)
    print(json.dumps(result, indent=2))