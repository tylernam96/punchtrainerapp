import numpy as np
from scipy.signal import find_peaks
import math

def smooth_signal(data, window_size=5):
    """Apply simple moving average to smooth the signal"""
    if len(data) < window_size:
        return data
    
    smoothed = []
    half_window = window_size // 2
    
    for i in range(len(data)):
        start = max(0, i - half_window)
        end = min(len(data), i + half_window + 1)
        smoothed.append(np.mean(data[start:end]))
    
    return smoothed

def detect_individual_punches(frame_data, fps=30):
    """
    Detect individual punches in a video sequence using hand extension analysis
    
    Args:
        frame_data: List of frame analysis data
        fps: Frames per second of the video
    
    Returns:
        List of punch segments with detailed timing information
    """
    
    if not frame_data or len(frame_data) < 10:
        return []
    
    print(f"Analyzing {len(frame_data)} frames for punch detection...")
    
    # Extract hand extension distances (distance from shoulder to wrist)
    hand_extensions = [frame['hand_extension'] for frame in frame_data]
    arm_angles = [frame['right_arm_angle'] for frame in frame_data]
    
    # Smooth the signals to reduce noise
    smooth_extensions = smooth_signal(hand_extensions, window_size=5)
    smooth_angles = smooth_signal(arm_angles, window_size=5)
    
    # Normalize extensions (0 = closest to body, 1 = furthest from body)
    min_ext = min(smooth_extensions)
    max_ext = max(smooth_extensions)
    
    if max_ext - min_ext < 0.01:  # Very little movement
        print("Warning: Very little hand movement detected")
        return []
    
    normalized_extensions = [(ext - min_ext) / (max_ext - min_ext) for ext in smooth_extensions]
    
    # Find extension peaks (full extension points)
    # Use prominence to avoid detecting tiny movements
    min_prominence = 0.2  # Minimum peak prominence (20% of range)
    min_distance = int(fps * 0.3)  # Minimum 0.3 seconds between punches
    
    try:
        peak_indices, peak_properties = find_peaks(
            normalized_extensions,
            prominence=min_prominence,
            distance=min_distance,
            height=0.3  # Must extend at least 30% of maximum range
        )
    except Exception as e:
        print(f"Peak detection error: {e}")
        return []
    
    if len(peak_indices) == 0:
        print("No clear punch peaks detected")
        return []
    
    print(f"Found {len(peak_indices)} potential punches at frames: {peak_indices}")
    
    punches = []
    
    for i, peak_frame in enumerate(peak_indices):
        try:
            # Find the start of this punch (valley before the peak)
            search_start = max(0, peak_frame - int(fps * 0.8))  # Look back max 0.8 seconds
            if i > 0:
                # Don't go before the previous punch's peak
                search_start = max(search_start, peak_indices[i-1])
            
            pre_peak_extensions = normalized_extensions[search_start:peak_frame]
            if pre_peak_extensions:
                # Find the minimum extension before this peak
                min_idx = np.argmin(pre_peak_extensions)
                start_frame = search_start + min_idx
            else:
                start_frame = search_start
            
            # Find the end of this punch (valley after the peak)
            search_end = min(len(frame_data), peak_frame + int(fps * 0.8))  # Look ahead max 0.8 seconds
            if i < len(peak_indices) - 1:
                # Don't go past the next punch's peak
                search_end = min(search_end, peak_indices[i+1])
            
            post_peak_extensions = normalized_extensions[peak_frame:search_end]
            if post_peak_extensions:
                # Find the minimum extension after this peak
                min_idx = np.argmin(post_peak_extensions)
                end_frame = peak_frame + min_idx
            else:
                end_frame = search_end - 1
            
            # Calculate punch metrics
            duration = (end_frame - start_frame) / fps
            extension_time = (peak_frame - start_frame) / fps
            return_time = (end_frame - peak_frame) / fps
            
            # Validate punch duration
            if duration < 0.2 or duration > 2.0:  # Reasonable punch duration
                print(f"Skipping punch {i+1}: duration {duration:.2f}s is unrealistic")
                continue
            
            # Get angles for this punch segment
            punch_arm_angles = arm_angles[start_frame:end_frame+1]
            max_extension_angle = max(punch_arm_angles) if punch_arm_angles else 0
            min_extension_angle = min(punch_arm_angles) if punch_arm_angles else 0
            
            punch_data = {
                'punch_number': len(punches) + 1,
                'start_frame': start_frame,
                'end_frame': end_frame,
                'peak_frame': peak_frame,
                'start_time': frame_data[start_frame]['timestamp'],
                'peak_time': frame_data[peak_frame]['timestamp'],
                'end_time': frame_data[end_frame]['timestamp'],
                'duration': duration,
                'extension_time': extension_time,
                'return_time': return_time,
                'peak_extension': float(normalized_extensions[peak_frame]),
                'max_extension_angle': float(max_extension_angle),
                'min_extension_angle': float(min_extension_angle),
                'extension_range': float(max_extension_angle - min_extension_angle)
            }
            
            punches.append(punch_data)
            print(f"Punch {len(punches)}: {start_frame}-{peak_frame}-{end_frame} ({duration:.2f}s)")
            
        except Exception as e:
            print(f"Error analyzing punch {i+1}: {e}")
            continue
    
    return punches

def analyze_individual_punch(frame_data, punch_segment, overall_fps):
    """Analyze a single punch segment in detail"""
    
    start_frame = punch_segment['start_frame']
    end_frame = punch_segment['end_frame']
    peak_frame = punch_segment['peak_frame']
    
    # Extract data for this punch only
    punch_frames = frame_data[start_frame:end_frame+1]
    
    if not punch_frames:
        return None
    
    try:
        # Extract metrics for this punch
        arm_angles = [frame['right_arm_angle'] for frame in punch_frames]
        hip_rotations = [frame['hip_rotation'] for frame in punch_frames]
        weight_shifts = [frame['weight_shift'] for frame in punch_frames]
        foot_separations = [frame['foot_separation'] for frame in punch_frames]
        chin_protections = [frame['chin_protection'] for frame in punch_frames]
        
        # Timing analysis
        timing_analysis = {
            'total_duration': float(punch_segment['duration']),
            'extension_time': float(punch_segment['extension_time']),
            'return_time': float(punch_segment['return_time']),
            'extension_frames': peak_frame - start_frame,
            'return_frames': end_frame - peak_frame,
            'max_extension_angle': float(max(arm_angles)),
            'min_extension_angle': float(min(arm_angles)),
            'angle_range': float(max(arm_angles) - min(arm_angles))
        }
        
        # Hip rotation analysis
        hip_analysis = {
            'max_rotation_degrees': float(max(hip_rotations)),
            'min_rotation_degrees': float(min(hip_rotations)),
            'average_rotation': float(np.mean(hip_rotations)),
            'rotation_range': float(max(hip_rotations) - min(hip_rotations))
        }
        
        # Balance and footwork
        balance_analysis = {
            'max_weight_shift': float(max(weight_shifts)),
            'weight_stability': float(np.std(weight_shifts)),
            'maintains_balance': bool(max(weight_shifts) < 0.2),
            'foot_movement': float(max(foot_separations) - min(foot_separations)),
            'stable_stance': bool((max(foot_separations) - min(foot_separations)) < 0.1)
        }
        
        # Defense analysis
        defense_analysis = {
            'average_chin_protection': float(np.mean(chin_protections)),
            'chin_protection_consistency': float(np.std(chin_protections)),
            'shoulder_guards_chin': bool(np.mean(chin_protections) < 0)
        }
        
        # Generate feedback for this specific punch
        feedback = generate_punch_feedback(timing_analysis, hip_analysis, balance_analysis, defense_analysis)
        
        analysis = {
            'punch_number': punch_segment['punch_number'],
            'timing': timing_analysis,
            'hip_rotation': hip_analysis,
            'balance': balance_analysis,
            'defense': defense_analysis,
            'feedback': feedback,
            'quality_score': calculate_punch_quality_score(timing_analysis, hip_analysis, balance_analysis, defense_analysis)
        }
        
        return analysis
        
    except Exception as e:
        print(f"Error analyzing punch {punch_segment['punch_number']}: {e}")
        return None

def generate_punch_feedback(timing, hip, balance, defense):
    """Generate specific feedback for a single punch"""
    feedback = []
    
    # Timing feedback
    if timing['extension_time'] < 0.15:
        feedback.append("⚡ Very fast extension!")
    elif timing['extension_time'] > 0.5:
        feedback.append("⚠️ Extension too slow")
    else:
        feedback.append("✅ Good extension timing")
    
    if timing['return_time'] > 0.4:
        feedback.append("⚠️ Slow return to guard")
    else:
        feedback.append("✅ Good return speed")
    
    # Hip rotation feedback
    if hip['max_rotation_degrees'] < 15:
        feedback.append("🔄 More hip rotation needed")
    elif hip['max_rotation_degrees'] > 45:
        feedback.append("⚠️ Too much hip rotation")
    else:
        feedback.append("✅ Good hip rotation")
    
    # Balance feedback
    if not balance['maintains_balance']:
        feedback.append("⚖️ Work on balance")
    else:
        feedback.append("✅ Good balance")
    
    # Footwork feedback
    if not balance['stable_stance']:
        feedback.append("👣 Keep feet stable")
    else:
        feedback.append("✅ Stable stance")
    
    # Extension feedback
    if timing['max_extension_angle'] < 160:
        feedback.append("💪 Extend arm fully")
    else:
        feedback.append("✅ Good arm extension")
    
    # Defense feedback
    if not defense['shoulder_guards_chin']:
        feedback.append("🛡️ Protect chin better")
    else:
        feedback.append("✅ Good chin protection")
    
    return feedback

def calculate_punch_quality_score(timing, hip, balance, defense):
    """Calculate a quality score (0-100) for the punch"""
    score = 0
    
    # Timing score (25 points)
    if 0.15 <= timing['extension_time'] <= 0.4:
        score += 12
    if timing['return_time'] <= 0.4:
        score += 13
    
    # Hip rotation score (20 points)
    if 15 <= hip['max_rotation_degrees'] <= 45:
        score += 20
    elif hip['max_rotation_degrees'] >= 10:
        score += 10
    
    # Balance score (20 points)
    if balance['maintains_balance']:
        score += 10
    if balance['stable_stance']:
        score += 10
    
    # Extension score (20 points)
    if timing['max_extension_angle'] >= 160:
        score += 20
    elif timing['max_extension_angle'] >= 140:
        score += 10
    
    # Defense score (15 points)
    if defense['shoulder_guards_chin']:
        score += 15
    elif defense['average_chin_protection'] < 0.1:
        score += 8
    
    return int(score)