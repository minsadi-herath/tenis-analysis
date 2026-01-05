from utils import read_video,save_video
from trackers import PlayerTracker, BallTracker
from court_line_detector import CourtLineDetector

def main():
    #Read the Input Video
    input_video_path = "input_video/input_video1.mp4"
    video_frames = read_video(input_video_path)

    #Detect players
    player_tracker = PlayerTracker(model_path = "yolo12n.pt")
    player_detections = player_tracker.detect_frames(video_frames, read_from_stub = True, stub_path = "tracker_stubs/player_detection.pkl")

    #Detect Tennis Ball
    ball_tracker = BallTracker(model_path = "model/best.pt")
    ball_detections = ball_tracker.detect_frames(video_frames,read_from_stub = True, stub_path="tracker_stubs/ball_detection.pkl")

    #Detect tennis court
    court_model_path = "model/keypoints_model_50.pth"
    court_line_detector = CourtLineDetector(court_model_path)
    court_keypoints = court_line_detector.predict(video_frames[0])



    #Draw output
    #Draw player bounding boxes
    output_video_frames = player_tracker.draw_bboxes(video_frames,player_detections)

    #Draw tennis ball bounding boxes
    output_video_frames = ball_tracker.draw_bboxes(output_video_frames, ball_detections)

    #Draw tennis keypoints
    output_video_frames = court_line_detector.draw_keypoints_on_video(output_video_frames,court_keypoints)
    #save video
    save_video(output_video_frames,"output_videos/output.avi")

if __name__ == "__main__":
    main()
