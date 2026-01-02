from utils import read_video,save_video
from trackers import PlayerTracker

def main():
    #Read the Input Video
    input_video_path = "input_video/input_video1.mp4"
    video_frames = read_video(input_video_path)

    #Detect players
    player_tracker = PlayerTracker(model_path = "yolo12n.pt")
    player_detections = player_tracker.detect_frames(video_frames, read_from_stub = True, stub_path = "tracker_stubs/player_detection.pkl")
    #Draw output
    #Draw player bounding boxes
    output_video_frames = player_tracker.draw_bboxes(video_frames,player_detections)
    #save video

    save_video(output_video_frames,"output_videos/output.avi")

if __name__ == "__main__":
    main()
