import cv2
from pathlib import Path
import pyscopg2 

def main():
    video_link = r"C:\ofir\ofir's_videos"
    video_folder = Path(video_link)
    if video_folder.exists() and video_folder.is_dir():
        for video_file in video_folder.iterdir():
            if video_file.is_file() and video_file.suffix.lower() in [".mp4", ".avi", ".mov"]:
                cap = cv2.VideoCapture(str(video_file))
                while True:
                    success, frame = cap.read()
                    if not success:
                        break
                    frame = cv2.resize(frame, (640, 480))
                    cv2.imshow(f"video: {video_file.name}", frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
                cap.release()
                cv2.destroyAllWindows()



if __name__ == "__main__":
    main()  