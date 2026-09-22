import cv2

image_path = r"C:\Users\GOLDC\Downloads\pexels-ja-kubislav-374578277-14585222.jpg"
image = cv2.imread(image_path)
if image is None:
	raise FileNotFoundError(f"Could not load image: {image_path}")

max_width, max_height = 1200, 800
height, width = image.shape[:2]
scale = min(max_width / width, max_height / height, 1.0)
display_image = cv2.resize(
	image,
	(int(width * scale), int(height * scale)),
	interpolation=cv2.INTER_AREA,
)

cv2.namedWindow("Display window", cv2.WINDOW_NORMAL)
cv2.imshow("Display window", display_image)
cv2.waitKey(0)
cv2.destroyAllWindows()