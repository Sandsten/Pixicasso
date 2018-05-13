import cv2
import sys

def checkImage(img, path):
	if img is None:
		print("Error: Invalid path:", path, "\n")
		sys.exit("Error while reading files. Exiting the program.\n")


#Read Image
content_path = 'content-img/lion.jpg'
content_img = cv2.imread(content_path)
checkImage(content_img, content_path)

style_path = 'style-img/starry-night.jpg' 
style_img = cv2.imread(style_path)
checkImage(style_img, style_path)

#Display Image
cv2.imshow('image',style_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imwrite('result/result.png',style_img)

