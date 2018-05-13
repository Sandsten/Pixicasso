import cv2
import sys
import argparse

def parse_args():
	parser = argparse.ArgumentParser(description='Style transfer.')
	parser.add_argument('--style_img', type=str, 
		help='Filenames of the style image (example: starry-night.jpg)', 
    	default='starry-night.jpg')
	parser.add_argument('--content_img', type=str, default='lion.jpg',
    	help='Filename of the content image (example: lion.jpg)')
	args = parser.parse_args()
	return args

def checkImage(img, path):
	if img is None:
		sys.exit("Error: Invalid path {}\nExiting the program.\n".format(path))

########## MAIN FUNCTION ##########
global args
args = parse_args()

#Read Images
content_path = 'content-img/' + args.content_img
content_img = cv2.imread(content_path)
checkImage(content_img, content_path)

style_path = 'style-img/' + args.style_img
style_img = cv2.imread(style_path)
checkImage(style_img, style_path)

#Display Image (this is just for testing)
cv2.imshow('image',style_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

#Save result
cv2.imwrite('result/result.png',style_img)

