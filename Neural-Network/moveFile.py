import os
import shutil
import fnmatch
import cv2

rightDirs = ["B0010","B0011","B0012","B0014","B0016","B0017", "B0018", "B0019", "B0021", "B0022", "B0023", "B0032"]

def gen_find(filepat, top):
	for path, dirlist, filelist in os.walk(top):
		for name in fnmatch.filter(filelist, filepat):
			yield os.path.join(path, name)

if __name__ == "__main__":
	src = "E:\\New folder\\Dropbox\\Baggages"
	dst = "E:\\Data\\data_all\\label_gun"
	dstEval = "E:\\Data\\data_eval\\label_0"
	dstTrain = "E:\\Data\\data_dir\\label_0"
	
	if not os.path.exists(dst):
		os.makedirs(dst)

	subdirs = [name for name in os.listdir(src)
				if os.path.isdir(os.path.join(src, name))]

	for subdir in subdirs:
		if subdir in rightDirs:
			print(subdir)
			filesToMove = gen_find("*.png", os.path.join(src, subdir))
			for name in filesToMove:
				print(name)
				image = cv2.imread(name)
				resized_image = cv2.resize(image, (300, 220))
				cv2.imwrite(os.path.join(dst, os.path.basename(name)), resized_image)