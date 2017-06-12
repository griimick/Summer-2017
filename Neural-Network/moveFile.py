import os
import shutil
import fnmatch

wrongDirs = ["B002"]
rightDirs = ["B0023", "B0024"]

def gen_find(filepat, top):
	for path, dirlist, filelist in os.walk(top):
		for name in fnmatch.filter(filelist, filepat):
			yield os.path.join(path, name)

if __name__ == "__main__":
	src = "E:\\New folder\\Dropbox\\Baggages"
	dst = "E:\\Data\\data_dir\\label_1"
	
	if not os.path.exists(dst):
		os.makedirs(dst)

	subdirs = [name for name in os.listdir(src)
				if os.path.isdir(os.path.join(src, name))]

	for subdir in subdirs:
		if subdir in rightDirs:
			print(subdir)
			filesToMove = gen_find("*.png", os.path.join(src, subdir))
			for name in filesToMove:
				shutil.copy(name, dst)