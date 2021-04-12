import cv2
import os.path


video_name = 'video.avi'

images = []
image_folder = '/home/v/Documents/ml/frames-desert-sand2snow'
for al in range(30, 80+1):
    al = str(round(0.01*al, 2))
    if len(al.split('.')[1]) == 1:
        al = al + '0'
    im_path = os.path.join(image_folder, al + '.png')
    assert os.path.exists(im_path), 'no "{}"'.format(im_path)
    images.append(im_path)


image_folder = '/home/v/Documents/ml/frames-desert-snow2gross'
for al in list(range(0, 45+1, 7)) + list(range(45, 90+1)):
    al = str(round(0.01*al, 2))
    if len(al.split('.')[1]) == 1:
        al = al + '0'
    im_path = os.path.join(image_folder, al + '.png')
    assert os.path.exists(im_path), 'no "{}"'.format(im_path)
    images.append(im_path)


frame = cv2.imread(images[0])
height, width, layers = frame.shape

video = cv2.VideoWriter(video_name, 0, 16, (width,height))

for image in images:
    video.write(cv2.imread(image))

cv2.destroyAllWindows()
video.release()
