#coding=utf-8
PICTURE_PATH = './data/var/'
PICTURE_NUM = 5
SIZE = (224, 224)
ROTATE = [0, 90, 180, 270]
ROTATE_FOLDER = './rotate_folder/'  # 旋转后保存的位置

    
def resize_rotate(path, num, rotate, size, rotate_folder):
    from PIL import Image
    for i in range(num):
        picture_path = path + str(i) + '.jpg'
        for ro in rotate:
            rotate_picture = rotate_folder + str(ro) + '/' + str(i) + '_' + str(ro) + ".jpg"
            im = Image.open(picture_path)
            im_resize = im.resize(size)
            im_rotate = im_resize.rotate(ro)
            im_rotate.save(rotate_picture)


if __name__ == "__main__":
    resize_rotate(PICTURE_PATH, PICTURE_NUM, ROTATE, SIZE, ROTATE_FOLDER)