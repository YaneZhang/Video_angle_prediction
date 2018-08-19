# 全局变量
EXTRACT_FREQUENCY = 20  # 帧提取频率
VIDEO_NUM = 198  # 视频数量
PICTURE_PATH = './picture/'
SIZE = (224, 224)
ROTATE = [0, 90, 180, 270]
ROTATE_FOLDER = './rotate_folder/'  # 旋转后保存的位置


def extract_frames(video_path, dst_folder, index):
    # 主操作
    import cv2
    video = cv2.VideoCapture()
    if not video.open(video_path):
        print("can not open the video")
        exit(1)
    count = 1
    while True:  # 执行一次读一帧
        _, frame = video.read()
        if frame is None:
            break
        if count % EXTRACT_FREQUENCY == 0:
            save_path = "{}/{:>d}.jpg".format(dst_folder, index)
            cv2.imwrite(save_path, frame)
            index += 1  # 记录文件夹中图片数量
        count += 1
    video.release()
    # 打印出所提取帧的总数
    print("Totally save {:d} pics".format(index-1))
    return index


def resize_rotate(path, size, rotate_folder, rotate, folder_num, picture_num):
    from PIL import Image
    rotate_picture = rotate_folder + str(rotate) + '/' + str(folder_num) + '_' + str(picture_num) + '_' + str(rotate) + ".jpg"  # 图片旋转后存放位置与命名规则：原图片文件夹名+原图片名+旋转角度
    im = Image.open(path)
    im_resize = im.resize(size)
    im_rotate = im_resize.rotate(rotate)
    im_rotate.save(rotate_picture)


def main():
    import shutil
    import os
    for index in ROTATE:
        rotate_path = ROTATE_FOLDER + str(index)
        # 递归删除之前存放旋转图片的文件夹，并新建一个
        try:
            shutil.rmtree(rotate_path)
        except OSError:
            pass
        os.mkdir(rotate_path)

    index = 1
    while index <= VIDEO_NUM:
        video_path = './videos/' + str(index) + '.mp4'  # 视频地址
        extract_folder = './picture/' + str(index)  # 存放帧图片的位置
        # 递归删除之前存放帧图片的文件夹，并新建一个
        try:
            shutil.rmtree(extract_folder)
        except OSError:
            pass
        os.mkdir(extract_folder)
        # 抽取帧图片，并保存到指定路径
        num = extract_frames(video_path, extract_folder, 1)
        i = 1
        while i < num:
            for rota in ROTATE:
                picture_path = PICTURE_PATH + str(index) + '/' + str(i) + '.jpg'  # 视频提取图片的存储位置
                resize_rotate(picture_path, SIZE, ROTATE_FOLDER, rota, index, i)
            i += 1
        index += 1


if __name__ == '__main__':
    main()
