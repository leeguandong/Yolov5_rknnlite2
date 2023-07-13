'''
@Time    : 2023/2/28 13:23
@Author  : leeguandon@gmail.com
'''
import cv2
import moviepy.video.io.ImageSequenceClip


def main():
    data_path = "results/"
    out_path = "kitch_multiperson_.mp4"
    fps = 25
    # width = 1920
    # height = 1080
    # fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    # video = cv2.VideoWriter(out_path, fourcc, fps, (width, height))
    #
    # for i in range(2026):
    #     image_path = data_path + "{:05d}.jpg".format(i)
    #     print(image_path)
    #     img = cv2.imread(image_path)
    #     video.write(img)
    #
    # video.release()
    # cv2.destroyAllWindows()

    image_file = []
    for i in range(2026):
        image_path = data_path + "{:05d}.jpg".format(i)
        image_file.append(image_path)
    clip = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(image_file, fps=fps)
    clip.write_videofile(out_path, fps=fps)
    return clip


if __name__ == "__main__":
    main()
