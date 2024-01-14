import random
import cv2
import os
import numpy as np
import imutils


def add_light(image, gamma):
    """
    Change brightness of image
    :param image: input image
    :param gamma: value of parameter used to control brightness
    :return: image with adjusted brightness
    """
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")
    image = cv2.LUT(image, table)
    return image


def noise(image, gray_value):
    """
    Add random noise to image
    :param image: input image
    :param gray_value: intensity value of pixels changed by noise
    :return: image with added noise
    """
    height = image.shape[0]
    width = image.shape[1]
    number_of_pixels = random.randint(1000, 10000)
    ret, thresh = cv2.threshold(image, 5, 255, cv2.THRESH_BINARY)
    image2 = image.copy()
    for i in range(number_of_pixels):
        x = random.randint(0, width-1)
        y = random.randint(0, height-1)
        image2[y][x] = gray_value
    image = cv2.bitwise_and(image2, thresh)
    return image


def main():
    images_dir = os.path.join('images', 'preprocessed')
    categories = ['ok', 'nok']

    for i in categories:
        path = os.path.join(images_dir, i)
        val_path = os.path.join('images', 'val', i)
        aug_test_train_path = os.path.join('images', 'aug_test_train', i)

        for images in os.listdir(path):
            if images != '.gitkeep':
                image = cv2.imread(f'{path}//{images}', cv2.IMREAD_COLOR)
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                if_val = random.randint(0, 9)

                # randomly decide whether image belongs to validation or train/test dataset
                # validation dataset
                if if_val < 3:
                    images = images.replace('_preprocessed.jpg', '_4_preprocessed.jpg')
                    val_name = os.path.join(val_path, images)
                    cv2.imwrite(val_name, gray)

                # train/test dataset
                else:
                    # save original image and 4 images after augmentation
                    original_image_path = os.path.join(aug_test_train_path, images)
                    cv2.imwrite(original_image_path, gray)
                    for i in range(4):
                        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                        copy_im = gray.copy()
                        file_name = images.replace('_preprocessed.jpg', '')
                        file_name += '_' + str(i)

                        # add random functions to image
                        if_flip = random.randint(0, 1)
                        if_blur = random.randint(0, 1)
                        if_contrast = random.randint(0, 3)
                        if_noise = random.randint(0, 4)

                        # rotate image
                        angle = random.randint(1, 360)
                        copy_im = imutils.rotate(copy_im, angle)
                        file_name += '_rotate'

                        # flip image
                        if if_flip == 0:
                            copy_im = cv2.flip(copy_im, 1)
                            file_name += '_flip'

                        # blur image
                        if if_blur == 0:
                            sigma_x = random.randint(50, 200)
                            kernel = 2 * random.randint(1, 3) + 1
                            copy_im = cv2.GaussianBlur(copy_im, (kernel, kernel), sigma_x)
                            file_name += '_blur'

                        # change contrast
                        if if_contrast == 0:
                            alpha = random.randint(8, 15)/ 10
                            beta = random.randint(8, 12) / 10
                            copy_im = cv2.convertScaleAbs(copy_im, alpha, beta)
                            file_name += '_contrast'

                        # add noise
                        if if_noise == 0 or (if_flip != 0 and if_blur != 0 and if_contrast != 0):
                            gray2 = copy_im.copy()
                            ret, mask = cv2.threshold(copy_im, 5, 255, cv2.THRESH_BINARY)
                            mean = round(cv2.mean(copy_im, mask)[0])
                            gray_value = random.randint(mean - 20, mean + 20)
                            noise_image = noise(gray2, gray_value)
                            file_name += '_noise.jpg'
                            aug_tes_train_name = os.path.join(aug_test_train_path, file_name)
                            cv2.imwrite(aug_tes_train_name, noise_image)

                        elif if_noise != 0:
                            file_name += '.jpg'
                            aug_tes_train_name = os.path.join(aug_test_train_path, file_name)
                            cv2.imwrite(aug_tes_train_name, copy_im)


if __name__ == '__main__':
    main()
