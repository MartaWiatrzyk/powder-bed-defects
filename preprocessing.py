import cv2
import os
import numpy as np


def find_min_radius(images_dir: str, categories: list[str]) -> int:
    """
    Find minimum radius of 3d printer's table for given dataset
    :param categories: labels for images (ok/ nok)
    :param images_dir: directory of raw images (with background)
    :return: found minimum radius for all images
    """
    min_radius = 800
    for i in categories:
        path = os.path.join(images_dir, i)
        print(path)
        for images in os.listdir(path):
            if images != '.gitkeep':
                print('image: ', images)
                image = cv2.imread(f'{path}//{images}', cv2.IMREAD_COLOR)
                # cv2.imshow('erode', image)
                # cv2.waitKey(0)
                # cv2.destroyAllWindows()
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                rows = gray.shape[0]
                circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, rows/2, param1=100, param2=30, minRadius=460,
                                        maxRadius=530)
                if circles is not None:
                    circles = np.uint16(np.around(circles))
                    circle = circles[0, 0]
                    if circle[2] < min_radius:
                        min_radius = circle[2]
    return min_radius


def remove_background(images_dir: str, min_radius: int, preprocessed_dir: str, categories: list[str]) -> None:
    """
    Remove background (leave only 3d printer's table)
    :param categories: labels for images (ok/ nok)
    :param images_dir: directory of raw images (with background)
    :param min_radius: found minimum radius for all images
    :param preprocessed_dir: directory for saving images without background
    """
    for i in categories:
        path = os.path.join(images_dir, i)
        for images in os.listdir(path):
            if images != '.gitkeep':
                image = cv2.imread(f'{path}//{images}', cv2.IMREAD_COLOR)

                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                rows = gray.shape[0]

                # find circles on image
                circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, rows/2, param1=100, param2=30, minRadius=460,
                                        maxRadius=530)
                if circles is not None:
                    circles = np.uint16(np.around(circles))
                    circle = circles[0, 0]
                    radius = circle[2]

                    # calculate resize value
                    resize_value = min_radius/radius
                    dim = (round(image.shape[1]*resize_value), round(image.shape[0]*resize_value))
                    mask = np.zeros_like(image)
                    mask = cv2.resize(mask, dim, interpolation=cv2.INTER_AREA)
                    center = (round(circle[0]*resize_value), round(circle[1]*resize_value))
                    image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)

                    # reduce image to circle
                    cv2.circle(mask, center, round(radius*resize_value) - 25, (255, 255, 255), -1)
                    result = cv2.bitwise_and(image, mask)
                    cropped_image = result[round(circle[1]*resize_value)-min_radius:
                                        round(circle[1]*resize_value)+min_radius,
                                        round(circle[0]*resize_value)-min_radius:
                                        round(circle[0]*resize_value)+min_radius]

                    result_name = os.path.join(preprocessed_dir, i, images.rstrip('.bmp') + "_preprocessed.jpg")
                    cv2.imwrite(result_name, cropped_image)
                else:
                    print(images)


def main():
    raw_images_dir = os.path.join('images', 'raw_images')
    preprocessed_dir = os.path.join('images', 'preprocessed')

    categories = ['ok', 'nok']

    # crete directory if it doesn't exist
    for category in categories:
        os.makedirs(os.path.join(preprocessed_dir, category), exist_ok=True)

    min_radius = find_min_radius(raw_images_dir, categories)
    remove_background(raw_images_dir, min_radius, preprocessed_dir, categories)


if __name__ == '__main__':
    main()
