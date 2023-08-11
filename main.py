import cv2
import numpy as np

BLACK = 0
WHITE = 255

def double_size(image):
    if len(image.shape) == 3:
        h, w, c = image.shape
    if len(image.shape) == 2:
        h, w = image.shape
    result = 255 * np.ones(shape=(h * 2, w * 2, 1), dtype=np.uint8)

    for row in range(0, h):
        for column in range(0, w):
            row_index = row * 2
            column_index = column * 2

            if image[row][column] == BLACK:
                result[row_index][column_index] = BLACK
                result[row_index + 1][column_index] = WHITE

                result[row_index][column_index + 1] = WHITE
                result[row_index + 1][column_index + 1] = BLACK

            elif image[row][column] == WHITE:
                result[row_index][column_index] = WHITE
                result[row_index + 1][column_index] = BLACK

                result[row_index][column_index + 1] = BLACK
                result[row_index + 1][column_index + 1] = WHITE
    return result


if __name__ == '__main__':
    original = cv2.imread("input/lichtburg.png", cv2.IMREAD_GRAYSCALE)
    original = cv2.threshold(original, 155, 255, cv2.THRESH_BINARY)[1]

    master_original = cv2.imread("input/master.png", cv2.IMREAD_GRAYSCALE)
    master_original = cv2.threshold(master_original, 155, 255, cv2.THRESH_BINARY)[1]

    h, w = original.shape
    output = 255 * np.ones(shape=(h, w, 1), dtype=np.uint8)

    for row in range(0, h):
        for column in range(0, w):
            if original[row][column] == WHITE and master_original[row][column] == BLACK:
                output[row][column] = BLACK
            elif original[row][column] == BLACK and master_original[row][column] == WHITE:
                output[row][column] = BLACK
            elif original[row][column] == WHITE and master_original[row][column] == WHITE:
                output[row][column] = WHITE
            elif original[row][column] == BLACK and master_original[row][column] == BLACK:
                output[row][column] = WHITE

    cv2.imwrite("output.png", double_size(output))
    cv2.imwrite("master.png", double_size(master_original))

    # cv2.imshow("master", master)    # cv2.imshow("output", output)    # cv2.waitKey(0)
