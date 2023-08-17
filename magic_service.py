import cv2
import numpy as np


class MagicService:

    def __init__(self):
        self.BLACK = 0
        self.WHITE = 255
        self.TRANSPARENT = 0

    def do_magic(self, original, master, dimension):
        # white blank image
        output = self.WHITE * np.ones(shape=(dimension[1], dimension[0], 1), dtype=np.uint8)

        for row in range(0, dimension[1]):
            for column in range(0, dimension[0]):
                if original[row][column] == self.WHITE and master[row][column] == self.BLACK:
                    output[row][column] = self.BLACK
                elif original[row][column] == self.BLACK and master[row][column] == self.WHITE:
                    output[row][column] = self.BLACK
                elif original[row][column] == self.WHITE and master[row][column] == self.WHITE:
                    output[row][column] = self.WHITE
                elif original[row][column] == self.BLACK and master[row][column] == self.BLACK:
                    output[row][column] = self.WHITE

        cv2.imwrite("output.png", self.double_size(output))

        master = self.convert_white_to_transparent(self.double_size(master), dimension)
        cv2.imwrite("master.png", master)

    def do_magic_with_path(self, original_path, master_path, dimension):
        original = self.convert(original_path, dimension)
        master = self.convert(master_path, dimension)

        self.do_magic(original, master, dimension)

    def double_size(self, image):
        height = 0
        width = 0
        if len(image.shape) == 3:
            height, width, c = image.shape
        if len(image.shape) == 2:
            height, width = image.shape
        result = self.WHITE * np.ones(shape=(height * 2, width * 2, 1), dtype=np.uint8)

        for row in range(0, height):
            for column in range(0, width):
                row_index = row * 2
                column_index = column * 2

                if image[row][column] == self.BLACK:
                    result[row_index][column_index] = self.BLACK
                    result[row_index + 1][column_index] = self.WHITE

                    result[row_index][column_index + 1] = self.WHITE
                    result[row_index + 1][column_index + 1] = self.BLACK

                elif image[row][column] == self.WHITE:
                    result[row_index][column_index] = self.WHITE
                    result[row_index + 1][column_index] = self.BLACK

                    result[row_index][column_index + 1] = self.BLACK
                    result[row_index + 1][column_index + 1] = self.WHITE
        return result

    @staticmethod
    def convert(path, dimension):
        original = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        original = cv2.resize(original, dimension)
        return cv2.threshold(original, 155, 255, cv2.THRESH_BINARY)[1]

        # cv2.imshow("original", original)  # cv2.imshow("resize", original)  # cv2.imshow("threshold", original)

    def convert_white_to_transparent(self, master, dimension):
        alpha = self.WHITE * np.ones(shape=(dimension[1] * 2, dimension[0] * 2, 1), dtype=np.uint8)
        alpha[master == self.WHITE] = self.TRANSPARENT
        return cv2.merge([master, master, master, alpha])
