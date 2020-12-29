import unittest


class MyTestCase(unittest.TestCase):
    def test_extract_text_from_image_1(self):
        """
        Sample
        :return:
        """
        # pytesseract https://medium.com/@MicroPyramid/extract-text-with-ocr-for-all-image-types-in-python-using-pytesseract-ec3c53e5fc3a
        from PIL import Image
        import pytesseract as t  # dl http://digi.bib.uni-mannheim.de/tesseract/tesseract-ocr-setup-4.00.00dev.exe
        t.pytesseract.tesseract_cmd = r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe"
        res = t.image_to_string(Image.open('../data/Sample.jpg'))
        print(len(res), res)
        #print(image_to_string(Image.open('test-english.jpg'), lang='fr'))
        assert("POUSSES EPINARD 500G" in res)
        assert("POUSSES EPINARD 500 1.99€ 1" in res)
        #self.assertEqual(True, True)

    def test_extract_text_from_image_2(self):
        """
        Reverse
        :return:
        """
        self.assertEqual(True, False)

    def test_extract_text_from_image_3(self):
        """
        Multi
        :return:
        """
        self.assertEqual(True, False)

    def test_extract_text_from_image_4(self):
        """
        Complex
        :return:
        """# pytesseract https://medium.com/@MicroPyramid/extract-text-with-ocr-for-all-image-types-in-python-using-pytesseract-ec3c53e5fc3a
        import math
        from typing import Tuple, Union

        import cv2
        import numpy as np

        from deskew import determine_skew

        def rotate(
                image: np.ndarray, angle: float, background: Union[int, Tuple[int, int, int]]
        ) -> np.ndarray:
            old_width, old_height = image.shape[:2]
            angle_radian = math.radians(angle)
            width = abs(np.sin(angle_radian) * old_height) + abs(np.cos(angle_radian) * old_width)
            height = abs(np.sin(angle_radian) * old_width) + abs(np.cos(angle_radian) * old_height)

            image_center = tuple(np.array(image.shape[1::-1]) / 2)
            rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
            rot_mat[1, 2] += (width - old_width) / 2
            rot_mat[0, 2] += (height - old_height) / 2
            return cv2.warpAffine(image, rot_mat, (int(round(height)), int(round(width))), borderValue=background)

        image = cv2.imread('rotated.png')
        image = cv2.imread('../data/Complex.jpg')

        grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        angle = determine_skew(grayscale)
        rotated = rotate(image, angle, (0, 0, 0))
        cv2.imwrite('rotated_correted.png', rotated)

        from PIL import Image
        import pytesseract as t  # dl http://digi.bib.uni-mannheim.de/tesseract/tesseract-ocr-setup-4.00.00dev.exe
        t.pytesseract.tesseract_cmd = r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe"
        res = t.image_to_string(Image.open('../data/Sample.jpg'))
        print(res)
        print("-----------------------------------------------")
        res = t.image_to_string(Image.open('../data/Complex.jpg'))
        print(res)
        print("-----------------------------------------------")
        res = t.image_to_string(Image.open('rotated_correted.png'))
        print(res)


if __name__ == '__main__':
    unittest.main()
