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
        #custom_config = r'--oem 3 --psm 6'
        res = t.image_to_string(Image.open('../data/Sample.jpg'))#, config=custom_config)
        print(res)
        print("-----------------------------------------------")
        res = t.image_to_string(Image.open('../data/Complex.jpg'))#, config=custom_config)
        print(res)
        print("-----------------------------------------------")
        res = t.image_to_string(Image.open('rotated_correted.png'))#, config=custom_config)
        print(res)

    def test_boxing_text(self):
        import cv2
        import pytesseract
        from matplotlib import pyplot as plt
        from pytesseract import Output
        pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe"

        image = cv2.imread("rotated_correted.png")

        # Saving a original image and shape
        orig = image.copy()

        # Display the image with bounding box and recognized text
        orig_image = orig.copy()

        # Moving over the results and display on the image
        for ((start_X, start_Y, end_X, end_Y), text) in results:
            # display the text detected by Tesseract
            print("{}\n".format(text))

            # Displaying text
            text = "".join([x if ord(x) < 128 else "" for x in text]).strip()
            cv2.rectangle(orig_image, (start_X, start_Y), (end_X, end_Y),
                          (0, 0, 255), 2)
            cv2.putText(orig_image, text, (start_X, start_Y - 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        plt.imshow(orig_image)
        plt.title('Output')
        plt.show()




    def test_extract_text_from_image_5(self):
        """
        Complexes and find pattern for ALDI Ticket
        :return:
        """
        import ticketanalyser
        dir = "../data/"
        path_img_list = [dir + "Aldi_1.jpg", dir + "Aldi_2.jpg"]
        for path_img in path_img_list:
            ticketanalyser.ticket_image_to_text(path_img)

    def test_rotate(self):
        # import the necessary packages
        import numpy as np
        import argparse
        import imutils
        import cv2

        # Read the image and create a blank mask
        image  = cv2.imread("../data/Aldi_1.jpg")
        grayScale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Find edges in the image using canny edge detection method
        # Calculate lower threshold and upper threshold using sigma = 0.33
        sigma = 0.33
        v = np.median(grayScale)
        low = int(max(0, (1.0 - sigma) * v))
        high = int(min(255, (1.0 + sigma) * v))

        edged = cv2.Canny(grayScale, low, high)

        # After finding edges we have to find contours
        # Contour is a curve of points with no gaps in the curve
        # It will help us to find location of shapes

        # cv2.RETR_EXTERNAL is passed to find the outermost contours (because we want to outline the shapes)
        # cv2.CHAIN_APPROX_SIMPLE is removing redundant points along a line
        (cnts, _) = cv2.findContours(edged,
                                        cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        '''
        We are going to use contour approximation method to find vertices of
        geometric shapes. The alogrithm  is also known as Ramer Douglas Peucker alogrithm.
        In OpenCV it is implemented in cv2.approxPolyDP method.abs
        detectShape() function below takes a contour as parameter and
        then returns its shape
         '''

        def detectShape(cnt):
            shape = 'unknown'
            # calculate perimeter using
            peri = cv2.arcLength(c, True)
            # apply contour approximation and store the result in vertices
            vertices = cv2.approxPolyDP(c, 0.04 * peri, True)

            # If the shape it triangle, it will have 3 vertices
            if len(vertices) == 3:
                shape = 'triangle'

            # if the shape has 4 vertices, it is either a square or
            # a rectangle
            elif len(vertices) == 4:
                # using the boundingRect method calculate the width and height
                # of enclosing rectange and then calculte aspect ratio

                x, y, width, height = cv2.boundingRect(vertices)
                aspectRatio = float(width) / height

                # a square will have an aspect ratio that is approximately
                # equal to one, otherwise, the shape is a rectangle
                if aspectRatio >= 0.95 and aspectRatio <= 1.05:
                    shape = "square"
                else:
                    shape = "rectangle"

            # if the shape is a pentagon, it will have 5 vertices
            elif len(vertices) == 5:
                shape = "pentagon"

            # otherwise, we assume the shape is a circle
            else:
                shape = "circle"

            # return the name of the shape
            return shape

        # Now we will loop over every contour
        # call detectShape() for it and
        # write the name of shape in the center of image

        # loop over the contours
        for c in cnts:
            # compute the moment of contour
            M = cv2.moments(c)
            # From moment we can calculte area, centroid etc
            # The center or centroid can be calculated as follows
            cX = int(M['m10'] / M['m00'])
            cY = int(M['m01'] / M['m00'])

            # call detectShape for contour c
            shape = detectShape(c)

            # Outline the contours
            cv2.drawContours(image, [c], -1, (0, 255, 0), 2)

            # Write the name of shape on the center of shapes
            cv2.putText(image, shape, (cX, cY), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (255, 255, 255), 2)

            imS = cv2.resize(image, (960, 540))  # Resize image
            cv2.imshow("output", imS)  # Show image

        cv2.waitKey(0)
        cv2.destroyAllWindows()



    def test_extract_text_from_image_6(self):
        """
        Complexes and find pattern for Carrefour Market
        :return:
        """

if __name__ == '__main__':
    unittest.main()
