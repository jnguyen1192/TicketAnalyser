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
        from PIL import Image
        import pytesseract as t  # dl http://digi.bib.uni-mannheim.de/tesseract/tesseract-ocr-setup-4.00.00dev.exe
        t.pytesseract.tesseract_cmd = r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe"
        res = t.image_to_string(Image.open('../data/Complex.jpg'))
        print(len(res), res)
        self.assertEqual(True, False)


if __name__ == '__main__':
    unittest.main()
