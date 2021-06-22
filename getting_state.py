# Program for partial screenshot
  
import pyscreenshot, numpy
from PIL import Image
from datetime import datetime

class Frame:
    """
        getting frame from streaming car game.
        full image = (80, 100, 1800, 1000) (900, 1760)
        cropped image = (300, 100, 1660, 1000) (900,1360)

    """

    def __init__(self):
        
        self.coordinates = (40, 100, 1800, 1000)
        self.image = None
    
    def take_screenshot(self):
        # im=pyscreenshot.grab(bbox=(x1,x2,y1,y2))
        self.image = pyscreenshot.grab(bbox=self.coordinates)

    def save_image(self):

        now = datetime.now()

        current_time = now.strftime("%H-%M-%S")
        self.image.save("frames/frame-"+current_time+".png")

    def get_pixels(self):
        """
            convert to numpy array and gray scale
        """
        pix = numpy.asarray(self.image.convert('L')) ## gray scale image
        return pix

    def get_frame(self):
        ## getting refined frame
        self.take_screenshot()
        self.save_image()
        # self.image.show()
        return self.get_pixels()

# if __name__ == "__main__":

#     frame = Frame()
#     print(frame.get_frame().shape)
