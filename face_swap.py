from os import truncate
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.popup import Popup
from kivy.graphics.texture import Texture
from kivy.clock import Clock
from kivy.core.window import Window
from kivy.config import Config
from kivy.properties import ListProperty
from os.path import expanduser
from os.path import join
import cv2

from face_swap_helper_webcam import FaceSwapHelperWebcam
from image_capture import ImageCapture
from webcam import WebCam
from file_chooser import FileChooser

Config.set('input', 'mouse', 'mouse,disable_multitouch')

class FaceSwap(BoxLayout):
    HOME_DIR = expanduser('~')
    IMAGE_CAPTURE_FILE_EXTENSION = ".png"

    helper = FaceSwapHelperWebcam()
    
    Window.minimum_width = 400
    Window.minimum_height = 300
    
    fullscreen_camera = False

    def build(self):
        self.set_webcam_spinner()
        
    def update(self, dt):
        try:
            self.helper.update_face_swap()
        except cv2.error as e:
            print(e)
        except RuntimeError as e:
            print(e)
        
        if self.helper.paused:
            self.ids['notification'].text = "paused"
            self.ids['notification'].color = (0, 0, 1, 1)
            self.ids['swapped_image'].texture = self.get_texture(
                self.helper.webcam_image.image)
            return
        
        if self.helper.successful:
            self.ids['swapped_image'].texture = self.get_texture(
                self.helper.swapped_face)
            self.ids['notification'].text = "successful"
            self.ids['notification'].color = (0, 1, 0, 1)
        else:
            self.ids['swapped_image'].texture = self.get_texture(
                self.helper.webcam_image.image)
            self.ids['notification'].text = "no face detected"
            self.ids['notification'].color = (1, 0, 0, 1)            

    def get_texture(self, img):
        flipped_img = cv2.flip(img, 0)
        texture = Texture.create(size=(img.shape[1], img.shape[0]), colorfmt='bgr') 
        texture.blit_buffer(flipped_img.tobytes(), colorfmt='bgr', bufferfmt='ubyte')
        return texture

    def capture(self):
        ImageCapture().capture(self.helper, self.HOME_DIR + "\\Pictures\\Face Swap\\", 
            self.IMAGE_CAPTURE_FILE_EXTENSION)

    def set_webcam_spinner(self):
        webcams = []
        for i in range(WebCam.count_webcams()):
            webcams.append("Webcam " + str(i))
        self.ids["webcam_spinner"].values = webcams
        self.ids["webcam_spinner"].bind(text = self.change_webcam)

    def change_webcam(self, spinner, text):
        self.ids["webcam_spinner"].text = text
        webcam_number = int(text[len(text) - 1 : len(text)])
        self.helper.set_webcam(webcam_number)

    def open_file_chooser(self):
        content = FileChooser(load=self.change_source_image, cancel=self.dismiss_file_chooser)
        content.ids['filechooser'].rootpath = self.HOME_DIR
        content.ids['filechooser'].filters = ['*.bmp', '*.pbm', '*.pgm', '*.ppm', 
               '*.sr', '*.ras', '*.jpeg', '*.jpg', 
               '*.jpe', '*.jp2', '*.tiff', '*.tif', 
               '*.png']
        self.file_chooser_popup = Popup(title="Load image", content=content,
                            size_hint=(0.9, 0.9))
        self.file_chooser_popup.open()
        self.helper.paused = True

    def dismiss_file_chooser(self):
        self.file_chooser_popup.dismiss()
        self.helper.paused = False

    def change_source_image(self, path, filename):
        filepath = join(path, filename[0]).replace("\\", "\\\\")
        try:
            self.helper.set_source_image(filepath)
        except:
            self.ids['notification'].text = "cannot detect face in source image"
            self.ids['notification'].color = (1, 0, 0, 1)
            return
        self.ids['source_image'].texture = self.get_texture(
            self.helper.source_image.image)
        self.file_chooser_popup.dismiss()
        self.helper.paused = False

    def change_flip_camera(self):
        self.helper.flip_camera = not self.helper.flip_camera
        if(self.helper.flip_camera):
            self.ids['flip_camera'].text = "Flip Camera (on)"
        else:
            self.ids['flip_camera'].text = "Flip Camera (off)"
    
    def toogle_camera_fullscreen(self, root):
        if(self.fullscreen_camera):
            self.fullscreen_camera = False
            self.ids['swapped_image'].size = (root.width / 2, root.height / 2) 
            self.ids['swapped_image'].center = (self.parent.center[0], self.parent.center[1]) #self.parent.center #
        else:
            self.fullscreen_camera = True
            self.ids['swapped_image_button'].size = (root.width, root.height) 
            self.ids['swapped_image'].size = (root.width, root.height) 
            self.ids['swapped_image'].center = (root.center_x, root.center_y)

class FaceSwapApp(App):
    def build(self):
        self.icon = 'images/logo.png'
        self.title = "Face Swap"
        self.fs = FaceSwap()
        self.fs.build()
        Clock.schedule_interval(
            self.fs.update, 1.0/self.fs.helper.get_framerate())
        return self.fs

if __name__ == '__main__':
    FaceSwapApp().run()
