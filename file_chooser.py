from kivy.uix.boxlayout import BoxLayout
from kivy.uix.floatlayout import FloatLayout
from kivy.properties import ObjectProperty


class FileChooser(BoxLayout):
    load = ObjectProperty(None)
    cancel = ObjectProperty(None)
    
    def select_image(self, filename):
        try:
            self.ids['preview_image'].source = filename[0]
        except:
            pass