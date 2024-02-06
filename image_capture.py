import os

class ImageCapture:
    def capture(self, helper, file_path, file_extension):
        file_name = self.get_numbered_filename(file_path, file_extension)
        print(file_name)
        helper.save_swapped_face(file_name)
        print("captured")

    def get_numbered_filename(self, file_path, file_extension):
        name = "Face Swap "
        number = 1
        if not os.path.isdir(file_path):
            os.mkdir(file_path)
        while os.path.isfile(file_path +  name + str(number) + file_extension):
            number = number + 1
        return file_path + name + str(number) + file_extension