from tkinter import Tk, Label, Button, filedialog, Frame, IntVar, Radiobutton, messagebox
from PIL import Image, ImageTk
import tkinter.font as tkFont

import cv2
import numpy as np
import statistics
import matplotlib.pyplot as plt

from Smooth import AverageSmoothing, GaussianSmoothing, MedianSmoothing


def gaussian_smooth(input_path, kernel=(3, 3)):
    try:
        # Đọc ảnh bằng OpenCV
        original_image = cv2.imread(input_path)
        original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

        # Áp dụng bộ lọc Gaussian Blur
        # blurred_image = GaussianSmoothing(original_image, kernel[0])

        # using openCV cv2.GaussianBlur
        blurred_image = cv2.GaussianBlur(original_image, kernel, 0)

        return blurred_image

    except Exception as e:
        print(e)
def average_smooth(input_path, kernel=(3, 3)):
    try:
        # Đọc ảnh bằng OpenCV
        original_image = cv2.imread(input_path)
        original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

        # Apply average smoothing
        # smoothed_image = AverageSmoothing(original_image, kernel[0])

        # Apply average smoothing (box blur) using cv2.blur
        smoothed_image = cv2.blur(original_image, kernel)

        # Return the smoothed image
        return smoothed_image

    except Exception as e:
        print(e)


def median_smooth(input_path, kernel=(3, 3)):
    try:
        # Đọc ảnh bằng OpenCV
        original_image = cv2.imread(input_path)
        original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

        # Áp dụng Median Filter
        smoothed_image = MedianSmoothing(original_image, kernel[0])
        
        # Áp dụng Median Filter sử dụng thư viện openCV medianBlur
        # smoothed_image = cv2.medianBlur(original_image, kernel[0])

        return smoothed_image

    except Exception as e:
        print(e)


class ImageProcessorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Smooth Image - Image Processing")

        # Biến lưu đường dẫn ảnh đầu vào và đầu ra
        self.input_path = ""
        self.output_path = ""

        # 
        self.img_arr = None
        self.label_image_result = None
        self.label_image_source = None

        # Cố định kích thước của sổ
        self.root.geometry("1100x600")

        # Tạo các thành phần giao diện

        # Frame
        self.frame_top_bar = Frame(self.root, width=1100, height=200, padx=5, pady=5, bg='')
        self.frame_top_bar.pack_propagate(False)
        # self.frame_top_bar.grid(row = 1, column = 0)

        self.frame_transform = Frame(self.root, width=1100, height=600, padx=5, pady=5, bg='')
        self.frame_transform.pack_propagate(False)
        # self.frame_transform.grid(row = 1, column = 0)

        self.frame_smooth = Frame(self.root, width=1100, height=600, padx=5, pady=5, bg='')
        self.frame_smooth.pack_propagate(False)
        self.frame_smooth.grid(row = 1, column = 0)



        # Frame
        self.frame_side_bar = Frame(self.frame_smooth, width=200, height=600, padx=5, pady=5, bg='#C8CBD5')
        self.frame_side_bar.pack_propagate(False)
        self.frame_side_bar.grid(row = 0, column = 0)

        self.frame_image_src = Frame(self.frame_smooth,width=300, height=600, padx=5, pady=5)
        self.frame_image_src.pack_propagate(False)
        self.frame_image_src.grid(row = 0, column = 1)

        self.frame_middle = Frame(self.frame_smooth,width=300, height=600, padx=5, pady=5)
        self.frame_middle.pack_propagate(False)
        self.frame_middle.grid(row = 0, column = 2)

        self.frame_image_rsl = Frame(self.frame_smooth,width=300, height=600, padx=5, pady=5)
        self.frame_image_rsl.pack_propagate(False)
        self.frame_image_rsl.grid(row = 0, column = 3)

        # playholder cho các ảnh
        text_title_s = Label(self.frame_image_src, text="Source image",font = (None, 16))
        text_title_s.pack(pady=10)
        text_title_r = Label(self.frame_image_rsl, text="Result image",font = (None, 16))
        text_title_r.pack(pady=10)

        placeholder_image = Image.open("./image/image-files.png")
        placeholder_image = placeholder_image.resize((250, 250), Image.LANCZOS)
        self.placeholder_photo = ImageTk.PhotoImage(placeholder_image)


        # label chứa các ảnh
        self.label_image_source = Label(self.frame_image_src)
        self.label_image_result = Label(self.frame_image_rsl)
        # 
        # placeholder cho ảnh kết quả
        self.placeholder_source = Label(self.frame_image_src, image=self.placeholder_photo)
        self.placeholder_source.image =  self.placeholder_photo
        self.placeholder_source.pack(pady=40)

        # placeholder cho ảnh kết quả
        self.placeholder_result = Label(self.frame_image_rsl, image=self.placeholder_photo)
        self.placeholder_result.image =  self.placeholder_photo
        self.placeholder_result.pack(pady=40)
        # Button
        load_button = Button(self.frame_image_src, text="Load image", command=self.load_image, padx=10, pady=10, width=30)
        load_button.pack(pady=40, side='bottom')

        process_button = Button(self.frame_middle, text="Process image", command=self.process_image, padx=10, pady=10, width=30)
        process_button.pack(pady=40, side='bottom')

        save_button = Button(self.frame_image_rsl, text="Save image", command=self.save_image, padx=10, pady=10, width=30)
        save_button.pack(pady=40, side='bottom')

        # Các label trong sidebar
        text_title = Label(self.frame_side_bar, text="Choose type of filter",font = (None, 16), bg='#C8CBD5')
        text_title.pack(pady=20)

        self.OPTION = IntVar()
        self.OPTION.set("2")

        average_radio_button = Radiobutton(self.frame_side_bar, text='Average Filter', variable=self.OPTION, value=1, padx=10, pady=10, width=20)
        average_radio_button.pack(pady=10, padx=0)

        gaussian_radio_button = Radiobutton(self.frame_side_bar, text='Gaussian Filter', variable=self.OPTION, value=2, padx=10, pady=10, width=20)
        gaussian_radio_button.pack(pady=10, padx=0)

        median_radio_button = Radiobutton(self.frame_side_bar, text='Median Filter', variable=self.OPTION, value=3, padx=10, pady=10, width=20)
        median_radio_button.pack(pady=10, padx=0)

        reset_button = Button(self.frame_side_bar, text="Reset", command=self.reset_status, padx=10, pady=10, width=30, bg='#6876F6')
        reset_button.pack(pady=20, padx=0)

    def load_image(self):
        
        # Mở hộp thoại để chọn ảnh
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.png;*.jpg;*.jpeg;*.gif")])

        if file_path:
            self.input_path = file_path
            self.output_path = "smoothed_" + file_path.split("/")[-1]
            # Xoá các ảnh cũ nếu có
            self.clear_status()
            # Hiển thị ảnh trên giao diện
            image_source = Image.open(self.input_path)
            # image_source.thumbnail((250, 250))
            image_source = image_source.resize((250, 250), Image.LANCZOS)
            photo_src = ImageTk.PhotoImage(image_source)
            
            # xoá placeholer
            self.placeholder_source.pack_forget()

            self.label_image_source = Label(self.frame_image_src, image = photo_src)
            self.label_image_source.image = photo_src
            self.label_image_source.pack(pady=40, anchor='center')

    def process_image(self):

        if self.input_path:
            result_array =None
            if self.OPTION.get() == 1:
                result_array = average_smooth(self.input_path, (3,3))
            if self.OPTION.get() == 2:
                result_array = gaussian_smooth(self.input_path, (3,3))
            if self.OPTION.get() == 3: 
                result_array = median_smooth(self.input_path, (3,3))

            self.img_arr = result_array
            # Convert NumPy array to ImageTk format
            image_result = Image.fromarray(result_array)
            image_result = image_result.resize((250, 250), Image.LANCZOS)
            photo_rsl = ImageTk.PhotoImage(image_result)

            self.placeholder_result.pack_forget()
            self.label_image_result.pack_forget()

            self.label_image_result = Label(self.frame_image_rsl, image=photo_rsl)
            self.label_image_result.image = photo_rsl
            self.label_image_result.pack(pady=40, anchor='center')
            # Notifi
            messagebox.showinfo("Notifi", f"Successfully smooth an image !")

    def save_image(self):
        if self.input_path:
            # Lưu ảnh đã được làm trơn
            image_result = cv2.cvtColor(self.img_arr, cv2.COLOR_RGB2BGR)
            cv2.imwrite(self.output_path,image_result)

            # Hiển thị thông báo
            # self.result_label = Label(self.frame_image_rsl, text=f"Successfully save image: {self.output_path}")
            # result_label.pack(pady=10)
            messagebox.showinfo("Notifi", f"Successfully save image: {self.output_path}")

    def clear_status(self):
        self.label_image_source.pack_forget()
        self.label_image_result.pack_forget()
        self.placeholder_result.pack(pady=40, anchor='center')

    def reset_status(self):

        self.label_image_source.pack_forget()
        self.label_image_result.pack_forget()
        self.placeholder_source.pack_forget()
        self.placeholder_result.pack_forget()

        self.placeholder_source.pack(pady=40)
        self.placeholder_result.pack(pady=40)

        self.input_path = ""

# Tạo và khởi chạy ứng dụng
if __name__ == "__main__":
    root = Tk()
    app = ImageProcessorApp(root)
    root.mainloop()
