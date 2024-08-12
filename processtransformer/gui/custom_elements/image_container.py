

from tkinter import ttk

from PIL import ImageTk, Image

from processtransformer.gui.custom_elements.common_settings import common_padding_x, common_padding_y


class ImageContainer(ttk.Frame):
    def __init__(self, parent, num_rows: int, image_path: str, column_span):
        super().__init__(parent)

        self.image = Image.open(image_path)
        self.tk_image = ImageTk.PhotoImage(self.image)
        self.label = ttk.Label(parent, image=self.tk_image)
        self.label.grid(row=num_rows, column=0, columnspan=column_span,
                        padx=common_padding_x, pady=common_padding_y)

    def destroy(self) -> None:
        super().destroy()
        self.label.destroy()
