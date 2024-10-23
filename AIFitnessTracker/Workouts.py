# Import Libraries
import os
import tkinter as tk
from tkinter import Listbox, Scrollbar, Toplevel, messagebox
from PIL import Image, ImageTk # Allows us to display images
import datetime

# Function to load files from a directory and sort them by creation date.
def load_and_sort_files(directory):
    # List comprehension to get full file paths.
    files = [os.path.join(directory, f) for f in os.listdir(directory)]
    # Sorting files by modification time, newest first.
    files.sort(key=os.path.getmtime, reverse=True)
    return files

# Function to display the selected image in a new window.
def display_image(file_path):
    # Check if the file is an image based on its extension
    if file_path.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
        # Creating a top-level window to show the image.
        new_window = Toplevel(root)
        new_window.title("Image Viewer")
        # Opening and resizing the image to fit the display window.
        img = Image.open(file_path)
        img = img.resize((500, 500), Image.LANCZOS)  # LANCZOS filter for high-quality downscaling.
        img = ImageTk.PhotoImage(img)
        # Displaying the image in a label widget.
        panel = tk.Label(new_window, image=img, bg='black')  # Adding a background color to the label.
        panel.image = img
        panel.pack()
    else:
        # If the file is not an image, display an error message
        tk.messagebox.showerror("Error", "Selected file is not an image.")

# Function to build the user interface.
def create_ui(file_list):
    # Creating a Listbox to display the list of files.
    listbox = Listbox(root, width=50, height=20, bg='black', fg='yellow')  # Adding color to the Listbox.
    listbox.pack(side=tk.LEFT, fill=tk.Y)

    # Adding a scrollbar to the Listbox.
    scrollbar = Scrollbar(root, orient="vertical")
    scrollbar.config(command=listbox.yview)
    scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

    listbox.config(yscrollcommand=scrollbar.set)

    # Populating the Listbox with file names and modification times.
    for file in file_list:
        file_name = os.path.basename(file)
        file_time = datetime.datetime.fromtimestamp(os.path.getmtime(file)).strftime('%Y-%m-%d %H:%M:%S')
        listbox.insert(tk.END, f"{file_name} (Modified: {file_time})")

    # Binding a function to the Listbox selection event.
    listbox.bind('<<ListboxSelect>>', lambda event: display_image(file_list[event.widget.curselection()[0]]))

# Function to quit the program when 'q' is pressed
def quit_program(event=None):
    root.destroy()

# Main function to initialize the program.
def main(directory):
    global root
    root = tk.Tk()
    root.title("Workout History Viewer")
    root.configure(bg='darkgrey')  # Adding a background color to the root window.

    root.bind('<q>', quit_program) # if 'q' is pressed quit

    # Loading and sorting files from the specified directory.
    file_list = load_and_sort_files(directory)
    # Creating the user interface with the sorted file list.
    create_ui(file_list)

    root.mainloop()


history_directory = "History"  # Directory containing the history of workouts/images.
main(history_directory)
