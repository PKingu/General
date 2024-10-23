#import libraries
import tkinter as tk
from tkinter import messagebox, font
import subprocess

# Function to close the application
def close_app(event=None):
    root.destroy()

# Function to open the Workouts.py script
def open_workouts():
    subprocess.run(["python", "Workouts.py"])

# Function to open the BicepTracking.py script
def open_bicep_tracking():
    subprocess.run(["python", "BicepTracking.py"])

# Function to display a message for upcoming features
def feature_coming_soon():
    messagebox.showinfo("Coming Soon", "This feature is under development.")

# Create the main application window
root = tk.Tk()
root.title("Fitness Tracker Menu")
root.geometry("400x400")  # Set size of the window
root.configure(bg='darkslategray')  # Set a cool dark background color

# Define a custom font
customFont = font.Font(family="Helvetica", size=12, weight="bold")

# Create a frame for the top section
top_frame = tk.Frame(root, bg='darkslategray', width=400, height=50)
top_frame.pack(side=tk.TOP, fill=tk.X)

# Create a title label in the top frame
title_label = tk.Label(top_frame, text="Fitness Tracker", font=("Helvetica", 18, "bold"), bg='darkslategray', fg='lightgreen')
title_label.pack(pady=10)

# Create a frame for the buttons
button_frame = tk.Frame(root, bg='darkslategray')
button_frame.pack(expand=True)

# Function to create styled buttons
def create_button(parent, text, command, bg_color, fg_color):
    return tk.Button(parent, text=text, command=command, bg=bg_color, fg=fg_color, font=customFont, padx=20, pady=10, relief=tk.RAISED, borderwidth=3)

# Create and pack buttons within the button frame
btn_history = create_button(button_frame, "History", open_workouts, 'steelblue', 'white')
btn_history.pack(pady=10)

btn_biceps = create_button(button_frame, "Biceps", open_bicep_tracking, 'seagreen', 'white')
btn_biceps.pack(pady=10)

btn_squats = create_button(button_frame, "Squats", feature_coming_soon, 'firebrick', 'white')
btn_squats.pack(pady=10)

btn_pushups = create_button(button_frame, "Pushups", feature_coming_soon, 'purple', 'white')
btn_pushups.pack(pady=10)

# Bind 'q' keypress to the close_app function
root.bind('<q>', close_app)

# Start the Tkinter event loop
root.mainloop()