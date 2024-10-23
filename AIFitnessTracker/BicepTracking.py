# Importing the necessary libraries for the program
import cv2  # OpenCV library for capturing and processing the video feed
import mediapipe as mp  # MediaPipe library for pose detection
import numpy as np  # NumPy library for numerical computations
import time  # Time library to access the system time for timestamps
import matplotlib.pyplot as plt  # Import the plotting module for data visualization
import os   # Import for directory operations
import datetime   # Import to know the date for default figure names
import tkinter as tk
from tkinter import simpledialog, messagebox   # Import for GUI dialogs

# Utilizing drawing utilities from MediaPipe for visualizing landmarks
mp_drawing = mp.solutions.drawing_utils
# Utilizing the pose estimation model from MediaPipe
mp_pose = mp.solutions.pose

# Recording the start time of the program execution to calculate time intervals
epoch = time.time()

# Lists to store the count of repetitions and time gaps between each repetition
counterR_list = []
counterL_list = []
gapR_list = []
gapL_list = []

# Initialize a variable to keep track of the last print time outside the function
last_print_time = 0

# Initialize gradients with default values (could be None or any other appropriate value)
gradient_left, gradient_right = 0, 0

# Defining a function to calculate the angle between three points
def calculate_angle(a, b, c):
    a = np.array(a)  # Converts the first point (shoulder) into an array for vector operations
    b = np.array(b)  # Converts the second point (elbow) into an array for vector operations
    c = np.array(c)  # Converts the third point (wrist) into an array for vector operations

    # Calculate the angle in radians between the lines formed by (a, b) and (c, b)
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    # Convert the angle from radians to degrees and take its absolute value
    angle = np.abs(radians * 180.0 / np.pi)

    # Normalize the angle value to ensure it is less than 180 degrees
    if angle > 180.0:
        angle = 360 - angle

    return angle  # The function returns the calculated angle


# Defining our zoom in and out function
def on_scroll(event):
    # Acquire the current axis of the plot to manipulate
    ax = plt.gca()

    # Fetch the current limits of the x and y axes of the plot
    cur_xlim = ax.get_xlim()
    cur_ylim = ax.get_ylim()

    # Retrieve the mouse cursor's x and y coordinates at the time of the scroll event
    xdata = event.xdata
    ydata = event.ydata

    # If the scroll event occurred outside the plot boundaries, do nothing
    if xdata is None or ydata is None:
        return


    # Define a base scale factor for zooming; here, it's set to 10%
    scale_factor = 0.1
    # For an 'up' scroll, zoom in by reducing the scale factor (closer view)
    if event.button == 'down':
        scale_factor = 1 - scale_factor
    # For a 'down' scroll, zoom out by increasing the scale factor (wider view)
    elif event.button == 'up':
        scale_factor = 1 + scale_factor

    # Set new x and y limits based on the scroll direction, centered around the cursor
    ax.set_xlim([xdata - (xdata - cur_xlim[0]) * scale_factor,
                 xdata + (cur_xlim[1] - xdata) * scale_factor])
    ax.set_ylim([ydata - (ydata - cur_ylim[0]) * scale_factor,
                 ydata + (cur_ylim[1] - ydata) * scale_factor])

    # Redraw the plot to update the view
    plt.draw()


# Function to detect and notify bad form during exercises
def detect_bad_form(image, shoulder, elbow, wrist):

    # Use the global keyword to modify the variable outside the function
    global last_print_time

    # Load the incorrect form image from the file system
    incorrect_form_img = cv2.imread('incorrect.png')

    # Calculate the position to overlay the incorrect form image at the bottom left of the video feed
    position = (50, image.shape[0] - incorrect_form_img.shape[0] - 10)

    # Calculate the angle at the elbow joint
    angle = calculate_angle(shoulder, elbow, wrist)

    # Define the minimum and maximum angles for good form
    good_form_angle_min = 0.00001
    good_form_angle_max = 180

    current_time = time.time()  # Get the current time

    # Check if the calculated angle falls outside the good form range
    if not good_form_angle_min <= angle <= good_form_angle_max:
        # Overlay the incorrect form image onto the video feed if the form is bad
        image[position[1]:position[1] + incorrect_form_img.shape[0],
        position[0]:position[0] + incorrect_form_img.shape[1]] = incorrect_form_img
        if current_time - last_print_time > 2:  # Check if more than 2 seconds have passed since the last print
            # Print a message to the console to alert the user to adjust their form
            print("Adjust your form.")
            last_print_time = current_time # Update Last Print Time

    # Calculate the horizontal distance between the elbow and wrist
    horizontal_distance = abs(elbow[0] - wrist[0])

    # Define a threshold for the acceptable horizontal distance indicating bad form
    horizontal_threshold = 0.15  # This value can be adjusted based on observation and research

    # Check if the horizontal distance exceeds the threshold, indicating bad form
    if horizontal_distance > horizontal_threshold:
        # Overlay the incorrect form image onto the video feed if the form is bad
        image[position[1]:position[1] + incorrect_form_img.shape[0],
        position[0]:position[0] + incorrect_form_img.shape[1]] = incorrect_form_img
        if current_time - last_print_time > 2:  # Check if more than 2 seconds have passed since the last print
            # Print a message to the console to alert the user to adjust their form
            print("Adjust your form.")
            last_print_time = current_time  # Update Last Print Time

# Setting up the video capture object to use the first camera device
cap = cv2.VideoCapture(0)

# Initializing variables to keep track of the number of curls and the current stage of a curl
counter_left = 0
counter_right = 0
stage_left = None
stage_right = None

# Creating an instance of the MediaPipe Pose class with specified confidence levels
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    # Beginning a loop to process video feed frame by frame
    while cap.isOpened():
        # Capturing a frame from the webcam
        ret, frame = cap.read()

        # Convert the image color from the default BGR to RGB for MediaPipe processing
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Locking the image to prevent MediaPipe from writing on it, improving performance
        image.flags.writeable = False

        # Processing the image to detect the pose and store the result
        results = pose.process(image)

        # Unlocking the image and converting the color back from RGB to BGR to display it properly
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Attempting to extract landmarks from the result of the pose detection
        try:
            landmarks = results.pose_landmarks.landmark

            # Acquiring the x and y coordinates of the left shoulder, elbow, and wrist landmarks
            shoulder_left = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                        landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            elbow_left = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                     landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
            wrist_left = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                     landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]

            # Acquiring the x and y coordinates of the right shoulder, elbow, and wrist landmarks
            shoulder_right = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                             landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
            elbow_right = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                          landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
            wrist_right = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                          landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]

            # Calculating the angle at the elbow using the coordinates of the shoulder, elbow, and wrist
            angle_left = calculate_angle(shoulder_left, elbow_left, wrist_left)
            angle_right = calculate_angle(shoulder_right, elbow_right, wrist_right)

            # Call upon the incorrect form function each tick
            detect_bad_form(image, shoulder_left, elbow_left, wrist_left)
            detect_bad_form(image, shoulder_right, elbow_right, wrist_right)

            # Displaying the calculated angle near the left elbow joint on the image
            cv2.putText(image, str(angle_left),
                        tuple(np.multiply(elbow_left, [640, 480]).astype(int)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                        )

            # Displaying the calculated angle near the right elbow joint on the image
            cv2.putText(image, str(angle_right),
                        tuple(np.multiply(elbow_right, [640, 480]).astype(int)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                        )

            # Logic to track the curl count based on the angle of the left elbow
            if angle_left > 140:
                stage_left = "down"
            if angle_left < 50 and stage_left == 'down':
                stage_left = "up"
                counter_left += 1  # Incrementing the count when the arm returns to the 'up' position
                counterL_list.append(counter_left)  # Storing the count in a list
                # Calculating the time elapsed since the last repetition
                gap = (time.time() - epoch)
                gapL_list.append(gap)  # Storing the time gap, since epoch, in a list
                # Printing the current count and time gap to the console for debugging/information
                print(counter_left, gap)

        # Logic to track the curl count based on the angle of the right elbow
            if angle_right > 140:
               stage_right = "down"
            if angle_right < 50 and stage_right == 'down':
               stage_right = "up"
               counter_right += 1  # Incrementing the count when the arm returns to the 'up' position
               counterR_list.append(counter_right)  # Storing the count in a list
               # Calculating the time elapsed since the last repetition
               gap = (time.time() - epoch)
               gapR_list.append(gap)  # Storing the time gap, since epoch, in a list
               # Printing the current count and time gap to the console for debugging/information
               print(counter_right, gap)


        except:
            # If no landmarks are detected, skip the current frame
            pass

        # Creating a rectangle on the image to display the count and stage of the curl
        cv2.rectangle(image, (0, 0), (225, 73), (245, 117, 16), -1)
        cv2.rectangle(image, (350, 0), (1000, 73), (245, 117, 16), -1)

        # Adding text to the rectangle to show the current count
        cv2.putText(image, 'REPS', (400, 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(image, str(counter_left),
                    (395, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(image, 'REPS', (15, 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(image, str(counter_right),
                    (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)


        # Adding text to the rectangle to show the current stage ('up' or 'down')
        cv2.putText(image, 'STAGE', (490, 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(image, stage_left,
                    (485, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(image, 'STAGE', (65, 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(image, stage_right,
                    (60, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)

        # Drawing the pose landmarks on the image
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                  mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                                  mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
                                  )

        # Displaying the image with the overlayed pose landmarks and the curl counter
        cv2.imshow('Mediapipe Feed', image)

        # Waiting for a short interval for a key press, and breaking the loop if 'q' is pressed
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    # Graph Plotting

    #  Calculate the gradient (slope) of reps over time
    if len(gapL_list) > 0 and len(counterL_list) > 0:
        gradient_left = np.polyfit(gapL_list, counterL_list, 1)[0]
    if len(gapR_list) > 0 and len(counterR_list) > 0:
        gradient_right = np.polyfit(gapR_list, counterR_list, 1)[0]

    # Begin plotting the graph for the workout session data for left arm
    plt.plot(gapL_list, counterL_list, label=f" Left Arm - Gradient: {gradient_left:.2f}")  # Plot a line graph of time vs reps alongside a gradient label
    plt.scatter(gapL_list, counterL_list)  # Add a scatter plot on top of the line graph for individual data points

    # Begin plotting the graph for the workout session data for right arm
    plt.plot(gapR_list, counterR_list, label=f" Right Arm - Gradient: {gradient_right:.2f}")  # Plot a line graph of time vs reps alongside a gradient label
    plt.scatter(gapR_list, counterR_list)  # Add a scatter plot on top of the line graph for individual data points


    # Displaying recommendation based on the gradient value for left arm
    if gradient_left > 1:  # If the gradient is steep, suggesting quick rep completion
        recommendation_left = "Increase intensity for more challenge - Left"
    elif gradient_left < 0.5:  # If the gradient is shallow, suggesting slower rep completion
        recommendation_left = "Consider reducing intensity for more reps - Left"
    elif gradient_left >= 0.5 and gradient_left <= 1:
        recommendation_left = "Perfect intensity - Left"  # If gradient is within correct range, suggest no changes

    # Annotating the graph with the recommendation
    # Positioning it at the bottom right of the graph for clear visibility
    plt.annotate(recommendation_left, (0.82, 0.03), xycoords="axes fraction", ha="center", va="center", fontsize=6,
                 bbox=dict(boxstyle="round", alpha=0.5))  # Setting text style and background

    # Displaying recommendation based on the gradient value for right arm
    if gradient_right > 1:  # If the gradient is steep, suggesting quick rep completion
        recommendation_right = "Increase intensity for more challenge - Right"
    elif gradient_right < 0.5:  # If the gradient is shallow, suggesting slower rep completion
        recommendation_right = "Consider reducing intensity for more reps - Right"
    elif gradient_right >= 0.5 and gradient_left <= 1:
        recommendation_right = "Perfect intensity - Right"  # If gradient is within correct range, suggest no changes

    # Annotating the graph with the recommendation
    # Positioning it at the bottom right of the graph for clear visibility, above the one for the left arm
    plt.annotate(recommendation_right, (0.82, 0.07), xycoords="axes fraction", ha="center", va="center", fontsize=6,
                 bbox=dict(boxstyle="round", alpha=0.5))  # Setting text style and background

    plt.title("Arm Training Assistant Data")  # Set the title of the graph to indicate the content
    plt.xlabel("Time(seconds)")  # Label the x-axis to show that it represents the time elapsed
    plt.ylabel("Reps")  # Label the y-axis to show that it represents the number of repetitions

    plt.grid(True)  # Enable a grid for easier visualization of data points
    plt.tight_layout()  # Adjust the layout to ensure there is no content overlap
    plt.legend()

    # Connect the scroll event to the zoom function
    plt.connect('scroll_event', on_scroll)

    # Release the resources held by OpenCV
    cap.release()  # Release the webcam
    cv2.destroyAllWindows()  # Close all OpenCV windows

    # Ask the user if they want to save the graph
    save_file = messagebox.askyesno("Save Graph", "Do you want to save the graph?")

    # Proceed only if the user clicked "Yes"
    if save_file:
        # Define the directory to save the figure
        history_dir = "History"
        if not os.path.exists(history_dir):
            os.makedirs(history_dir)

        # Generate default filename based on current date and time
        default_filename = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S.png")
        print(f"Default filename for the graph: '{default_filename}'")

        # Ask the user for a custom filename or to use the default through a popup window
        user_filename = simpledialog.askstring("Save Graph",
                                               "Enter a filename to save the graph (leave blank to use the default):")
        filename = user_filename if user_filename else default_filename

        # Complete file path
        file_path = os.path.join(history_dir, filename)

        # Save the figure
        plt.savefig(file_path, dpi=300)
        print(f"Graph saved as: {file_path}")

    plt.show()  # Display the plotted graph in a window