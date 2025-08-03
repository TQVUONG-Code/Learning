import cv2 #thư viện xử lý hình ảnh và video trong Python. Nó cung cấp các chức năng để đọc, ghi và xử lý các hình ảnh từ các nguồn đầu vào khác nhau
import os
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import cvlib #một thư viện xử lý hình ảnh dựa trên OpenCV, cung cấp các công cụ giúp phát hiện khuôn mặt, đồng thời cung cấp chức năng nhận biết giới tính và cảm xúc từ khuôn mặt
import tkinter as tk
from tkinter import *
from PIL import ImageTk, Image
from tkinter import messagebox
import threading

# Load model
# face_classifier = cv2.CascadeClassifier('face_detection.xml')
gender_model = load_model('Gender1.h5')
emotion_model = load_model('Emotion1.h5')

gender_labels = ['Male', 'Female']
emotion_labels = ['Neutral', 'Happy', 'Sad', 'Surprised', 'Angry']

# Create tkinter window
root = tk.Tk()
root.geometry('1050x620')
root.resizable(False, False)
root.title('University of Economics Ho Chi Minh City')

icon = PhotoImage(file='img/Logo_UEH_xanh.png')
root.iconphoto(True, icon)


is_running = False

def use_camera():
    global is_running
    is_running = True
    start_button.config(state="disabled")
    stop_button.config(state="normal")
    exit_button.config(state="normal")

    worker_thread = threading.Thread(target=camera_worker)
    worker_thread.start()
    
def quit_program():
    answer = messagebox.askyesno("Quit", "Do you want to exit?")
    if answer:
        root.destroy()

def cancel_feed():
    global is_running
    is_running = False
    start_button.config(state="normal")
    stop_button.config(state="disabled")

def camera_worker():
    capture = cv2.VideoCapture(0)

    while is_running:
        ret, frame = capture.read()

        # Face detection
        faces, confidences = cvlib.detect_face(frame)

        for face, confidence in zip(faces, confidences):
            # Get the coordinates of the face rectangle
            (startX, startY) = face[0], face[1]
            (endX, endY) = face[2], face[3]

            # Draw rectangle around the face
            cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 0, 255), 2) #BGR

            # Crop the detected face region
            face_crop = np.copy(frame[startY:endY, startX:endX])

            if face_crop.shape[0] < 10 or face_crop.shape[1] < 10:
                continue

            # Preprocess the face for gender prediction
            face_crop = cv2.resize(face_crop, (150, 150))
            face_crop = face_crop.astype("float") / 255.0
            face_crop = img_to_array(face_crop)
            face_crop = np.expand_dims(face_crop, axis=0)

            # Predict gender
            conf_model_gender = gender_model.predict(face_crop)[0]
            idx_model_gender = np.argmax(conf_model_gender)
            label_model_gender = gender_labels[idx_model_gender]
            
            # Predict emotion
            conf_model_emotion = emotion_model.predict(face_crop)[0]
            idx_model_emotion = np.argmax(conf_model_emotion)
            label_model_emotion = emotion_labels[idx_model_emotion]

            label = "{},{}".format(label_model_gender, label_model_emotion)

            Y = startY - 10 if startY - 10 > 10 else startY + 10

            # Write the predicted gender label on the image
            cv2.putText(frame, label, (startX, Y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        # Convert the image from OpenCV BGR format to PIL Image
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        image = image.resize((640, 480), Image.ANTIALIAS)

        # Convert the PIL Image to ImageTk to display on Tkinter label
        imgtk = ImageTk.PhotoImage(image=image)

        # Update the image on the label
        image_label.configure(image=imgtk)
        image_label.image = imgtk

        if cv2.waitKey(1) & 0xFF == ord('s'):
            break

    capture.release()
    cv2.destroyAllWindows()

# Main frame
main_frame = tk.Frame(root, bg='#faf79b')
main_frame.pack(side=tk.LEFT)
main_frame.pack_propagate(False)
main_frame.configure(width=1050, height=620)

# Title 1
label_title = tk.Label(main_frame, text='PREDICT GENDER AND EMOTION BASED ON HUMAN FACES', 
                       font=("Arial", 20),
                       fg="red",
                       bg='#faf79b')
# Title 2
label_title2 = tk.Label(main_frame, text='Artificial Intelligent in Business', 
                        font=("Arial", 15), 
                        fg="blue",
                        bg='#faf79b')
# Title 3
label_title3 = tk.Label(main_frame, text='Le Thi Huyen', 
                        font=("Arial", 15), 
                        fg="blue",
                        bg='#faf79b')

# Camera frame
image_label = tk.Label(main_frame, bg='#D9EAF4')
image_label.place(x=160, y=110, width=750, height=450)

# Start button
start_button = tk.Button(main_frame, 
                         text="START", 
                         font=('Bold', 15), 
                         fg='white', 
                         bd=0,
                         bg='blue', 
                         command=use_camera)
start_button.place(x=300, y=570, width=80, height=35)

# Stop button
stop_button = tk.Button(main_frame, text="STOP", 
                        font=('Bold', 15),
                        fg='white',
                        bd=0,
                        bg='blue',
                        command=cancel_feed, 
                        state="disabled")
stop_button.place(x=500, y=570, width=80, height=35)

# Exit button
exit_button = tk.Button(main_frame, text="EXIT", 
                        font=('Bold', 15), 
                        fg='white', 
                        bd=0,
                        bg='blue',
                        command=quit_program, 
                        state="normal")
exit_button.place(x=700, y=570, width=80, height=35)

label_title.pack()
label_title2.pack()
label_title3.pack()


root.mainloop()

#save model
realtime.py.save("nhandienkhuonmat.h5")
print("Finish model!")