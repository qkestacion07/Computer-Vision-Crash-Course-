import cv2
import os
##Provides input for the face recognition 
##with the help of face detection

def clear_folder(folder):
    """
    Deletes all images in the folder
    This assumes there are only files and 
    no directories in the folder
    """
    print(f"{folder} cleared")
    for file in os.listdir(folder):
        file_path = os.path.join(folder, file)
        if os.path.isfile(file_path):
            os.remove(file_path)

def scan_files(folder_path, save_folder):
    """"Takes the roi of the image and saves it"""
    clear_folder(save_folder)
    image_files = []
    video_files = []
    for file_name in os.listdir(folder_path):
        if file_name.endswith(('.png', '.jpg', '.jpeg')):
            image_files.append(os.path.join(folder_path, file_name))
        elif file_name.endswith(('.mp4','.mkv')):
            video_files.append(os.path.join(folder_path, file_name))
    image_detect(image_files, save_folder)
    video_detect(video_files, save_folder)

def video_detect(video_files, save_folder):
    print("Detecting video")
    face_cascade = cv2.CascadeClassifier('Activity7_FaceRecognition/haarcascade_frontalface_default.xml')
    
    j = 1 #counter of saved images
    
    for file in video_files:
        video = cv2.VideoCapture(file)
        if not video.isOpened():
            print(f"Error: Could not open video file {file}")
            continue

        while True:
            ret, frame = video.read()
            if not ret:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            faces = face_cascade.detectMultiScale(gray, 1.1, 5)

            for (x, y, w, h) in faces:
                roi_gray = gray[y:y + h, x:x + w]
                cv2.imwrite(os.path.join(save_folder, f'image_{j}.jpg'), roi_gray)
                j += 1
        video.release()

def image_detect(image_files, save_folder):
    print("Detecting images")
    face_cascade = cv2.CascadeClassifier('Activity7_FaceRecognition/haarcascade_frontalface_default.xml')
    
    i = 1 #counter of input images
    j = 1 #counter of saved images
    n = len(image_files)
    if n==0: return
    while True:
        print(i)
        image = cv2.imread(image_files[i-1])
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray, 1.1, 5)
        if len(faces) == 0:
            print(f"No faces detected in {image_files[i-1]}")
         
        for (x, y, w, h) in faces:
            roi_gray = gray[y:y + h, x:x + w]
            cv2.imwrite(os.path.join(save_folder, f'image_{j}.jpg'), roi_gray)
            j += 1
        i += 1
        if i>n:
            break
        
if __name__ == "__main__":
    root = 'Activity7_FaceRecognition/'
    students = [['Marie','Hernandez'],['Kenneth','Estacion'],
                ['Ed','Figueroa'],['Jericho','Ecubin'],
                ['Jose','Figuerras']]
    for (firstname, lastname) in students:
      print(lastname)
      scan_files(root+f'raw_images/{firstname}',root+f'images/{lastname}')
    
