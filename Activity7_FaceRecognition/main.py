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

def detect(folder_path, save_folder):
    """"Takes the roi of the image and saves it"""
    clear_folder(save_folder)
    face_cascade = cv2.CascadeClassifier('Activity7_FaceRecognition/haarcascade_frontalface_default.xml')
    
    image_files = []
    for file_name in os.listdir(folder_path):
        if file_name.endswith(('.png', '.jpg', '.jpeg')):
            image_files.append(os.path.join(folder_path, file_name))
    
    i = 1 #counter of input images
    j = 1 #counter of saved images
    n = len(image_files)
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
    root = 'Activity7_FaceRecognition/images/'
    students = [['Marie','Hernandez'],['Kenneth','Estacion'],
                ['Ed','Figueroa'],['Jericho','Ecubin']]
    for (firstname, lastname) in students:
      print(lastname)
      detect(root+firstname,root+lastname)
