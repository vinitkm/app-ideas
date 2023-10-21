#please add some photos containing human faces with different reaction and renamed them to  photo (1), photo (2).. in the same directory to get the output.

import cv2
from deepface import DeepFace
import os

def get_image_files(directory):
    image_files = []
    for file in os.listdir(directory):
        if file.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_files.append(file)
    return image_files

def select_image(image_files):
    print("Available images in the directory:")
    for idx, file in enumerate(image_files):
        print(f"{idx + 1}. {file}")

    while True:
        try:
            user_choice = int(input("Select the image (enter the number): "))
            if 1 <= user_choice <= len(image_files):
                break
            else:
                print("Invalid selection. Please try again.")
        except ValueError:
            print("Invalid input. Please enter a number.")

    return image_files[user_choice - 1]

def load_image(image_path):
    image = cv2.imread(image_path)
    cv2.imshow('Static Image', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return image

def detect_emotion(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = faceCascade.detectMultiScale(gray, 1.1, 4)

    for (x, y, w, h) in faces:
        face_image = image[y:y+h, x:x+w]
        results = DeepFace.analyze(face_image, actions=('emotion'), enforce_detection=False)

        # Retrieve dominant emotion and display on the terminal
        if len(results) > 0:
            emotion = results[0]['dominant_emotion']
            print("Detected Emotion:", emotion)
            return emotion

    print("No faces detected or no emotion found in the selected image.")
    return None

def main():
    current_directory = "./"
    image_files = get_image_files(current_directory)

    if not image_files:
        print("No image files found in the current directory.")
        return

    selected_image = select_image(image_files)
    image_path = os.path.join(current_directory, selected_image)

    image = load_image(image_path)
    detect_emotion(image)

if __name__ == "__main__":
    main()
