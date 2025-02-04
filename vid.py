import cv2
import numpy as np

def run():
    cap = cv2.VideoCapture('./assets/vids/8132-207209040.mp4')
    # cap.set(3, 640) # for camera
    # cap.set(4, 480) # for camera
    # cap.set(10, 100) # for camera
    
    
    while True:
        success, image = cap.read()
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # Convert to grayscale / конвертировать в оттенки серого
        cv2.imshow("Video", image)

        lower_red = np.array([150, 150, 50])
        upper_red = np.array([180, 255, 150])
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()