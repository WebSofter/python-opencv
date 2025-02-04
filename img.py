import cv2
import numpy as np
from matplotlib import pyplot as plt
import imutils
import utils.transform as trans
import easyocr

def run():
    # Load the image
    image = cv2.imread("./assets/imgs/src_imgs/_SAILUN_215_70_R15/cap0013.jpg")
    
    ##
    # Color operations
    ##
    # image = cv2.GaussianBlur(image, (5, 5), 3) # (5, 5) is the kernel size and 3 is the sigma / сглаживание
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # Convert to grayscale / конвертировать в оттенки серого
    image = cv2.Canny(image, 100, 100) # Apply Canny edge detection / краевое обнаружение
    kernel = np.ones((3, 3), np.uint8) # Create a 3x3 kernel / создать ядро 3x3
    image = cv2.dilate(image, kernel, iterations=1) # Dilate the edges / расширить края
    image = cv2.erode(image, kernel, iterations=1) # Erode the edges / разрушить края
    
    ##
    # Transform operations
    ##
    # image = cv2.resize(imgage, (0, 500))
    image = cv2.resize(image, (image.shape[1] // 4, image.shape[0] // 4)) # Resize the image to 1/4 of the original size / изменить размер изображения до 1/4 от исходного размера
    image = cv2.flip(image, 0) # Flip the image vertically / перевернуть изображение вертикально
    
    image = trans.rotate(image, 45) # Rotate the image 45 degrees / повернуть изображение на 45 градусов
    image = trans.move(image, 20, 20) # Move the image by x and y pixels / переместить изображение на x и y пикселей
    
    shape = image.shape # (height, width, channels) / (высота, ширина, каналы)
    print(shape)
    
    cv2.imshow("Image", image) # Показать изображение
    cv2.waitKey(0) # Ждать нажатия клавиши
    cv2.destroyAllWindows() # Закрыть все окна
    
    
def run_canvas():
    canvas = np.zeros((500, 500, 3), np.uint8) # Create a black canvas / создать черный холст
    canvas[:] = 255, 255, 155 # Set the canvas to white (BGR) / установить холст в белый цвет
    canvas[100:150,200:280] = 180, 90, 200 # Set a pixel to purple (BGR) / установить пиксель в фиолетовый цвет
    canvas = cv2.line(canvas, (0, 0), (500, 500), (0, 255, 0), 3) # Draw a green line / нарисовать зеленую линию
    canvas = cv2.line(canvas, (0, canvas.shape[0] // 2), (canvas.shape[1], canvas.shape[0]), (255, 0, 0), 3) # Draw a green line / нарисовать зеленую линию
    canvas = cv2.rectangle(canvas, (100, 100), (200, 200), (255, 0, 0), thickness=2) # Draw a blue rectangle / нарисовать синий прямоугольник
    canvas = cv2.circle(canvas, (canvas.shape[1] // 2, canvas.shape[0] // 2), 50, (0, 0, 255), thickness=2) # Draw a red circle / нарисовать красный круг
    canvas = cv2.putText(canvas, "Hello, World!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2) # Write text / написать текст
    
    cv2.imshow("Image", canvas) # Показать изображение
    cv2.waitKey(0) # Ждать нажатия клавиши
    cv2.destroyAllWindows() # Закрыть все окна
    
    
def run_contour():
    image = cv2.imread("./assets/imgs/any/any4.png") # Load the image / загрузить изображение
    new_mage = np.zeros(image.shape, dtype=np.uint8) # Create a black canvas / создать черный холст
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # Convert to grayscale / конвертировать в оттенки серого
    image = cv2.GaussianBlur(image, (5, 5), 0) # (5, 5) is the kernel size and 0 is the sigma / сглажив
    image = cv2.Canny(image, 100, 170) # Пороговые цвета, до которых будут определяться границы
    contours, hierarchy = cv2.findContours(image, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE) # Найти контуры
    cv2.drawContours(new_mage, contours, -1, (200, 190, 150), 1) # Нарисовать контуры
    cv2.imshow("Image", new_mage) # Показать изображение
    cv2.waitKey(0) # Ждать нажатия клавиши
    cv2.destroyAllWindows() # Закрыть все окна
    
    
def run_color():
    image = cv2.imread("./assets/imgs/any/any4.png") # Load the image / загрузить изображение
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV) # Convert to HSV / конвертировать в HSV (в один слой)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB) # Convert to LAB / конвертировать в LAB (в один слой)
    image = cv2.cvtColor(image, cv2.COLOR_LAB2BGR) # Convert back to BGR / конвертировать обратно в BGR (в три слоя)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # Convert to RGB / конвертировать в RGB (в три слоя)
    r, g, b = cv2.split(image) # Split the image into RGB channels / разделить изображение на каналы RGB
    image = cv2.merge((b, g, r)) # Merge the RGB channels / объединить каналы RGB
    
    cv2.imshow("Image", image) # Показать изображение
    cv2.waitKey(0) # Ждать нажатия клавиши
    cv2.destroyAllWindows() # Закрыть все окна
    
    
def run_threshold():
    canvas = np.zeros((500, 500), dtype=np.uint8) # Create a black canvas / создать черный холст
    circle = cv2.circle(canvas.copy(), (0, 0), 80, 255, -1) # Draw a white circle / нарисовать белый круг
    square = cv2.rectangle(canvas.copy(), (0, 0), (200, 200), 355, -1) # Draw a white square / нарисовать белый квадрат
    ellipse = cv2.ellipse(canvas.copy(), (0, 0), (250, 150), 0, 0, 160, 155, -1) # Draw a white ellipse / нарисовать белый эллипс
    # img = cv2.bitwise_and(circle, square) # Perform a bitwise AND operation / выполнить побитовую операцию И
    # img = cv2.bitwise_or(circle, ellipse) # Perform a bitwise OR operation / выполнить побитовую операцию ИЛИ
    # img = cv2.bitwise_xor(circle, square) # Perform a bitwise XOR operation / выполнить побитовую операцию ИСКЛЮЧАЮЩЕЕ ИЛИ
    img = cv2.bitwise_not(circle) # Perform a bitwise NOT operation / выполнить побитовую операцию НЕ
    cv2.imshow("Image", img) # Показать изображение
    cv2.waitKey(0) # Ждать нажатия клавиши
    cv2.destroyAllWindows() # Закрыть все окна
    
def run_mask():
    photo = cv2.imread("./assets/imgs/any/any4.png") # Load the image / загрузить изображение
    canvas = np.zeros(photo.shape[:2], dtype=np.uint8) # Create a black canvas / создать черный холст
    circle = cv2.circle(canvas.copy(), (350, 100), 150, 255, -1) # Draw a white circle / нарисовать белый круг
    square = cv2.rectangle(canvas.copy(), (0, 0), (200, 200), 355, -1) # Draw a white square / нарисовать белый квадрат
    ellipse = cv2.ellipse(canvas.copy(), (0, 0), (250, 150), 0, 0, 160, 155, -1) # Draw a white ellipse / нарисовать белый эллипс
    img = cv2.bitwise_and(photo, photo, mask=circle) # Perform a bitwise AND operation / выполнить побитовую операцию И
    # img = cv2.bitwise_or(circle, ellipse) # Perform a bitwise OR operation / выполнить побитовую операцию ИЛИ
    # img = cv2.bitwise_xor(circle, square) # Perform a bitwise XOR operation / выполнить побитовую операцию ИСКЛЮЧАЮЩЕЕ ИЛИ
    # img = cv2.bitwise_not(circle) # Perform a bitwise NOT operation / выполнить побитовую операцию НЕ
    cv2.imshow("Image", img) # Показать изображение
    cv2.waitKey(0) # Ждать нажатия клавиши
    cv2.destroyAllWindows() # Закрыть все окна
    
    
def run_recognize_face():
    image = cv2.imread("./assets/imgs/hummans/hum1.jpg") # Load the image / загрузить изображение
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # Convert to HSV / конвертировать в HSV (в один слой)
    
    faces = cv2.CascadeClassifier("./assets/models/haarcascade_frontalcatface.xml") # Load the face classifier / загрузить классификатор лица
    results = faces.detectMultiScale(image_gray, scaleFactor=1.1, minNeighbors=2) # Detect faces / найти лица
    cv2.cvtColor(image_gray, cv2.COLOR_GRAY2BGR) # Convert to BGR / конвертировать в BGR (в три слоя)
    
    for (x, y, w, h) in results: # For each face / для каждого лица
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), thickness = 1) # Draw a rectangle around the face / нарисовать прямоугольник вокруг лица
    
    cv2.imshow("Image", image) # Показать изображение
    cv2.waitKey(0) # Ждать нажатия клавиши
    cv2.destroyAllWindows() # Закрыть все окна
    
def run_recognize_number():
    image = cv2.imread("./assets/imgs/vehicle/num1.png") # Load the image / загрузить изображение
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # Convert to HSV / конвертировать в HSV (в один слой)
    
    image_filter = cv2.bilateralFilter(image_gray, 11, 17, 17) # Apply bilateral filter / применить билатеральный фильтр
    edges = cv2.Canny(image_filter, 30, 200) # Apply Canny edge detection / применить Canny edge detection
    # img = cv2.cvtColor(edges, cv2.COLOR_BGR2RGB) # Convert to RGB / конвертировать в RGB (в три слоя)
    
    contour = cv2.findContours(edges.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contour = imutils.grab_contours(contour)
    
    contour = sorted(contour, key=cv2.contourArea, reverse=True)[:8]
    
    position = None
    for c in contour:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            position = approx
            break
    # print(position)
    mask = np.zeros(image_gray.shape, np.uint8)
    image_contour = cv2.drawContours(mask, [position], 0, 255, -1)
    image_bitwise = cv2.bitwise_and(image, image, mask=mask)
    
    x, y = np.where(mask == 255)
    (x1, y1) = (np.min(x), np.min(y))
    (x2, y2) = (np.max(x), np.max(y))
    cropped_image = image_gray[x1:x2+1, y1:y2+1]
    
    text = easyocr.Reader(['en'])
    result = text.readtext(cropped_image)
    label = result[0][-2]
    print(label)
    final_image = cv2.putText(image, label, (x1, y2), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 1)
    final_image = cv2.rectangle(image, (x1, x2), (y1, y2), (0, 255, 0), 2)

    # lower, upper = 150, 254
    # bright = cv2.inRange(image, lower, upper)
    # plt.imshow(bright,cmap='gray')
    
    plt.imshow(cv2.cvtColor(final_image, cv2.COLOR_BGR2RGB)) # Показать изображение
    plt.show() # Показать изображение