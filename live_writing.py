import cv2
import numpy as np
import torch
import torch.nn as nn  
import torch.nn.functional as F 
from torchvision import transforms
from model import CNN
import json

loadFromSys = True
try:
    model = CNN()
    model.load_state_dict(torch.load('cnn_mnist.pth'))
    model.eval()
    print("CNN model loaded successfully from cnn_mnist.pth")
except FileNotFoundError:
    print("Error: cnn_mnist.pth not found. Digit recognition will not work.")
    model = None
except RuntimeError as e:
    print(f"Error loading CNN model: {e}. Check model architecture and saved state_dict.")
    model = None

def load_stats(stats_path='stats.json'):
    with open(stats_path, 'r') as f:
        stats = json.load(f)
    return stats['mean'], stats['std']

mean, std = load_stats('stats.json')

preprocess = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Grayscale(),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((mean,), (std,))
])

if loadFromSys:
    hsv_value = np.load('hsv_value.npy')
    print(hsv_value)

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4,720)
kernel = np.ones((5, 5), np.int8)

canvas = np.zeros((720, 1280, 3), dtype=np.uint8)

x1 = 0
y1 = 0
noise_thresh = 800
prediction_text_to_display = "" 

while True:
    _, frame = cap.read()
    frame = cv2.flip(frame, 1)

    # if canvas is not None:
    #   canvas = np.zeros_like(frame)

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    if loadFromSys:
        lower_range = hsv_value[0]
        upper_range = hsv_value[1]

    mask = cv2.inRange(hsv, lower_range, upper_range)

    mask = cv2.erode(mask, kernel, iterations = 2)
    mask = cv2.dilate(mask, kernel, iterations = 2)

    contours, heirarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours  and cv2.contourArea(max(contours, key = cv2.contourArea)) > noise_thresh:
        c = max(contours, key = cv2.contourArea)
        x2, y2 ,w, h = cv2.boundingRect(c)

        if x1 == 0 and y1 == 0:
            x1,y1 = x2,y2
        else:
            canvas = cv2.line(canvas, (x1,y1), (x2,y2), [255,255,255], 25)

        x1,y1 = x2,y2

    else:
        x1,y1 = 0, 0

    frame = cv2.add(canvas, frame)
    stacked = np.hstack((canvas, frame))

    key = cv2.waitKey(1) 

    if key == 10 or key == ord('\r') or key == ord('\n'): 
        break 
    elif key == ord('p'): 
        if model is not None:
            print("Predicting digit...")
            canvas_gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
            try:
                image_tensor = preprocess(canvas_gray).unsqueeze(0)
            except Exception as e:
                print(f"Preprocessing error: {e}")
                continue
            with torch.no_grad():
                output = model(image_tensor)
                prediction = torch.argmax(output, dim=1).item()
                prediction_text_to_display = f"Predicted Digit: {prediction}" 
            print(f"Predicted Digit: {prediction}")
        else:
            prediction_text_to_display = "CNN model not loaded."
            print("CNN model not loaded. Cannot predict digit.")
        processed_image_numpy = image_tensor.squeeze().cpu().numpy()
        processed_image_display = (processed_image_numpy) * 255
        processed_image_display = np.clip(processed_image_display, 0, 255).astype(np.uint8)
        processed_image_display = cv2.resize(processed_image_display, (140, 140))
        cv2.imshow("Preprocessed Digit", processed_image_display) 
    elif key == ord('c'):  
        print("check")
        canvas = np.zeros((720, 1280, 3), dtype=np.uint8)
        prediction_text_to_display = "" 
    elif key == 27:  
        break 

   
    if prediction_text_to_display:
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7  
        font_color = (255, 255, 255) 
        font_thickness = 2
        text_position = (10, 30) 
        stacked_with_text = stacked.copy()
        cv2.putText(stacked_with_text, prediction_text_to_display, text_position, font, font_scale, font_color, font_thickness, cv2.LINE_AA)
        stacked = stacked_with_text 
    cv2.imshow('Screen_Pen', cv2.resize(stacked, None, fx = 0.6, fy = 0.6))


cv2.destroyAllWindows()
cap.release()