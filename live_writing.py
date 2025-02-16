import cv2
import numpy as np
import torch
import torch.nn as nn  # You might not need this line directly anymore, but it's safe to keep
import torch.nn.functional as F # Same here
from torchvision import transforms
from model import CNN

loadFromSys = True
try:
    model = CNN() # Now it will use the CNN class imported from model.py
    model.load_state_dict(torch.load('cnn_mnist.pth'))
    model.eval()
    print("CNN model loaded successfully from cnn_mnist.pth")
except FileNotFoundError:
    print("Error: cnn_mnist.pth not found. Digit recognition will not work.")
    model = None
except RuntimeError as e:
    print(f"Error loading CNN model: {e}. Check model architecture and saved state_dict.")
    model = None

preprocess = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Grayscale(),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
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

while True:
	_, frame = cap.read()
	frame = cv2.flip(frame, 1)

	# if canvas is not None:
	# 	canvas = np.zeros_like(frame)

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
			canvas = cv2.line(canvas, (x1,y1), (x2,y2), [255,255,255], 20)

		x1,y1 = x2,y2
	
	else:
		x1,y1 = 0, 0

	frame = cv2.add(canvas, frame)

	stacked = np.hstack((canvas, frame))
	cv2.imshow('Screen_Pen', cv2.resize(stacked, None, fx = 0.6, fy = 0.6))

	key = cv2.waitKey(1)  # Call waitKey ONCE and store the result

	if key == 10 or key == ord('\r') or key == ord('\n'): # Check for Enter key (using or to be safe)
		break  # Exit loop on Enter
	elif key == ord('p'):  # Check for 'p' key
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
			print(f"Predicted Digit: {prediction}")
		else:
			print("CNN model not loaded. Cannot predict digit.")
	elif key == ord('c'):  # Check for 'c' key
		print("check")
		canvas = np.zeros((720, 1280, 3), dtype=np.uint8) # Clear canvas
	elif key == 27:  # Check for ESC key
		break # Exit loop on ESC


cv2.destroyAllWindows()
cap.release()