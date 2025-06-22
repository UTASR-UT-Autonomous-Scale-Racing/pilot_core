from pilot_vision.models.threshold.threshold import Threshold
import cv2

#model = Threshold(model_path=None, num_classes=0)
#cap = cv2.VideoCapture('/home/zebrafish/Desktop/0_UTASR/pilot_core/temp/PXL_20250601_205921114.mp4')
#
#while cap.isOpened():
#    ret, frame = cap.read()
#    if not ret:
#        break
#    frame = cv2.resize(frame, (640, 400))
#    steering, throttle, masked_frame = model.get_action(frame)
#    print(f"Steering: {steering}, Throttle: {throttle}")
#    cv2.imshow("Masked Frame", masked_frame)
#    if cv2.waitKey(1) & 0xFF == ord('q'):
#        break
#
#cap.release()
#cv2.destroyAllWindows()

model = Threshold(model_path=None, num_classes=0)
frame = cv2.imread('/home/zebrafish/Desktop/0_UTASR/pilot_core/temp/PXL_20250601_210004317.jpg')
frame = cv2.resize(frame, (640, 400))
steering, throttle, masked_frame = model.get_action(frame)
print(f"Steering: {steering}, Throttle: {throttle}")
cv2.imshow("Masked Frame", masked_frame)
cv2.waitKey(0)
cv2.destroyAllWindows()