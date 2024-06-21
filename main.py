import cv2
cap = cv2.VideoCapture('input/car_mov1.mp4')


car_cascade = cv2.CascadeClassifier('model/haarcascade_cars.xml')

car_counter = 0

while True:

    ret, frame = cap.read()

    if not ret:
      
        break

    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    
    cars = car_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    for (x, y, w, h) in cars:
       
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        if y < 400 and y + h > 400:
            car_counter += 1

    cv2.line(frame, (0, 400), (700, 400), (255, 255, 0), 2)

    
    cv2.putText(frame, f"Count: {car_counter}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()