import cv2
import numpy as np
from os import listdir
from os.path import isfile, join

isFirst = True

cap = cv2.VideoCapture(0)
cap3 = None

def detect_person():
    if not isFirst:
        cap3.release()
        cv2.VideoCapture(0)

    # Load Yolo
    net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
    classes = []
    with open("coco.names", "r") as f:
        classes = [line.strip() for line in f.readlines()]
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    colors = np.random.uniform(0, 255, size=(len(classes), 3))
    detect_cnt = 0
    while True:
        # Loading image
        ret, img = cap.read()
        #img = cv2.imread("room_ser.jpg")
        img = cv2.resize(img, None, fx=1.8, fy=2)
        height, width, channels = img.shape

        # Detecting objects
        blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

        net.setInput(blob)
        outs = net.forward(output_layers)

        # Showing informations on the screen
        class_ids = []
        confidences = []
        boxes = []
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:
                    # Object detected
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)

                    # Rectangle coordinates
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
        print(indexes)
        font = cv2.FONT_HERSHEY_PLAIN
        label = ''
        for i in range(len(boxes)):
            if i in indexes:
                x, y, w, h = boxes[i]
                label = str(classes[class_ids[i]])
                # label -> name of object that detected in a frame
                color = colors[i]
                cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
                cv2.putText(img, label, (x, y + 30), font, 3, color, 3)
        if label == 'person':
            detect_cnt += 1
        if cv2.waitKey(1) == 13:
            return 'exit'
        if detect_cnt >= 6:
            return 'person'
        cv2.imshow("Image", img)


def detect_face():
    isFirst = False
    cap.release()
    cv2.VideoCapture(0)
    data_path = 'faces/'

    onlyfiles = [f for f in listdir(data_path) if isfile(join(data_path, f))]

    Training_Data, Labels = [], []

    for i, files in enumerate(onlyfiles):
        image_path = data_path + onlyfiles[i]
        images = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        Training_Data.append(np.asarray(images, dtype=np.uint8))
        Labels.append(i)

    Labels = np.asarray(Labels, dtype=np.int32)

    model = cv2.face.LBPHFaceRecognizer_create()

    model.train(np.asarray(Training_Data), np.asarray(Labels))

    print("사람 감지, 안면인식 시작")

    face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


    def face_detector(img, size=0.5):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_classifier.detectMultiScale(gray, 1.3, 5)

        if faces is ():
            return img, []

        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 255), 2)
            roi = img[y:y + h, x:x + w]
            roi = cv2.resize(roi, (200, 200))

        return img, roi


    cap3 = cv2.VideoCapture(0)
    while True:

        ret, frame = cap3.read()

        gsm_image, face = face_detector(frame)

        try:
            face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
            result = model.predict(face)

            if result[1] < 500:
                confidence = int(100 * (1 - (result[1]) / 300))
                display_string = str(confidence) + '% similar'
            cv2.putText(gsm_image, display_string, (100, 120), cv2.FONT_HERSHEY_COMPLEX, 1, (250, 120, 255), 2)

            if confidence > 75:
                cv2.putText(gsm_image, "GSM Person.", (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
                cv2.imshow('observer', gsm_image)

            else:
                cv2.putText(gsm_image, "outsider", (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
                cv2.imshow('observer', gsm_image)

        except:
            cv2.putText(gsm_image, "can not found face", (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 2)
            cv2.imshow('observer', gsm_image)
            pass

        if cv2.waitKey(1) == 13:
            return 'exit'
        elif cv2.waitKey(1) == 27:
            return 'restart'


def finish():
    cv2.destroyAllWindows()


def run():
    while True:
        state = detect_person()
        if state == 'person':
            state = detect_face()
            if state == 'restart':
                state = detect_person()
            elif state == 'exit':
                pass
        elif state == 'exit':
            pass


# execute session
while True:
    run()


