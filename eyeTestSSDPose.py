from socket import *
import numpy as np
import urllib.request
import threading
import cv2
from imutils import face_utils
import dlib
import sys
import time
from PIL import ImageFont, ImageDraw, Image
import datetime


def url_to_image(url):
    # url로 이미지 불러오기
    resp = urllib.request.urlopen(url)
    image = np.asarray(bytearray(resp.read()), dtype="uint8")
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    return image


def distance_to_camera_ref(refWidth, refDistance, imgWidth):
    # get distance by using a reference image.
    dist = (refWidth/imgWidth)*refDistance
    # rounding
    dist /= 10
    dist = round(dist)
    dist /= 10
    return dist


def center_eye(eye_points):
    # get the center of eye.
    x1, y1 = np.amin(eye_points, axis=0)
    x2, y2 = np.amax(eye_points, axis=0)
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    return int(cx), int(cy)


def crop_eye(img, eye_points):
    x1, y1 = np.amin(eye_points, axis=0)
    x2, y2 = np.amax(eye_points, axis=0)
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    w = (x2 - x1) * 1
    h = w
    margin_x, margin_y = w / 2, h / 2
    min_x, min_y = int(cx - margin_x), int(cy - margin_y)
    max_x, max_y = int(cx + margin_x), int(cy + margin_y)
    eye_rect = np.rint([min_x, min_y, max_x, max_y]).astype(np.int)
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    w = (x2 - x1) * 1.2
    h = w
    margin_x, margin_y = w / 2, h / 2
    min_x, min_y = int(cx - margin_x), int(cy - margin_y)
    max_x, max_y = int(cx + margin_x), int(cy + margin_y)
    eye_r = np.rint([min_x, min_y, max_x, max_y]).astype(np.int)
    eye_img = img[eye_rect[1]:eye_rect[3], eye_rect[0]:eye_rect[2]]
    only_eyes = img[eye_r[1]:eye_r[3], eye_r[0]:eye_r[2]]
    return eye_img, eye_rect,only_eyes


def face_detect(cap):
    # @param  cap - capture object.
    eye_dist = 0
    face_height = 0
    list = []
    # if camera is opened
    if cap.isOpened:
        # frame - image,   ret - success(true) or fail(false)
        ret, frame = capture.read()
        img = frame
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # cv2 blobFromImage.   (src, scale factor, dest size, mean value, swapRB, crop)
        # output will be blob.[ 4 dimensional. (imageID, channels, width, height) ]
        blob = cv2.dnn.blobFromImage(img, 1.0, (300, 300), [104, 117, 123], False, False)
        frameWidth = img.shape[1]
        frameHeight = img.shape[0]
        # prediction.  (most of time is used)
        net.setInput(blob)
        detections = net.forward()

        bboxes = []
        conf_threshold = 0.3
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > conf_threshold:
                x1 = int(detections[0, 0, i, 3] * frameWidth)
                y1 = int(detections[0, 0, i, 4] * frameHeight)
                x2 = int(detections[0, 0, i, 5] * frameWidth)
                y2 = int(detections[0, 0, i, 6] * frameHeight)

                left = x1
                top = y1
                right = x2
                bottom = y2

                dlibRect = dlib.rectangle(left, top, right, bottom)
                bboxes.append(dlibRect)

        if len(bboxes) == 1:
            for face in bboxes:
                shapes = predictor(gray, face)
                shapes = face_utils.shape_to_np(shapes)

                # eye to eye distance
                left_cx, left_cy = center_eye(eye_points=shapes[36:42])
                right_cx, right_cy = center_eye(eye_points=shapes[42:48])
                eye_dist = np.sqrt((right_cx - left_cx) ** 2 + (right_cy - left_cy) ** 2)
                # face height
                face_width = face.right() - face.left()
                face_height = face.bottom() - face.top()

                # eye check
                eye_img_l, eye_rect_l, only_eye_l = crop_eye(gray, eye_points=shapes[36:42])
                eye_img_r, eye_rect_r, only_eye_r = crop_eye(gray, eye_points=shapes[42:48])

            img = frame.copy()
            #eye box
            cv2.rectangle(img, pt1=tuple(eye_rect_l[0:2]), pt2=tuple(eye_rect_l[2:4]), color=(255, 255, 255),
                          thickness=2)
            cv2.rectangle(img, pt1=tuple(eye_rect_r[0:2]), pt2=tuple(eye_rect_r[2:4]), color=(255, 255, 255),
                          thickness=2)
            #eye to eye
            cv2.arrowedLine(img, (left_cx, left_cy), (right_cx, right_cy), (255, 255, 0), thickness=4,
                            tipLength=0.2)
            cv2.arrowedLine(img, (right_cx, right_cy), (left_cx, left_cy), (255, 255, 0), thickness=4,
                            tipLength=0.2)
            #face
            cv2.rectangle(img, (face.left(), face.top()), (face.right(), face.bottom()), color=(0, 255, 0),
                          thickness=2)
            list = [eye_img_l, eye_rect_l, only_eye_l, eye_img_r, eye_rect_r, only_eye_r]
    return frame, img, gray, eye_dist, face_height, list


def dist(p1, p2):
    return np.sqrt(((p1[0] - p2[0]) ** 2) + ((p1[1] - p2[1]) ** 2))


def takePose(img, net):
    THRESHOLD = 1.2
    BODY_PARTS = {"Head": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
                  "LShoulder": 5, "LElbow": 6, "LWrist": 7}

    POSE_PAIRS = [["Head", "Neck"], ["Neck", "RShoulder"], ["RShoulder", "RElbow"],
                  ["RElbow", "RWrist"], ["Neck", "LShoulder"], ["LShoulder", "LElbow"],
                  ["LElbow", "LWrist"]]
    lists = {0: "Head", 1: "Neck", 4: "RWrist", 7: "LWrist"}
    left = True
    right = True
    frame = img.copy()
    # frame.shape = 불러온 이미지에서 height, width, color 받아옴
    start_time = time.time()
    imageHeight, imageWidth, _ = frame.shape
    inHeight, inWidth = 240, 180

    # network에 넣기위해 전처리
    # frame, scalefactor, size, mean(subtraction), channel swap(BGR -> RGB), image crop
    inpBlob = cv2.dnn.blobFromImage(frame, 1.0 / 255, (inWidth, inHeight), (0, 0, 0), swapRB=False, crop=False)

    net.setInput(inpBlob)
    # 결과 받아오기
    output = net.forward()

    # output -> 0 - image ID, 1 - fields, 2 - height, 3 - width
    # output fileds -> 0 ~ 18 - confidence maps, 19~56 - PAF(part affinity fields) [ 2d vector -> 2 * 19 ]

    H = output.shape[2]
    W = output.shape[3]
    # 키포인트 검출시 이미지에 그려줌
    points = []
    x0, y0 = 0, 0
    dist_0_to_1 = sys.maxsize
    dist_0_to_4 = sys.maxsize
    dist_0_to_7 = sys.maxsize
    for i in range(0, 8):
        # 해당 신체부위 신뢰도 얻음.
        probMap = output[0, i, :, :]
        # global 최대값 찾기
        # we assume that there is only one person in the image. In this case,
        # we can easily find the pose points by using maximum value(confidence map).
        minVal, prob, minLoc, point = cv2.minMaxLoc(probMap)
        # 원래 이미지에 맞게 점 위치 변경
        x = (imageWidth * point[0]) / W
        y = (imageHeight * point[1]) / H
        if i == 0:
            x0, y0 = x, y
        elif i == 1:
            dist_0_to_1 = dist((x0, y0), (x, y))
        elif i == 4:
            dist_0_to_4 = dist((x0, y0), (x, y))
        elif i == 7:
            dist_0_to_7 = dist((x0, y0), (x, y))
        if (dist_0_to_4 < dist_0_to_1 * THRESHOLD) and (dist_0_to_7 < dist_0_to_1 * THRESHOLD):
            right = False
            left = False
            res = "2 eye close"
        # 인식하는 시간
        #
        elif dist_0_to_4 < dist_0_to_1 * THRESHOLD:
            right= False
            res = "left open"
        elif dist_0_to_7 < dist_0_to_1 * THRESHOLD:
            left = False
            res = "right open"
        else:
            res = "2 eye open"
            right = True
            left = True
        # 키포인트 검출한 결과가 0.1보다 크면(검출한곳이 위 BODY_PARTS랑 맞는 부위면) points에 추가, 검출했는데 부위가 없으면 None으로
        if prob > 0.1:
            color = (180, 180, 180)
            if i in lists:
                cv2.putText(frame, "{}".format(lists[i]), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1,lineType=cv2.LINE_AA)
                color = (0, 0, 255)
            cv2.circle(frame, (int(x), int(y)), 3, color, thickness=-1,lineType=cv2.FILLED)  # circle(그릴곳, 원의 중심, 반지름, 색)
            points.append((int(x), int(y)))
        else:
            points.append(None)
    for pair in POSE_PAIRS:
        partA = pair[0]  # Head
        partA = BODY_PARTS[partA]  # 0
        partB = pair[1]  # Neck
        partB = BODY_PARTS[partB]  # 1
        color = (180, 180, 180)
        if points[partA] and points[partB]:
            if partA == 0 and partB == 1:
                color = (0, 0, 255)
            cv2.line(frame, points[partA], points[partB], color, 2)

    # "HEAD", "NECK",  "HEAD", RWrist,  HEAD, "LWrist"
    if points[0] != (0, 0):
        cv2.line(frame, points[0], points[7], (0, 255, 0), 2)
        cv2.line(frame, points[0], points[4], (255, 0, 0), 2)

    finish_time = time.time() - start_time
    cv2.putText(frame, "time %.2f sec" % finish_time, (15, frame.shape[0] - 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2,
                lineType=cv2.LINE_AA)
    cv2.putText(frame, "Head-Neck %.1f (pixel distance)" % (dist_0_to_1 * THRESHOLD), (15, frame.shape[0] - 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2,
                lineType=cv2.LINE_AA)
    cv2.putText(frame, "Head-Left %.1f" % dist_0_to_7, (15, frame.shape[0] - 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0),
                2 ,lineType=cv2.LINE_AA)
    cv2.putText(frame, "Head-Right %.1f" % dist_0_to_4, (15, frame.shape[0] - 30), cv2.FONT_HERSHEY_SIMPLEX, 1,
                (255, 0, 0), 2, lineType=cv2.LINE_AA)
    cv2.putText(frame, "{}".format(res), (frame.shape[1]- 200, frame.shape[0] - 30), cv2.FONT_HERSHEY_SIMPLEX, 1,
                (255, 255, 0), 2, lineType=cv2.LINE_AA)

    return frame, left, right


def put_text(_img, txt):
    image2 = Image.fromarray(_img)
    draw = ImageDraw.Draw(image2)
    draw.text((15, 360), txt, font=ImageFont.truetype("./batang.ttc", 30), fill=(0, 0, 0))
    return np.array(image2)


def opencv():
    global clientSock, check_command, com, capture, eye_rect_l, eye_rect_r, net2
    status = -1
    distance = 0
    distance1 = 0
    distance2 = 0
    mean_eye_dist = 0.0
    mean_face_height = 0.0
    face_start = False
    x1 = 0
    y1 = 0
    x2 = 0
    y2 = 0

    main_img = cv2.imread("./design/main.png")
    face_start_img = cv2.imread("./design/face_start.png")
    face_img = cv2.imread("./design/face.png")
    dist_img = cv2.imread("./design/dist.png")
    left_eye_img = cv2.imread("./design/left_eye.png")
    right_eye_img = cv2.imread("./design/right_eye.png")
    result_img = cv2.imread("./design/result.png")
    prob_img = cv2.imread("./design/problem.png")

    whiteBackGround = np.full((480, 640, 3), 255, dtype=np.uint8)
    newimg = np.concatenate((main_img, whiteBackGround), axis=1)
    cv2.imshow("eyeTest", newimg)
    cv2.waitKey(300)

    while True:
        ori_img, img, gray, eye_dist, face_height, _ = face_detect(capture)

        if status == 0 and eye_dist > 0:
            distance1 = distance_to_camera_ref(mean_eye_dist, KNOWN_DISTANCE, eye_dist)
            distance2 = distance_to_camera_ref(mean_face_height, KNOWN_DISTANCE, face_height)
            distance = distance1


        if check_command == True:  # backend에서 찍으라고 데이터가 옴
            if com == 'takeFac':
                com = ''
                check_command = False
                mean_cnt = 0
                print('얼굴확인')

                while True:
                    ori_img, img, gray, eye_dist, face_height, eye_list = face_detect(capture)
                    newimg = np.concatenate((face_img, img), axis=1)
                    cv2.imshow("eyeTest", newimg)
                    cv2.waitKey(50)
                    if eye_dist == 0:
                        continue

                    mean_eye_dist += eye_dist
                    mean_face_height += face_height
                    mean_cnt += 1
                    if mean_cnt > 10:
                        mean_eye_dist /= mean_cnt
                        mean_face_height /= mean_cnt
                        print(mean_face_height)
                        print(mean_eye_dist)
                        result_face = ori_img
                        break

                clientSock.sendall('check'.encode())
                status = 0

            elif com == 'takeDis':  # 거리 인식해야하는 명령인지
                com = ''
                check_command = False
                print('거리인식', distance)

                label = str(int(round(distance)))
                print(label)
                #label = str(2)
                clientSock.sendall(label.encode())
                if label == '2':
                    status = 1
            elif com == 'next': # 왼쪽 눈 문제 끝나서 다시 눈 확인할 때, 화면에 다시 띄우기 위해서.
                status += 1
                com = ''
                print(status)
            elif com == 'change':
                face_start = True
                com = ''
            elif (com == 'takeEye'):  # 눈 인식 해야하는 명령인지
                com = ''
                check_command = 0  # 명령이랑 다시 또 찍지 않도록 초기화
                print('눈인식')
                #status = 1
                # something
                pose, left_open, right_open = takePose(ori_img, net2)
                print("Left open : ", left_open, "Right open : ", right_open)
                if right_open == True and left_open == True:
                    # 눈이 한개가 아닐 때
                    clientSock.sendall('noooo'.encode())
                else:
                    if right_open == True:
                        # 오른쪽 눈일 때
                        clientSock.sendall('right'.encode())
                    elif left_open == True:
                        # 왼쪽 눈일 때
                        clientSock.sendall('leftt'.encode())
                    else:
                        # 두눈이 감아져있을 때
                        clientSock.sendall('noooo'.encode())
            elif (com[1:4] == 'pic'):
                # 사진 보여주기
                check_command = 0
                sight = com[4:]
                r = com[0]
                com = ''
                print("sight: " + sight, "r: " + r)
                url = "https://storage.googleapis.com/eye-test-server/" + str(float(sight)) + "/" + r + ".png"
                img = url_to_image(url)
                w = len(img[0])
                h = len(img)
                sy = 240 - w//2
                sx = 320 - h//2
                whiteBackGround = prob_img.copy()
                whiteBackGround[sy:sy+w, sx:sx+w] = img
                cv2.putText(whiteBackGround, "{}".format(sight), (150, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 3)
                #cv2.putText(whiteBackGround, "위, 아래, 왼쪽, 오른쪽으로 정답을 말해주세요.", (15, 440), cv2.FONT_HERSHEY_SIMPLEX, 3.0, (0, 0, 0), 3)
                newimg = np.concatenate((whiteBackGround, pose), axis=1)
                cv2.imshow("eyeTest", newimg)  # 시력검사 이미지를 보여줌
                cv2.waitKey(500)
                print()
                if status == 1 or status == 3:
                    status += 1
            elif (com[:6] == 'result'):
                check_command = 0
                leftSight = ''
                for i in range(6, len(com)):
                    if (com[i] != '/'):
                        leftSight += com[i]
                    else:
                        tmp = i
                        break
                rightSight = com[tmp + 1:]
                com = ''
                frame = result_img.copy()
                print("result: " + leftSight + "  / " + rightSight)
                d = datetime.datetime.today()
                cv2.putText(frame, "{}".format(leftSight), (187, 370), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (0, 0, 0), 3)
                cv2.putText(frame, "{}".format(rightSight), (400, 370), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (0, 0, 0), 3)
                cv2.putText(frame, "{}".format(d.strftime('%Y-%m-%d')), (370, 180), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                            (0, 0, 0), 3)
                res_face_resize = cv2.resize(result_face, None, fx=0.25, fy=0.25)

                frame[80:80 + len(res_face_resize), 100:100 + len(res_face_resize[0])] = res_face_resize
                cv2.imshow("eyeTest", frame)  # 결과값 보여줌
                cv2.waitKey(0)
            elif (com == 'take'):  # 이건 그냥 테스트용 무시해도 됌
                com = ''
                check_command = 0
                print('take')
                clientSock.sendall('take'.encode())
        if status == -1:
            if face_start == True:
                newimg = np.concatenate((face_start_img, ori_img), axis=1)
            else:
                newimg = np.concatenate((main_img, ori_img), axis=1)
            cv2.imshow("eyeTest", newimg)  # 캠화면을 보여줌
            key = cv2.waitKey(50)  # 0.05s
        elif status == 0:
            if eye_dist > 0:
                cv2.putText(img, "%.1fm" % distance1,
                            (img.shape[1] - 350, img.shape[0] - 15), cv2.FONT_HERSHEY_SIMPLEX,
                            2.0, (0, 255, 0), 3)

            newimg = np.concatenate((dist_img, img), axis=1)
            cv2.imshow("eyeTest", newimg)  # 캠화면을 보여줌
            key = cv2.waitKey(50) # 0.05s
            if key == ord('q'):  # ctrl + z 을 누르면 캠 꺼짐
                break
        elif status == 1:
            newimg = np.concatenate((left_eye_img, ori_img), axis=1)
            cv2.imshow("eyeTest", newimg)  # 캠화면을 보여줌
            key = cv2.waitKey(50)  # 0.05s
        elif status == 3:
            newimg = np.concatenate((right_eye_img, ori_img), axis=1)
            cv2.imshow("eyeTest", newimg)  # 캠화면을 보여줌
            key = cv2.waitKey(50)  # 0.05s
    capture.release()
    cv2.destroyAllWindows()
    clientSock.close()  # 소켓을 닫습니다.
    exit(1)


# setting the camera, 640 x 480 resolution
capture = cv2.VideoCapture(cv2.CAP_DSHOW)
#capture = cv2.VideoCapture(0)
capture.set(3, 480)
capture.set(4, 640)

s_t = time.time()
DNN = "TF"
if DNN == "CAFFE":
    modelFile = "res10_300x300_ssd_iter_140000_fp16.caffemodel"
    configFile = "deploy.prototxt"
    net = cv2.dnn.readNetFromCaffe(configFile, modelFile)
else:
    modelFile = "opencv_face_detector_uint8.pb"
    configFile = "opencv_face_detector.pbtxt"
    net = cv2.dnn.readNetFromTensorflow(modelFile, configFile)
print("Load SSD model", time.time()-s_t)


MODEL_NAME = "COCO"
if MODEL_NAME == "COCO":
    #COCO Model
    protoFile = "pose_deploy_linevec.prototxt"
    weightsFile = "pose_iter_440000.caffemodel"
elif MODEL_NAME == "MPII":
    #MPII model
    protoFile = "pose_deploy_linevec_faster_4_stages.prototxt"
    weightsFile = "pose_iter_160000.caffemodel"
else:
    protoFile = "pose_deploy.prototxt"
    weightsFile = "pose_iter_584000.caffemodel"

# 위의 path에 있는 network 불러오기
s_t = time.time()
net2 = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)
print("Load openPose model", time.time()-s_t)


# load dlib face and shape detector
#detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor.dat')

# distance for initialization  ( 30 cm )
KNOWN_DISTANCE = 30.0

# TCP socket
PORT = 10080
clientSock = socket(AF_INET, SOCK_STREAM)
clientSock.connect(('34.97.244.124', PORT)) # relay server
print("서버에 접속하였습니다")

# check for server's command
check_command = False
com = ''
eye_rect_l=(0, 0, 0, 0)
eye_rect_r=(0, 0, 0, 0)
sys.setrecursionlimit(1000000)

t=threading.Thread(target=opencv)
t.start() # 캠화면은 계속 켜져있도록 쓰레드 돌림

while True:
    data = clientSock.recv(1024) #backend에서 데이터를 계속 기다립니다
    if not data:
        break
    # backend에서 온 명령 기억, check_command로 확인 여부 표시
    com = data.decode()
    check_command = True
    print(com)

