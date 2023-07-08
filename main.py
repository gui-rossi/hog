import pickle

import numpy as np
import cv2 as cv

import json
from matplotlib import pyplot as plt
from skimage import color
from skimage.feature import hog
from sklearn import svm
from sklearn.metrics import classification_report,accuracy_score
from subprocess import check_output

import dlib
import glob
import os
import sys
import time
import matplotlib.pyplot as plt
import pyautogui as pyg
import shutil
import xml.dom.minidom as minidom

def detectionHogForImage(folder_path):
    target_folder = 'C:\\Users\\Guilherme\\Dev\\HOG\\HogDetected\\'

    hog = cv.HOGDescriptor()
    hog.setSVMDetector(cv.HOGDescriptor_getDefaultPeopleDetector())

    detected = 0
    total_frames = 0

    for file in os.listdir(folder_path):
        if ('.xml' not in file):
            total_frames = total_frames + 1

            frame = cv.imread(folder_path + file)

            frame = cv.resize(frame, (640, 480))
            frame = cv.cvtColor(frame, cv.COLOR_RGB2GRAY)

            boxes, weights = hog.detectMultiScale(frame, winStride=(8, 8))

            if (len(boxes) != 0):
                detected = detected + 1

            boxes = np.array([[x, y, x + w, y + h] for (x, y, w, h) in boxes])

            for (xA, yA, xB, yB) in boxes:
                cv.rectangle(frame, (xA, yA), (xB, yB),
                             (0, 255, 0), 2)

            cv.imwrite(target_folder + file, frame)

            if cv.waitKey(1) & 0xFF == ord('q'):
                # breaking the loop if the user types q
                # note that the video window must be highlighted!
                break

        cv.waitKey(1)

    print('Detected: ',  detected, ' Total frames: ', total_frames)

def detectionHogForImageCustom(folder_path):
    target_folder = 'C:\\Users\\Guilherme\\Dev\\HOG\\HogDetected\\'

    # Load our trained detector
    detector = dlib.simple_object_detector("truck_detection.svm")

    start_time = time.time()

    # Setting the downscaling size, for faster detection
    # If you're not getting any detections then you can set this to 1
    scale_factor = 2.0

    # Initially the size of the hand and its center x point will be 0
    size, center_x = 0, 0

    # Initialize these variables for calculating FPS
    fps = 0
    frame_counter = 0

    for file in os.listdir(folder_path):
        if ('.xml' not in file):
            # Set the window name
            #cv.namedWindow(file, cv.WINDOW_NORMAL)

            frame = cv.imread(folder_path + file)

            # Calculate the Average FPS
            frame_counter += 1
            fps = (frame_counter / (time.time() - start_time))

            # Create a clean copy of the frame
            copy = frame.copy()

            # Downsize the frame.
            new_width = int(frame.shape[1] / scale_factor)
            new_height = int(frame.shape[0] / scale_factor)
            resized_frame = cv.resize(copy, (new_width, new_height))

            # Detect with detector
            detections = detector(resized_frame)

            # Loop for each detection.
            for detection in (detections):
                # Since we downscaled the image we will need to resacle the coordinates according to the original image.
                x1 = int(detection.left() * scale_factor)
                y1 = int(detection.top() * scale_factor)
                x2 = int(detection.right() * scale_factor)
                y2 = int(detection.bottom() * scale_factor)

                # Draw the bounding box
                cv.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv.putText(frame, 'Truck Detected', (x1, y2 + 20), cv.FONT_HERSHEY_COMPLEX, 0.6, (0, 0, 255), 2)

                # Calculate size of the truck.
                size = int((x2 - x1) * (y2 - y1))

                # Extract the center of the hand on x-axis.
                center_x = x2 - x1 // 2

            # Display FPS and size of hand
            cv.putText(frame, 'FPS: {:.2f}'.format(fps), (20, 20), cv.FONT_HERSHEY_COMPLEX, 0.6, (0, 0, 255), 2)

            # This information is useful for when you'll be building hand gesture applications
            cv.putText(frame, 'Center: {}'.format(center_x), (540, 20), cv.FONT_HERSHEY_COMPLEX, 0.5, (233, 100, 25))
            cv.putText(frame, 'size: {}'.format(size), (540, 40), cv.FONT_HERSHEY_COMPLEX, 0.5, (233, 100, 25))

            # Display the image
            #cv.imshow('frame', frame)
            cv.imwrite(target_folder + file, frame)

            if cv.waitKey(1) & 0xFF == ord('q'):
                break

    cv.destroyAllWindows()

def detectTruck():
    # Load our trained detector
    detector = dlib.simple_object_detector("truck_detection.svm")

    # Set the window name
    cv.namedWindow('frame', cv.WINDOW_NORMAL)

    # Initialize webcam
    cap = cv.VideoCapture('truck_dia.mp4')

    # Setting the downscaling size, for faster detection
    # If you're not getting any detections then you can set this to 1
    scale_factor = 2.0

    # Initially the size of the hand and its center x point will be 0
    size, center_x = 0, 0

    # Initialize these variables for calculating FPS
    fps = 0
    frame_counter = 0
    start_time = time.time()
    total_frames = 0
    detected_frames = 0

    # Set the while loop
    while (True):

        # Read frame by frame
        ret, frame = cap.read()

        if not ret:
            break

        # Laterally flip the frame
        frame = cv.flip(frame, 1)

        # Calculate the Average FPS
        frame_counter += 1
        fps = (frame_counter / (time.time() - start_time))

        # Create a clean copy of the frame
        copy = frame.copy()

        # Downsize the frame.
        new_width = int(frame.shape[1] / scale_factor)
        new_height = int(frame.shape[0] / scale_factor)
        resized_frame = cv.resize(copy, (new_width, new_height))

        # Detect with detector
        detections = detector(resized_frame)
        if (len(detections) > 0):
            detected_frames += 1

        total_frames += 1

        # Loop for each detection.
        for detection in (detections):
            # Since we downscaled the image we will need to rescale the coordinates according to the original image.
            x1 = int(detection.left() * scale_factor)
            y1 = int(detection.top() * scale_factor)
            x2 = int(detection.right() * scale_factor)
            y2 = int(detection.bottom() * scale_factor)

            # Draw the bounding box
            cv.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv.putText(frame, 'Truck Detected', (x1, y2 + 20), cv.FONT_HERSHEY_COMPLEX, 0.6, (0, 0, 255), 2)

            # Calculate size of the hand.
            size = int((x2 - x1) * (y2 - y1))

            # Extract the center of the hand on x-axis.
            center_x = x2 - x1 // 2

        # Display FPS and size of hand
        cv.putText(frame, 'FPS: {:.2f}'.format(fps), (20, 20), cv.FONT_HERSHEY_COMPLEX, 0.6, (0, 0, 255), 2)

        # This information is useful for when you'll be building hand gesture applications
        cv.putText(frame, 'Center: {}'.format(center_x), (540, 20), cv.FONT_HERSHEY_COMPLEX, 0.5, (233, 100, 25))
        cv.putText(frame, 'size: {}'.format(size), (540, 40), cv.FONT_HERSHEY_COMPLEX, 0.5, (233, 100, 25))

        # Display the image
        cv.imshow('frame', frame)

        print('total_frames: ', total_frames)
        print('detected_frames: ', detected_frames)

        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    # Relase the webcam and destroy all windows
    cap.release()
    cv.destroyAllWindows()

def detectPeople():
    hog = cv.HOGDescriptor()
    hog.setSVMDetector(cv.HOGDescriptor_getDefaultPeopleDetector())

    total_frames = 0
    detected_frames = 0

    cap = cv.VideoCapture('pessoas_noturno.mp4')

    # the output will be written to output.avi
    out = cv.VideoWriter(
        'output.avi',
        cv.VideoWriter_fourcc(*'MJPG'),
        15.,
        (640, 480))

    while (True):
        ret, frame = cap.read()

        frame = cv.resize(frame, (640, 480))
        gray = cv.cvtColor(frame, cv.COLOR_RGB2GRAY)

        boxes, weights = hog.detectMultiScale(gray, winStride=(8, 8))

        boxes = np.array([[x, y, x + w, y + h] for (x, y, w, h) in boxes])

        if (len(boxes) > 0):
            detected_frames += 1

        total_frames += 1

        for (xA, yA, xB, yB) in boxes:
            cv.rectangle(frame, (xA, yA), (xB, yB),
                          (0, 255, 0), 2)

        print('total_frames: ', total_frames)
        print('detected_frames: ', detected_frames)

        out.write(frame.astype('uint8'))
        cv.imshow('frame', frame)
        if cv.waitKey(1) & 0xFF == ord('q'):
            # breaking the loop if the user types q
            # note that the video window must be highlighted!
            break

    # When everything done, release the capture
    cap.release()
    # and release the output
    out.release()
    # finally, close the window
    cv.destroyAllWindows()
    cv.waitKey(1)

SZ=20
bin_n = 16 # Number of bins
affine_flags = cv.WARP_INVERSE_MAP|cv.INTER_LINEAR

def deskew(img):
    m = cv.moments(img)
    if abs(m['mu02']) < 1e-2:
        return img.copy()
    skew = m['mu11']/m['mu02']
    M = np.float32([[1, skew, -0.5*SZ*skew], [0, 1, 0]])
    img = cv.warpAffine(img,M,(SZ, SZ),flags=affine_flags)
    return img

def hog(img):
    gx = cv.Sobel(img, cv.CV_32F, 1, 0)
    gy = cv.Sobel(img, cv.CV_32F, 0, 1)
    mag, ang = cv.cartToPolar(gx, gy)
    bins = np.int32(bin_n*ang/(2*np.pi))    # quantizing binvalues in (0...16)
    bin_cells = bins[:10,:10], bins[10:,:10], bins[:10,10:], bins[10:,10:]
    mag_cells = mag[:10,:10], mag[10:,:10], mag[:10,10:], mag[10:,10:]
    hists = [np.bincount(b.ravel(), m.ravel(), bin_n) for b, m in zip(bin_cells, mag_cells)]
    hist = np.hstack(hists)     # hist is a 64 bit vector
    return hist

def trainClassifier():
    img = cv.imread('digits.png', 0)
    if img is None:
        raise Exception("we need the digits.png image from samples/data here !")

    cells = [np.hsplit(row, 100) for row in np.vsplit(img, 50)]

    # First half is trainData, remaining is testData
    train_cells = [i[:50] for i in cells]
    test_cells = [i[50:] for i in cells]

    deskewed = [list(map(deskew, row)) for row in train_cells]
    hogdata = [list(map(hog, row)) for row in deskewed]
    trainData = np.float32(hogdata).reshape(-1, 64)
    responses = np.repeat(np.arange(10), 250)[:, np.newaxis]

    svm = cv.ml.SVM_create()
    svm.setKernel(cv.ml.SVM_LINEAR)
    svm.setType(cv.ml.SVM_C_SVC)
    svm.setC(2.67)
    svm.setGamma(5.383)

    svm.train(trainData, cv.ml.ROW_SAMPLE, responses)
    svm.save('svm_data.xml')

    deskewed = [list(map(deskew, row)) for row in test_cells]
    hogdata = [list(map(hog, row)) for row in deskewed]
    testData = np.float32(hogdata).reshape(-1, bin_n * 4)
    result = svm.predict(testData)[1]

    mask = result == responses
    correct = np.count_nonzero(mask)
    print(correct * 100.0 / result.size)

def resize_image(image, width, height):
    resized_image = cv.resize(image, (width, height))
    return resized_image

def resizeImages():
    folder_dir_train = "C:\\Users\\Guilherme\\Dev\\HOG\\truckValidation\\train\\"
    folder_dir_val = "C:\\Users\\Guilherme\\Dev\\HOG\\truckValidation\\validate\\"

    for file in os.listdir(folder_dir_train):
        img = cv.imread(folder_dir_train + file)
        resized_image = resize_image(img, 300, 211)
        cv.imwrite(folder_dir_train + file, resized_image)

    for file in os.listdir(folder_dir_val):
        img = cv.imread(folder_dir_val + file)
        resized_image = resize_image(img, 300, 211)
        cv.imwrite(folder_dir_val + file, resized_image)

def createHOGCoordFiles():
    folder_dir_train = "C:\\Users\\Guilherme\\Dev\\HOG\\newTruckValidation\\train\\"
    folder_dir_val = "C:\\Users\\Guilherme\\Dev\\HOG\\newTruckValidation\\validate\\"

    train_coord = open(r"train_coord.txt", "w+")
    validate_coord = open(r"validate_coord.txt", "w+")

    for file in os.listdir(folder_dir_train):
        if (file.endswith(".jpg")):
            xml = minidom.parse('newTruckValidation\\train\\' + file.split('.')[0] + '.xml')
            bndboxes = xml.getElementsByTagName('bndbox')

            for box in bndboxes:
                train_coord.write(file + ":")
                train_coord.write("(" +
                str(int(box.childNodes[1].firstChild.data)) + "," +
                str(int(box.childNodes[3].firstChild.data)) + "," +
                str(int(box.childNodes[5].firstChild.data)) + "," +
                str(int(box.childNodes[7].firstChild.data)) + ")")
                train_coord.write(",\n")

    train_coord.close()

    for file in os.listdir(folder_dir_val):
        if (file.endswith(".jpg")):
            xml = minidom.parse('newTruckValidation\\validate\\' + file.split('.')[0] + '.xml')
            bndboxes = xml.getElementsByTagName('bndbox')

            for box in bndboxes:
                validate_coord.write(file + ":")
                validate_coord.write("(" +
                str(int(box.childNodes[1].firstChild.data)) + "," +
                str(int(box.childNodes[3].firstChild.data)) + "," +
                str(int(box.childNodes[5].firstChild.data)) + "," +
                str(int(box.childNodes[7].firstChild.data)) + ")")
                validate_coord.write(",\n")

    validate_coord.close()

def drawRect(num_samples, data):
    folder_dir_val = "C:\\Users\\Guilherme\\Dev\\HOG\\personValidation\\train\\"

    image_names = []
    samples = 0
    for file in os.listdir(folder_dir_val):
        if (samples >= num_samples):
            break

        if (file.endswith(".jpg")):
            image_names.append(file)
            samples = samples + 1

    cols = 1
    #rows = int(np.ceil(num_samples / cols))
    rows = 1
    #plt.figure(figsize=(cols * cols, rows * cols))

    for i in range(num_samples):
        # Extract the bonding box coordinates
        d_box = data[i][1][0]
        left, top, right, bottom = d_box.left(), d_box.top(), d_box.right(), d_box.bottom()

        # Get the image
        image = data[i][0]

        # Draw reectangle on the detected hand
        cv.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 3)

        # Display the image
        plt.subplot(rows, cols, i + 1);
        plt.imshow(image[:, :, ::-1]);
        plt.axis('off');

def expand_aspect_ratio(width, height, target_ratio):
    # Calculate the current aspect ratio
    current_ratio = width / height

    # If the current aspect ratio is already greater than or equal to the target ratio,
    # return the original width and height as they are
    if current_ratio >= target_ratio:
        return width, height

    # Otherwise, calculate the new width and height that will produce the target aspect ratio
    new_width = height * target_ratio
    new_height = height

    # Round the new dimensions to the nearest integer to avoid floating-point errors
    new_width = round(new_width)
    new_height = round(new_height)

    # Return the new dimensions as a tuple
    return new_width, new_height

def trainHOG():
    ########### PREPARE DATASET

    data = {}

    lines = ''
    lines_content = ''

    fp = open("train_coord.txt", "r")
    lines_content = fp.readlines()
    lines = len(lines_content)

    line = 0
    lowest_ar = 9999
    highest_ar = 0
    while line < lines:

        img = cv.imread(os.path.join("newTruckValidation\\train\\" + lines_content[line].split(":")[0]))

        coords = lines_content[line].split(":")[1].replace("(", "").replace(")", "")
        coords = coords[:len(coords) - 2]

        x1, y1, x2, y2 = coords.split(",")

        ar = (int(x2) - int(x1)) / (int(y2) - int(y1))
        new_width, new_height = expand_aspect_ratio(int(x2) - int(x1), int(y2) - int(y1), 13/5) #2.6 aspect ratio

        if (new_width/new_height <= lowest_ar):
            lowest_ar = new_width/new_height
        if (new_width/new_height >= highest_ar):
            highest_ar = new_width/new_height

        dlib_box = [dlib.rectangle(left=int(x1), top=int(y1), right=int(x2), bottom=int(y2))]
        data[line] = (img, dlib_box)

        line = line + 1

    ########### VIEW IMAGES

    drawRect(1, data)

    ########### TRAIN

    percent = 0.8

    split = int(len(data) * percent)

    images = [tuple_value[0] for tuple_value in data.values()]
    bounding_boxes = [tuple_value[1] for tuple_value in data.values()]

    options = dlib.simple_object_detector_training_options()

    options.add_left_right_image_flips = False

    options.C = 5

    st = time.time()

    detector = dlib.train_simple_object_detector(images, bounding_boxes, options)

    print('Training Completed, Total Time taken: {:.2f} seconds'.format(time.time() - st))

    file_name = 'truck_detection.svm'
    detector.save(file_name)

    win_det = dlib.image_window()
    win_det.set_image(detector)

    print("Training Metrics: {}".format(dlib.test_simple_object_detector(images[:split], bounding_boxes[:split], detector)))

def replicateXML():
    folder_dir_train = "C:\\Users\\Guilherme\\Dev\\HOG\\truckValidation\\train\\"
    folder_dir_validate = "C:\\Users\\Guilherme\\Dev\\HOG\\truckValidation\\validate\\"

    xmlPath = 'truckValidation\\train\\Image_013191.xml'

    for file in os.listdir(folder_dir_train):
        if (file.endswith(".jpg") and file != 'Image_013191.jpg'):
            shutil.copyfile(xmlPath, 'truckValidation\\train\\' + file.split('.')[0] + '.xml')

            #xml = minidom.parse('truckValidation\\train\\' + file.split('.')[0] + '.xml')

            #editableXML = open(r"" + file.split('.')[0] + ".xml", "w+")
            #filename = xml.getElementsByTagName('filename')

            #filename[0].firstChild.data = file

    for file in os.listdir(folder_dir_validate):
        if (file.endswith(".jpg") and file != 'Image_013191.jpg'):
            shutil.copyfile(xmlPath, 'truckValidation\\validate\\' + file.split('.')[0] + '.xml')


def main():
    detectPeople()
    #detectTruck()

    #detectionHogForImage('C:\\Users\\Guilherme\\Dev\\HOG\\newTruckValidation\\train\\')
    #detectionHogForImage('C:\\Users\\Guilherme\\Dev\\HOG\\newTruckValidation\\validate\\')
    #detectionHogForImage('C:\\Users\\Guilherme\\Dev\\HOG\\personValidationNoturna\\train\\')
    #detectionHogForImage('C:\\Users\\Guilherme\\Dev\\HOG\\personValidationNoturna\\validate\\')

    #testTruck()
    #trainClassifier()
    #createHOGCoordFiles()
    #trainHOG()
    #resizeImages()
    #replicateXML()

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
