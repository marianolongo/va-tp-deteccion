import cv2

from src.trackbar import create_trackbar, get_trackbar_value
from src.machine.utils.testing_model import predict_contour
from src.machine.utils.hu_moments_generation import generate_hu_moments_file
from src.machine.utils.training_model import train_model


def compare_contours_with_biggest(contours, biggest_contour, frame, compare_value):
    for contour in contours:
        print(cv2.matchShapes(contour, biggest_contour, cv2.CONTOURS_MATCH_I2, 0))
        if cv2.matchShapes(contour, biggest_contour, cv2.CONTOURS_MATCH_I2, 0) < compare_value:
            cv2.drawContours(frame, contour, -1, (0, 255, 0), 2)


def get_biggest_contour(min_area, contours):
    biggest_contour = None
    biggest_area = min_area
    print(len(contours))
    for i in range(len(contours)):
        if cv2.contourArea(contours[i]) > biggest_area:
            biggest_contour = contours[i]
    return biggest_contour


def get_center(contour):
    M = cv2.moments(contour)
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])
    return cX, cY


def draw_contours(denoised, original, min_contour_area, max_contour_area, model):
    result = []
    contours, hierarchy = cv2.findContours(denoised, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    for i in range(len(contours)):
        area = cv2.contourArea(contours[i])

        if min_contour_area <= area <= max_contour_area:
            result.append(contours[i])
            label = predict_contour(model, contours[i])
            cv2.drawContours(original, contours[i], -1, (0, 0, 255), 2)
            cv2.putText(original, label, get_center(contours[i]), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (255, 0, 0), 1, cv2.LINE_AA)
    if len(result) > 0:
        result.sort(key=cv2.contourArea)
    return result


def denoise(frame, noise_value):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (noise_value, noise_value))
    opening = cv2.morphologyEx(frame, cv2.MORPH_OPEN, kernel)
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
    return closing


def detection():
    window_name = 'Window'
    cv2.namedWindow(window_name)
    cap = cv2.VideoCapture(0)

    trackbar_thresh_value_name = "Threshold Value"
    trackbar_noise_val_name = "Noise Value"
    trackbar_contour_area_min_name = "Contour Area Min"
    trackbar_contour_area_max_name = "Contour Area Max"
    trackbar_compare_value_name = "Compare Value"
    threshold_max = 255
    noise_max = 20
    contour_area_max = 80000
    create_trackbar(trackbar_thresh_value_name, window_name, threshold_max, 100)
    create_trackbar(trackbar_noise_val_name, window_name, noise_max)
    create_trackbar(trackbar_contour_area_min_name, window_name, contour_area_max, 4000)
    create_trackbar(trackbar_contour_area_max_name, window_name, contour_area_max, 50000)
    create_trackbar(trackbar_compare_value_name, window_name, 200)

    generate_hu_moments_file()
    model = train_model()

    biggest_contour = None

    while True:
        threshold_value = get_trackbar_value(trackbar_name=trackbar_thresh_value_name, window_name=window_name)
        noise_value = get_trackbar_value(trackbar_name=trackbar_noise_val_name, window_name=window_name)
        contour_area_min = get_trackbar_value(trackbar_name=trackbar_contour_area_min_name, window_name=window_name)
        contour_area_max = get_trackbar_value(trackbar_name=trackbar_contour_area_max_name, window_name=window_name)
        compare_value = get_trackbar_value(trackbar_name=trackbar_compare_value_name, window_name=window_name)

        ret, original = cap.read()

        binary = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)

        ret1, thresh1 = cv2.threshold(binary, threshold_value, threshold_max, cv2.THRESH_BINARY)

        denoised = denoise(thresh1, noise_value)

        contours = draw_contours(denoised, original, contour_area_min, contour_area_max, model)
        if biggest_contour is not None:
            compare_contours_with_biggest(contours, biggest_contour, original, compare_value/100)

        cv2.imshow("Original", original)
        cv2.imshow("Denoised", denoised)
        if cv2.waitKey(1) & 0xFF == ord('t'):
            biggest_contour = get_biggest_contour(contour_area_min, contours)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()


detection()
