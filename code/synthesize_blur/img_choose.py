import numpy as np
import cv2
import os
from viz_flow import flow_to_image


def downsample(img):
    result = cv2.resize(img, None, fx=0.25, fy=0.25,
                        interpolation=cv2.INTER_CUBIC)
    height, width = result.shape[:2]
    h0 = (height-270) // 2
    w0 = (width-270) // 2
    result = result[h0:height-h0, w0:width-w0, :]
    return result


def big_enough(img):
    height, weight = img.shape[:2]
    if height >= 1080 and weight >= 1080:
        return True
    return False


def is_high_frequency(img, threshold=10):
    sobelx = cv2.Sobel(img, ddepth=cv2.CV_8U, dx=1, dy=0)
    sobely = cv2.Sobel(img, ddepth=cv2.CV_8U, dx=0, dy=1)
    sobel = np.abs(sobelx) + np.abs(sobely)
    avg = np.mean(sobel)
    if avg >= threshold:
        return True
    else:
        print('is not high frequency, avg gradient is {}'.format(avg))
        return False


def proper_motion(flow, percent=0.1, min_threshold=8, max_threshold=16):
    distance = np.max(np.abs(flow), axis=2)
    max_val = np.max(distance)
    if max_val > max_threshold:
        print('excess motion, max is {}'.format(max_val))
        #return False
    count = np.sum(distance >= min_threshold)
    shape = flow.shape[0] * flow.shape[1]
    prop = count*1.0 / shape
    if prop < percent:
        print('no sufficent motion, prop is {}'.format(prop))
        return False
    else:
        return True


def no_abrupt_change(img1, img2, flow):
    height, width = img1.shape
    y_coords, x_coords = np.mgrid[0:height, 0:width]
    coords = np.float32(np.dstack([x_coords, y_coords]))
    pixel_map = coords + flow
    new_frame = cv2.remap(img1, pixel_map, None, cv2.INTER_LINEAR)
    new_frame = np.where(new_frame==0, img2, new_frame)
    l1_distance = np.linalg.norm(new_frame-img2, ord=1)
    if l1_distance > 13*height*width:
        print("excess distance warp img and origin img, l1 is {}".format(l1_distance))
        return False
    return True


def is_linear_motion(flow1, flow2, threshold=0.8):
    flow1 = flow1.clip(-16, 16)
    flow2 = flow2.clip(-16, 16)
    diff = flow1 - flow2
    abs_diff = np.abs(diff)
    mean_diff = np.mean(abs_diff)
    print(mean_diff)
    if mean_diff > threshold:
        print('not linear motion, mean diff is {}'.format(mean_diff))
        return False
    return True


def triplet_check(data_path):
    data = []
    for item in data_path:
        print(item, end='\t')
        print('')
        img = cv2.imread(item)
        if not big_enough(img):
            print('not big enough')
            return False
        else:
            data.append(downsample(img))

    for item in data:
        if not is_high_frequency(item):
            return False

    gray_data = []
    for item in data:
        gray_data.append(cv2.cvtColor(item, cv2.COLOR_BGR2GRAY))

    flow1 = cv2.calcOpticalFlowFarneback(gray_data[0], gray_data[1], None, pyr_scale=0.5, levels=3, winsize=7, iterations=3, poly_n=7, poly_sigma=1.5, flags=cv2.OPTFLOW_FARNEBACK_GAUSSIAN)
    flow2 = cv2.calcOpticalFlowFarneback(gray_data[1], gray_data[2], None, pyr_scale=0.5, levels=3, winsize=7, iterations=3, poly_n=7, poly_sigma=1.5, flags=cv2.OPTFLOW_FARNEBACK_GAUSSIAN)
    flow_h_max, flow_w_max = gray_data[0].shape
    def unknown_flow_clip(flow, margin=0):
        flow_h = flow[..., 0]
        flow_w = flow[..., 1]
        flow[..., 0] = np.where(flow_h>flow_h_max-margin, 0, flow_h)
        flow[..., 1] = np.where(flow_w>flow_w_max-margin, 0, flow_w)

    unknown_flow_clip(flow1)
    unknown_flow_clip(flow2)
    #flow_img = flow_to_image(flow1)
    #cv2.imwrite(data_path[0][:-4]+'_flow1.jpg', flow_img)
    if not no_abrupt_change(gray_data[0], gray_data[1], flow1) or not no_abrupt_change(gray_data[1], gray_data[2], flow2):
        return False
    if not proper_motion(flow1) or not proper_motion(flow2):
        return False

    if not is_linear_motion(flow1, flow2):
        return False

    return True


def main(input_path, output_path):
    img_list = []
    for item in os.listdir(input_path):
        if item.split('.')[-1] in ['jpg']:
            img_list.append(os.path.join(input_path, item))
    img_list.sort(key=lambda x: x)
    data_list = []
    for i in range(len(img_list)-2):
        data_list.append([img_list[i], img_list[i+1], img_list[i+2]])

    result_list = []
    count = 0
    for item in data_list:
        if triplet_check(item):
            result = '\t'.join(item) + '\n'
            result_list.append(result)
        else:
            print(count)
        if count > 20:
            break
        count += 1
    with open(output_path, 'w') as f:
        f.writelines(result_list)


if __name__ == '__main__':
    input_path = './'
    output_path = 'triplets.txt'
    main(input_path, output_path)
