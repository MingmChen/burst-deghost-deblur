import random
import numpy as np
import cv2


def generate_kernel(step=128, kernel_size=51, is_viz=False):
    def vector2d(v):
        a, b = v
        r = np.sqrt(a**2 + b**2)
        theta = np.arctan(b/a)
        return np.array((r*np.cos(theta), r*np.sin(theta)))

    acc = np.random.normal(0, 1, size=(step, 2))  # 2d accelerate
    velocity0 = np.random.uniform(-1, 1, size=2)
    shift0 = np.array((0., 0.))
    shifts = []
    shifts.append(shift0.copy())

    shift = shift0
    velocity = velocity0
    for i in range(step):
        shift += velocity + 0.5*acc[i]
        velocity += acc[i]
        shifts.append(shift.copy())
    shifts_np = np.array(shifts)
    shifts_abs_max = np.abs(shifts_np).max()
    shifts_np /= shifts_abs_max
    select_num = (int)((step+1) * random.uniform(0.1, 0.8))
    select_idx = np.random.choice(np.array([x for x in range(step+1)]), select_num, replace=False)
    shifts_np = shifts_np[select_idx]
    random_scale = (kernel_size // 2) * random.uniform(0.5, 1)
    shifts_np *= random_scale

    kernel = np.zeros((kernel_size, kernel_size))
    for i in range(shifts_np.shape[0]):
        x, y = shifts_np[i] + (kernel_size // 2)
        a = int(x)
        b = a+1
        c = int(y)
        d = c+1
        kernel[a, c] += (x - a) * (y - c)
        kernel[a, d] += (x - a) * (d - y)
        kernel[b, c] += (b - x) * (y - c)
        kernel[b, d] += (b - x) * (d - y)
    kernel = kernel / kernel.sum()

    if is_viz:
        k_viz = (kernel - kernel.min()) / (kernel.max() - kernel.min()) * 255
        cv2.imwrite('kernel.png', k_viz)
    return kernel


def apply_blur(img, kernel, gauss_size=5, sigma_range=(0.8, 1.2)):
    sigma = 0.3*((gauss_size - 1)*0.5 - 1) + 0.8
    sigma *= random.uniform(*sigma_range)
    blur_img = cv2.filter2D(img, -1, kernel)
    blur_img = cv2.GaussianBlur(blur_img, (gauss_size, gauss_size), sigma)
    return blur_img


def main(generate_num=10):
    input_path = 'resize12k.jpg'
    img = cv2.imread(input_path)
    kernel = generate_kernel()
    blur_img = apply_blur(img, kernel)
    cv2.imwrite('blur.jpg', blur_img)


if __name__ == '__main__':
    main()
