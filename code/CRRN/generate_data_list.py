import os
import numpy as np

def generate(img_path, output_path, generate_per_img=5):
    img_list = [item for item in os.listdir(img_path)]
    length = len(img_list)
    items = []
    for i in range(length):
        for j in range(generate_per_img):
            content = img_path + img_list[i] + ','
            random_val = np.random.randint(0, length-1)
            content += img_path + img_list[random_val] + '\n'
            items.append(content)
    print(len(items))
    with open(output_path, 'w') as f:
        f.writelines(items)

            
def main():
    coco_val_path = '/mnt/lustre/share/DSK/datasets/mscoco2017/val2017/'
    coco_test_path = '/mnt/lustre/share/DSK/datasets/mscoco2017/test2017/'
    output_train = './train.txt'
    output_test = './test.txt'
    generate(coco_test_path, output_train)
    generate(coco_val_path, output_test)


if __name__ == "__main__":
    main()
