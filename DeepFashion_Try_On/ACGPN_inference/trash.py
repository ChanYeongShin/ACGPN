from PIL import Image
import cv2
import numpy as np
import argparse
import os

def get_arguments():
    """Parse all the arguments provided from the CLI.
    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Self Correction for Human Parsing")

    parser.add_argument("--input-dir", type=str, default='./sample', help="path of input image folder.")
    parser.add_argument("--output-dir", type=str, default='/home/ubuntu/Desktop/DeepFashion_Try_On-master/ACGPN_inference/sample2', help="path of output image folder.")

    return parser.parse_args()


def main():
    args = get_arguments()
    trans_dict = {
            0:0,
            1:1, 2:0,
            3:0, 4:4, 5:0, 6:0,
            7:0,
            8:8,
            9:9, 10:10,
            11:11,
            12:12,
            13:13,
            14:0, 15:0,
            16:0, 17:0
        }
    directory = '/home/ubuntu/Desktop/DeepFashion_Try_On-master/ACGPN_inference/sample'
    for filename in os.listdir(directory):
        img = Image.open(os.path.join(directory, filename))
        parsing_result_path = os.path.join(args.output_dir, filename[:-4] + '.png')
        output_arr = np.asarray(img, dtype=np.uint8)

        new_arr = np.full(output_arr.shape, 7)
        for old, new in trans_dict.items():
            new_arr = np.where(output_arr == old, new, new_arr)
        output_img = Image.fromarray(new_arr.astype(np.uint8))
        output_img.save(parsing_result_path)

main()
img = cv2.imread('/home/ubuntu/Desktop/DeepFashion_Try_On-master/weak_data/test_label/000021_0.png')
npy = np.asarray(img)
print(np.unique(npy))
print(npy.dtype)
img = cv2.imread('/home/ubuntu/Desktop/DeepFashion_Try_On-master/ACGPN_inference/sample/000021_0.jpg')
npy = np.asarray(img)
print(np.unique(npy))
print(npy.dtype)
img = cv2.imread('/home/ubuntu/Desktop/DeepFashion_Try_On-master/ACGPN_inference/sample2/000021_0.png')
npy = np.asarray(img)
print(np.unique(npy))
print(npy.dtype)