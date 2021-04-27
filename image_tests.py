import argparse
import os

import cv2
import torch
from torch import nn
from torchvision import transforms

from  PIL import Image

from label_mapping import DATASET_ID_MAPPING


class AnimalTester(object):

    def __init__(self, checkpoint_path: str):
        self._checkpoint_path = checkpoint_path
        self._vgg16 = torch.load(checkpoint_path, map_location=torch.device('cpu'))
        # self._vgg16.cuda()
        self._vgg16.eval()

        self._transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])


    def test_image_dir(self, image_dir_path):
        for image_name in os.listdir(image_dir_path):
            image_path = os.path.join(image_dir_path, image_name)
            self.test_image(image_path)

    def test_image(self, image_path: str):
        img = cv2.imread(image_path)
        pil_img = Image.fromarray(img)

        transformed = self._transform(pil_img)
        batch = transformed.unsqueeze(0)

        with torch.no_grad():
            outputs = self._vgg16(batch)
            _, predicted = torch.max(outputs.data, 1)

            class_id = predicted.numpy()[0]
            m = nn.Softmax(dim=1)
            percent = m(outputs).numpy().squeeze()[class_id]
            print("{}: {:.2f} %".format(DATASET_ID_MAPPING[class_id], percent*100.))

        cv2.imshow("view", img)
        cv2.waitKey()

def parse_args():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--image_path', dest="image_path", type=str, required=True, help='Path to image.')
    parser.add_argument('--checkpoint_path', dest="checkpoint_path", type=str, required=True, help='Path to checkpoint.')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()

    animal_tester = AnimalTester(args.checkpoint_path)
    animal_tester.test_image(args.image_path)

