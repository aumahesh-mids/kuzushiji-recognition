import argparse

import numpy as np
import chainer
from PIL.Image import Image

from hanya.kr.classifier.softmax.mobilenetv3 import MobileNetV3
from hanya.kr.datasets.kuzushiji_recognition import KuzushijiUnicodeMapping
from hanya.kr.detector.centernet.resnet import Res18UnetCenterNet


class HanyaOCR:
    def __init__(self, detector_model, classifier_model):
        self.detector = detector_model
        self.classifier = classifier_model

    def ocr(self, filename):
        mapping = KuzushijiUnicodeMapping()

        # load trained detector
        detector = Res18UnetCenterNet()
        chainer.serializers.load_npz(self.detector, detector)

        # load trained classifier
        classifier = MobileNetV3(out_ch=len(mapping))
        chainer.serializers.load_npz(self.classifier, classifier)

        # load image
        image = Image.open(filename)

        # character detection
        bboxes, bbox_scores = detector.detect(image)

        # character classification
        unicode_indices, unicode_scores = classifier.classify(image, bboxes)
        unicodes = [mapping.index_to_unicode(idx) for idx in unicode_indices]

        result = {
            'unicodes': unicodes,
            'unicode_scores': unicode_scores,
            'bboxes': bboxes,
            'bbox_scores': bbox_scores
        }

        return result

def init_argparse() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        usage="%(prog)s [OPTION] [FILE]...",
        description="Restor-AI-tion transformer."
    )
    parser.add_argument('--detector_model', type=str, help='detector model file', default='/home/ec2-user/code/t-hanya/kuzushiji-recognition/results/detector/model_700.npz')
    parser.add_argument('--classifier_model', type=str, help='classifier model file', default='/home/ec2-user/code/t-hanya/kuzushiji-recognition/results/classifier/model_1000.npz')
    parser.add_argument('--file', type=str, help='file to recognize')
    return parser

def do_ocr(detector_model, classifier_model, filename):
    hanya_ocr = HanyaOCR(detector_model, classifier_model)
    return hanya_ocr.ocr(filename)


if __name__ == '__main__':
    np.random.seed(1234)
    parser = init_argparse()
    args = parser.parse_args()
    do_ocr(args.detector_model, args.classifier_model, args.file)
