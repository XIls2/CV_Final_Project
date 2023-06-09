import pickle
import argparse
import json

import paddle
import numpy as np

from datasets import Compose, UniformSampleFrames, Resize, PoseCompact, CenterCrop, PoseDecode, GeneratePoseTarget, FormatShape, Collect
from datasets import PoseDataset
from models.resnet3d_slowonly import ResNet3dSlowOnly
from models.i3d_head import I3DHead
from models.recognizer3d import Recognizer3D
from utils import load_pretrained_model
from progress_bar import ProgressBar


def parse_args():
    parser = argparse.ArgumentParser(description='Model training')

    parser.add_argument(
        '--input_file',
        dest='input_file',
        help='The path of input file',
        type=str,
        default='test_tipc/data/predict_example.pkl')

    parser.add_argument(
        '--pretrained',
        dest='pretrained',
        help='The pretrained of model',
        type=str,
        default='output/best_model/model.pdparams')

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    left_kp = [1, 3, 5, 7, 9, 11, 13, 15]
    right_kp = [2, 4, 6, 8, 10, 12, 14, 16]
    tranforms = [
        UniformSampleFrames(clip_len=48, num_clips=10, test_mode=True),
        PoseDecode(),
        PoseCompact(hw_ratio=1., allow_imgpad=True),
        Resize(scale=(-1, 56)),
        CenterCrop(crop_size=56),
        GeneratePoseTarget(sigma=0.6,
                           use_score=True,
                           with_kp=True,
                           with_limb=False,
                           double=True,
                           left_kp=left_kp,
                           right_kp=right_kp),
        FormatShape(input_format='NCTHW'),
        Collect(keys=['imgs', 'label'], meta_keys=[])
    ]

    tranforms = Compose(tranforms)
    with open(args.input_file, 'rb') as f:
        data = pickle.load(f)
    files = [tranforms(f) for f in data]

    imgs = []
    for f in files:
        imgs.append(f["imgs"][np.newaxis, :, :, :, :, :])
    imgs = np.concatenate(imgs)


    backbone = ResNet3dSlowOnly(
        depth=50,
        pretrained=None,
        in_channels=17,
        base_channels=32,
        num_stages=3,
        out_indices=(2,),
        stage_blocks=(3, 4, 6),
        conv1_stride_s=1,
        pool1_stride_s=1,
        inflate=(0, 1, 1),
        spatial_strides=(2, 2, 2),
        temporal_strides=(1, 1, 2),
        dilations=(1, 1, 1)
    )
    head = I3DHead(num_classes=101, in_channels=512, spatial_type='avg', dropout_ratio=0.5)
    model = Recognizer3D(backbone=backbone, cls_head=head)
    load_pretrained_model(model, args.pretrained)
    with open('labels.json', 'r') as f:
        label_name = json.load(f)
    model.eval()
    imgs = paddle.to_tensor(imgs)
    for i in range(imgs.shape[0]):
        with paddle.no_grad():
            img = imgs[i:i+1]
            result = model(img)
            out = np.squeeze(result)
            out = np.argmax(out)
            print(f"File {data[i]['frame_dir']} is class {label_name[str(out)]}")
            pass

