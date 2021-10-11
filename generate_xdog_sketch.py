from PIL import Image
from scipy.ndimage import filters
import numpy as np
from glob import glob
import os
from tqdm import tqdm
import cv2
import argparse
from threadpool import ThreadPool, makeRequests

parser = argparse.ArgumentParser()
parser.add_argument('--input_path', type=str, required=True, help='input image path')
parser.add_argument('--output_path', type=str, required=True, help='output sketch path')
parser.add_argument('--postfix', type=str, default='png')
parser.add_argument('--parallel', type=int, default=12, help='Parallel num')
args = parser.parse_args()

src_paths = glob(args.input_path + '/*.{}'.format(args.postfix))
tar_path = args.output_path
size = 256
os.makedirs(tar_path, exist_ok=True)


def xdog(files):
    name = files.split('/')[-1]
    Gamma = 0.97
    Phi = 200
    Epsilon = 0.1
    k = 2.5
    Sigma = 1.5
    im = Image.open(files).convert('L')
    im2 = filters.gaussian_filter(im, Sigma)
    im3 = filters.gaussian_filter(im, Sigma * k)
    differencedIm2 = im2 - (Gamma * im3)
    edge = 1 + np.tanh(Phi * (differencedIm2 - Epsilon))
    edge = edge.clip(0, 1)

    edge *= 255
    edge = cv2.resize(edge, (size, size), interpolation=cv2.INTER_AREA)
    cv2.imwrite(tar_path + '/' + name.replace('.jpg', '.png'), edge)


with tqdm(total=len(src_paths), desc='Converting images...') as pbar:
    def callback(req, x):
        pbar.update()


    t_pool = ThreadPool(args.parallel)
    requests = makeRequests(xdog, src_paths, callback=callback)
    for req in requests:
        t_pool.putRequest(req)
    t_pool.wait()
