from tqdm import tqdm
import os
import torch
import numpy as np
import cv2
from scipy import linalg
from glob import glob
from torch.autograd import Variable
from torch.nn.functional import adaptive_avg_pool2d
from skimage.measure import compare_ssim
from skimage.color import rgb2gray
# from skimage.measure._structural_similarity import structural_similarity
from utils.inception import InceptionV3


def get_activations(images, model, batch_size=64, dims=2048,
                    cuda=False, verbose=False):
    """Calculates the activations of the pool_3 layer for all images.
    Params:
    -- images      : Numpy array of dimension (n_images, 3, hi, wi). The values
                     must lie between 0 and 1.
    -- model       : Instance of inception model
    -- batch_size  : the images numpy array is split into batches with
                     batch size batch_size. A reasonable batch size depends
                     on the hardware.
    -- dims        : Dimensionality of features returned by Inception
    -- cuda        : If set to True, use GPU
    -- verbose     : If set to True and parameter out_step is given, the number
                     of calculated batches is reported.
    Returns:
    -- A numpy array of dimension (num images, dims) that contains the
       activations of the given tensor when feeding inception with the
       query tensor.
    """
    model.eval()

    d0 = images.shape[0]
    if batch_size > d0:
        print(('Warning: batch size is bigger than the data size. '
               'Setting batch size to data size'))
        batch_size = d0

    n_batches = d0 // batch_size
    if d0 % batch_size != 0:
        n_batches += 1
    n_used_imgs = d0

    pred_arr = np.empty((n_used_imgs, dims))
    with torch.no_grad():
        for i in tqdm(range(n_batches)):
            if verbose:
                print('\rPropagating batch %d/%d' % (i + 1, n_batches),
                      end='', flush=True)
            start = i * batch_size
            end = min(start + batch_size, d0)

            batch = torch.from_numpy(images[start:end]).type(torch.FloatTensor)
            batch = Variable(batch)
            if cuda:
                batch = batch.cuda()

            pred = model(batch)[0]

            # If model output is not scalar, apply global spatial average pooling.
            # This happens if you choose a dimensionality not equal 2048.
            if pred.shape[2] != 1 or pred.shape[3] != 1:
                pred = adaptive_avg_pool2d(pred, output_size=(1, 1))

            pred_arr[start:end] = pred.cpu().data.numpy().reshape(end - start, -1)

        if verbose:
            print(' done')

    return pred_arr


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).
    Stable version by Dougal J. Sutherland.
    Params:
    -- mu1   : Numpy array containing the activations of a layer of the
               inception net (like returned by the function 'get_predictions')
               for generated samples.
    -- mu2   : The sample mean over activations, precalculated on an
               representive data set.
    -- sigma1: The covariance matrix over activations for generated samples.
    -- sigma2: The covariance matrix over activations, precalculated on an
               representive data set.
    Returns:
    --   : The Frechet Distance.
    """
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            # raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1) +
            np.trace(sigma2) - 2 * tr_covmean)


def calculate_activation_statistics(images, model, batch_size=64,
                                    dims=2048, cuda=False, verbose=False):
    """Calculation of the statistics used by the FID.
    Params:
    -- images      : Numpy array of dimension (n_images, 3, hi, wi). The values
                     must lie between 0 and 1.
    -- model       : Instance of inception model
    -- batch_size  : The images numpy array is split into batches with
                     batch size batch_size. A reasonable batch size
                     depends on the hardware.
    -- dims        : Dimensionality of features returned by Inception
    -- cuda        : If set to True, use GPU
    -- verbose     : If set to True and parameter out_step is given, the
                     number of calculated batches is reported.
    Returns:
    -- mu    : The mean over samples of the activations of the pool_3 layer of
               the inception model.
    -- sigma : The covariance matrix of the activations of the pool_3 layer of
               the inception model.
    """
    act = get_activations(images, model, batch_size, dims, cuda, verbose)
    mu = np.mean(act, axis=0)
    sigma = np.cov(act, rowvar=False)
    return mu, sigma


def _compute_statistics_of_path(path, model, batch_size, dims, cuda):
    npz_file = os.path.join(path, 'statistics.npz')
    if os.path.exists(npz_file):
        f = np.load(npz_file)
        m, s = f['mu'][:], f['sigma'][:]
        f.close()
    else:
        # path = pathlib.Path(path)
        # files = list(path.glob('*.jpg')) + list(path.glob('*.png'))
        files = list(glob(path + '/*.jpg')) + list(glob(path + '/*.png'))
        files = sorted(files, key=lambda x: x.split('/')[-1])

        imgs = []
        for fn in files:
            imgs.append(cv2.imread(str(fn)).astype(np.float32)[:, :, ::-1])
        imgs = np.array(imgs)

        # Bring images to shape (B, 3, H, W)
        imgs = imgs.transpose((0, 3, 1, 2))

        # Rescale images to be between 0 and 1
        imgs /= 255

        m, s = calculate_activation_statistics(imgs, model, batch_size, dims, cuda)
        # np.savez(npz_file, mu=m, sigma=s)

    return m, s


def calculate_fid_given_paths(paths, batch_size, cuda, dims):
    """Calculates the FID of two paths"""
    # for p in paths:
    #     if not os.path.exists(p):
    #         raise RuntimeError('Invalid path: %s' % p)

    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]

    model = InceptionV3([block_idx])
    if cuda:
        model.cuda()

    print('calculate path1 statistics...')
    m1, s1 = _compute_statistics_of_path(paths[0], model, batch_size, dims, cuda)
    print('calculate path2 statistics...')
    m2, s2 = _compute_statistics_of_path(paths[1], model, batch_size, dims, cuda)
    print('calculate frechet distance...')
    fid_value = calculate_frechet_distance(m1, s1, m2, s2)

    return fid_value


def get_inpainting_metrics(src, tgt, logger, fid_test=True):
    input_paths = sorted(glob(src + '/*'), key=lambda x: x.split('/')[-1])
    output_paths = sorted(glob(tgt + '/*'), key=lambda x: x.split('/')[-1])
    assert len(input_paths) == len(output_paths), (len(input_paths), len(output_paths))

    # PSNR and SSIM
    psnrs = []
    ssims = []
    maes = []
    mses = []
    max_value = 1.0
    for p1, p2 in tqdm(zip(input_paths, output_paths)):
        img1 = cv2.imread(p1)
        if img1 is None:
            print(p1, 'is bad image!')
        img2 = cv2.imread(p2)
        if img2 is None:
            print(p2, 'is bad image!')

        mse_ = np.mean((img1 / 255.0 - img2 / 255.0) ** 2)
        mae_ = np.mean(abs(img1 / 255.0 - img2 / 255.0))
        psnr_ = max_value - 10 * np.log(mse_ + 1e-7) / np.log(10)
        ssim_ = compare_ssim(rgb2gray(img1), rgb2gray(img2))
        psnrs.append(psnr_)
        ssims.append(ssim_)
        mses.append(mse_)
        maes.append(mae_)

    psnr = np.mean(psnrs)
    ssim = np.mean(ssims)
    mse = np.mean(mses)
    mae = np.mean(maes)

    res_text = 'PSNR:{0:.3f}, SSIM:{1:.3f}, MSE:{2:.6f}, MAE:{3:.6f}'.format(psnr, ssim, mse, mae)
    res = {'psnr': psnr, 'ssim': ssim, 'mse': mse, 'mae': mae}
    # FID
    if fid_test:
        fid = calculate_fid_given_paths([src, tgt], batch_size=16, cuda=True, dims=2048)
        res['fid'] = fid
        res_text += ', FID:{0:.3f}'.format(fid)
    res_text = '\n' + res_text + '\n'
    if logger is None:
        print(res_text)
    else:
        logger.info(res_text)
    return res


def get_sketch_metrics(input_paths, output_paths, logger, th=175):
    assert len(input_paths) == len(output_paths), (len(input_paths), len(output_paths))

    precisions = []
    recalls = []
    f1s = []
    maes = []
    mses = []
    for p1, p2 in tqdm(zip(input_paths, output_paths)):
        img1 = cv2.imread(p1)
        if img1 is None:
            print(p1, 'is bad image!')
        img2 = cv2.imread(p2)
        if img2 is None:
            print(p2, 'is bad image!')

        mse_ = np.mean((img1 / 255.0 - img2 / 255.0) ** 2)
        mae_ = np.mean(abs(img1 / 255.0 - img2 / 255.0))
        mses.append(mse_)
        maes.append(mae_)

        # flip, and 1 is black and 0 is white
        label = 1 - (img1 > th).astype(np.float)
        output = 1 - (img2 > th).astype(np.float)
        true_positive = ((label == output) * label)
        relevant = np.sum(label)
        selected = np.sum(output)
        recall = np.sum(true_positive) / (relevant + 1e-8)
        precision = np.sum(true_positive) / (selected + 1e-8)
        f1_score = (2 * precision * recall) / (precision + recall + 1e-8)
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1_score)

    mse = np.mean(mses)
    mae = np.mean(maes)
    precision = np.mean(precisions)
    recall = np.mean(recalls)
    f1 = np.mean(f1s)

    res_text = 'Precision:{0:.3f}, Recall:{1:.3f}, F1:{2:.3f}, ' \
               'MSE:{3:.6f}, MAE:{4:.6f}'.format(precision, recall, f1, mse, mae)
    res = {'precision': precision, 'recall': recall, 'f1': f1, 'mse': mse, 'mae': mae}
    res_text = '\n' + res_text + '\n'
    if logger is None:
        print(res_text)
    else:
        logger.info(res_text)
    return res
