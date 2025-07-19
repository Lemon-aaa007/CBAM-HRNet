# paddleseg\core\predict.py
"""
预测遥感大影像
使用gdal分块读取影响并预测和分开写入tif，写出时与原区域取最大值
坑：cv2读图片的顺序是第3、2、1波段[h,w,c]形式，gdal则是从第1波段开始顺序读取[c,h,w].
因此，需要在gdal读之后需要将第一波段和第三波段交换顺序，然后将通道移到最后一维
# 交换第一波段和第三波段
data[[0, 2], :] = data[[2, 0], :]
# [c,h,w]转为[h,w,c]的形式，第0通道放在最后
im = data.transpose((1, 2, 0))
"""

import os
import math
#import cv2
import numpy as np
import paddle
from paddleseg import utils
from paddleseg.core import infer
from paddleseg.utils import logger, progbar
from osgeo import gdal


def mkdir(path):
    sub_dir = os.path.dirname(path)
    if not os.path.exists(sub_dir):
        os.makedirs(sub_dir)


def partition_list(arr, m):
    """split the list 'arr' into m pieces"""
    n = int(math.ceil(len(arr) / float(m)))
    return [arr[i:i + n] for i in range(0, len(arr), n)]


def predict(model,
            model_path,
            transforms,
            image_list,
            image_dir=None,
            save_dir='output',
            aug_pred=False,
            scales=1.0,
            flip_horizontal=True,
            flip_vertical=False,
            is_slide=False,
            stride=None,
            crop_size=None):
    """
    predict and visualize the image_list.

    Args:
        model (nn.Layer): Used to predict for input image.
        model_path (str): The path of pretrained model.
        transforms (transform.Compose): Preprocess for input image.
        image_list (list): A list of image path to be predicted.
        image_dir (str, optional): The root directory of the images predicted. Default: None.
        save_dir (str, optional): The directory to save the visualized results. Default: 'output'.
        aug_pred (bool, optional): Whether to use mulit-scales and flip augment for predition. Default: False.
        scales (list|float, optional): Scales for augment. It is valid when `aug_pred` is True. Default: 1.0.
        flip_horizontal (bool, optional): Whether to use flip horizontally augment. It is valid when `aug_pred` is True. Default: True.
        flip_vertical (bool, optional): Whether to use flip vertically augment. It is valid when `aug_pred` is True. Default: False.
        is_slide (bool, optional): Whether to predict by sliding window. Default: False.
        stride (tuple|list, optional): The stride of sliding window, the first is width and the second is height.
            It should be provided when `is_slide` is True.
        crop_size (tuple|list, optional):  The crop size of sliding window, the first is width and the second is height.
            It should be provided when `is_slide` is True.

    """
    utils.utils.load_entire_model(model, model_path)
    model.eval()
    nranks = paddle.distributed.get_world_size()
    local_rank = paddle.distributed.get_rank()

    img_lists = [image_list]

    tif_saved_dir = save_dir
    if not os.path.exists(tif_saved_dir):
        os.mkdir(tif_saved_dir)
    logger.info("Start to predict...")

    # for i, im_path in enumerate(img_lists[local_rank]):
    im_path = img_lists[0][0]

    if image_dir is not None:
        im_file = im_path.replace(image_dir, '')
    else:
        im_file = os.path.basename(im_path)
    if im_file[0] == '/' or im_file[0] == '\\':
        im_file = im_file[1:]
    # 输入 im_path 输出outtif
    outtif = os.path.join(tif_saved_dir, im_file.rsplit(".")[0] + "_predict.tif")

    # 分片的像元行列数，输出为正方形的
    size = 512
    lap = 128

    # 区分每块的位置，000到999
    # dex = ["%.3d" % i for i in range(1000)]

    dataset = gdal.Open(im_path)
    projinfo = dataset.GetProjection()
    geotransform = dataset.GetGeoTransform()
    # 总行列数
    cols, rows = dataset.RasterXSize, dataset.RasterYSize
    # 获取分片的行数和列数
    numx = int(np.ceil((cols - size) / (size - lap) + 1))
    numy = int(np.ceil((rows - size) / (size - lap) + 1))
    progbar_pred = progbar.Progbar(target=numx * numy, verbose=1)
    # numx,numy=int(np.ceil(cols/size)),int(np.ceil(rows/size))

    numxs = [(size - lap) * (i - 1) for i in range(1, numx + 1)]
    numys = [(size - lap) * (i - 1) for i in range(1, numy + 1)]

    bandsNum = 3

    # 创建写出图像
    format = "GTiff"
    driver = gdal.GetDriverByName(format)
    outcols = numxs[-1] + size
    outrows = numys[-1] + size
    dst_ds = driver.Create(outtif, outcols, outrows, 1, gdal.GDT_Byte, options=['COMPRESS=LZW'])
    dst_ds.SetGeoTransform(geotransform)
    dst_ds.SetProjection(projinfo)
    outband = dst_ds.GetRasterBand(1)
    outband.SetNoDataValue(0)
    count = 0
    for j in numys:
        for i in numxs:
            if (i + size > cols) | (j + size > rows):
                # x向越界
                if (i + size > cols) & (j + size <= rows):
                    data2 = dataset.ReadAsArray(i, j, cols - i, size)
                # y向越界
                elif (i + size <= cols) & (j + size > rows):
                    data2 = dataset.ReadAsArray(i, j, size, rows - j)
                # xy方向均越界
                else:
                    data2 = dataset.ReadAsArray(i, j, cols - i, rows - j)
                smally, smallx = data2.shape[1:]

                # 创建一个画布，把其余的传进去
                data1 = np.zeros((3, size, size), dtype=np.uint8)
                data1[:, 0:smally, 0:smallx] = data2[0:3]

            else:
                data1 = dataset.ReadAsArray(i, j, size, size)[0:3]
            #

            # 交换第一波段和第三波段
            data1[[0, 2], :] = data1[[2, 0], :]
            # 转为opencv的格式
            im = data1.transpose((1, 2, 0))
            """
            此处添加预测的代码，输入img
            """
            with paddle.no_grad():
                # im = cv2.imread(im_path)
                ori_shape = im.shape[:2]
                im, _ = transforms(im)
                im = im[np.newaxis, ...]
                im = paddle.to_tensor(im)

                if aug_pred:
                    pred = infer.aug_inference(
                        model,
                        im,
                        ori_shape=ori_shape,
                        transforms=transforms.transforms,
                        scales=scales,
                        flip_horizontal=flip_horizontal,
                        flip_vertical=flip_vertical,
                        is_slide=is_slide,
                        stride=stride,
                        crop_size=crop_size)
                else:
                    pred = infer.inference(
                        model,
                        im,
                        ori_shape=ori_shape,
                        transforms=transforms.transforms,
                        is_slide=is_slide,
                        stride=stride,
                        crop_size=crop_size)
                pred = paddle.squeeze(pred)
                pred = pred.numpy().astype('uint8')

                # CHW转为HWC a.transpose((1,2,0))
                # HWC转为CHW a.transpose((2,0,1))
                # pred为结果
                # pred = data1[0]
                #
                origindata = outband.ReadAsArray(i, j, size, size)
                maxdata = np.maximum(pred, origindata)
                outband.WriteArray(maxdata, i, j)
                # outband.WriteArray(pred, i, j)
                count += 1
                progbar_pred.update(count)
    dst_ds = None
    dataset = None
