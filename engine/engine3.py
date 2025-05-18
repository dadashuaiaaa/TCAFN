import os
import time
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from utils.dataset import tokenize
from utils.misc import concat_all_gather
from loguru import logger


@torch.no_grad()
def inference(test_loader, model, args):
    iou_list = []
    tbar = tqdm(test_loader, desc='Inference:', ncols=100)
    model.eval()
    time.sleep(2)
    
    for img, param in tbar:
        # 数据预处理
        img = img.cuda(non_blocking=True)
        mask = cv2.imread(param['mask_dir'][0], flags=cv2.IMREAD_GRAYSCALE)

        # 保存原始图片 & mask
        if args.visualize:
            seg_id = param['seg_id'][0].cpu().numpy()
            img_name = '{}-img.jpg'.format(seg_id)
            mask_name = '{}-mask.png'.format(seg_id)

            # 保存原始图像
            ori_img = param['ori_img'][0].cpu().numpy()
            if len(ori_img.shape) == 2:  # 如果是灰度图，转换为 BGR
                ori_img = cv2.cvtColor(ori_img, cv2.COLOR_GRAY2BGR)
            cv2.imwrite(os.path.join(args.vis_dir, img_name), ori_img)

            # 保存掩码
            cv2.imwrite(os.path.join(args.vis_dir, mask_name), mask)

        # 处理多个文本描述
        for sent in param['sents']:
            mask = mask / 255.  # 归一化
            text = tokenize(sent, args.word_len, True).cuda(non_blocking=True)

            # 预测
            pred = model(img, text)
            pred = torch.sigmoid(pred)
            if pred.shape[-2:] != img.shape[-2:]:
                pred = F.interpolate(pred, size=img.shape[-2:], mode='bicubic', align_corners=True).squeeze()

            # 恢复原图尺寸
            h, w = param['ori_size'].numpy()[0]
            mat = param['inverse'].numpy()[0]
            pred = pred.cpu().numpy()
            pred = cv2.warpAffine(pred, mat, (w, h), flags=cv2.INTER_CUBIC, borderValue=0.)
            pred = np.array(pred > 0.35)  # 二值化

            # IoU 计算
            inter = np.logical_and(pred, mask)
            union = np.logical_or(pred, mask)
            iou = np.sum(inter) / (np.sum(union) + 1e-6)
            iou_list.append(iou)

            # 生成可视化叠加图
            if args.visualize:
                pred_mask = np.array(pred * 255, dtype=np.uint8)  # 变成 0-255
                pred_mask = cv2.applyColorMap(pred_mask, cv2.COLORMAP_JET)  # 伪彩色

                # 叠加掩码到原图
                overlay = cv2.addWeighted(ori_img, 0.6, pred_mask, 0.4, 0)

                # 保存叠加后的图片
                sent_str = "_".join(sent[0].split(" "))
                overlay_name = '{}-iou={:.2f}-{}.png'.format(seg_id, iou * 100, sent_str)
                cv2.imwrite(os.path.join(args.vis_dir, overlay_name), overlay)

    # 计算整体评估指标
    logger.info('=> Metric Calculation <=')
    iou_list = np.stack(iou_list)
    iou_list = torch.from_numpy(iou_list).to(img.device)
    
    prec_list = [(iou_list > thres).float().mean() for thres in torch.arange(0.5, 1.0, 0.1)]
    iou = iou_list.mean()

    # 记录 Precision
    prec = {}
    for i, thres in enumerate(range(5, 10)):
        key = 'Pr@{}'.format(thres * 10)
        prec[key] = prec_list[i].item()
    
    # 打印结果
    logger.info('Mean IoU={:.2f}'.format(100. * iou.item()))
    for k, v in prec.items():
        logger.info('{}: {:.2f}.'.format(k, 100. * v))

    return iou.item(), prec
