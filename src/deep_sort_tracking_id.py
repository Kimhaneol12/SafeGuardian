import argparse
import time
from pathlib import Path
import os
import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, non_max_suppression, scale_coords, xyxy2xywh, set_logging, increment_path
from utils.torch_utils import select_device, time_synchronized, TracedModel

from deep_sort_pytorch.utils.parser import get_config
from deep_sort_pytorch.deep_sort import DeepSort


def xyxy_to_xywh(*xyxy):
    bbox_left = min([xyxy[0].item(), xyxy[2].item()])
    bbox_top = min([xyxy[1].item(), xyxy[3].item()])
    bbox_w = abs(xyxy[0].item() - xyxy[2].item())
    bbox_h = abs(xyxy[1].item() - xyxy[3].item())
    x_c = (bbox_left + bbox_w / 2)
    y_c = (bbox_top + bbox_h / 2)
    return x_c, y_c, bbox_w, bbox_h


def compute_color_for_labels(label):
    palette = (2**11 - 1, 2**15 - 1, 2**20 - 1)
    color = [int((p * (label**2 - label + 1)) % 255) for p in palette]
    return tuple(color)


def draw_boxes(img, bbox, identities=None, offset=(0, 0)):
    for i, box in enumerate(bbox):
        x1, y1, x2, y2 = [int(i) for i in box]
        x1 += offset[0]
        x2 += offset[0]
        y1 += offset[1]
        y2 += offset[1]
        id = int(identities[i]) if identities is not None else 0
        color = compute_color_for_labels(id)
        label = f'ID {id}'
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)
    return img


def load_classes(path):
    with open(path, 'r') as f:
        names = f.read().split('\n')
    return list(filter(None, names))


def detect(opt):
    names, source, weights, view_img, save_txt, imgsz, trace = (
        opt.names, opt.source, opt.weights, opt.view_img, opt.save_txt,
        opt.img_size, not opt.no_trace
    )
    save_img = not opt.nosave and not source.endswith('.txt')
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://', 'https://'))

    save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)

    cfg_deep = get_config()
    cfg_deep.merge_from_file("deep_sort_pytorch/configs/deep_sort.yaml")
    deepsort = DeepSort(cfg_deep.DEEPSORT.REID_CKPT,
                        max_dist=cfg_deep.DEEPSORT.MAX_DIST,
                        min_confidence=cfg_deep.DEEPSORT.MIN_CONFIDENCE,
                        nms_max_overlap=cfg_deep.DEEPSORT.NMS_MAX_OVERLAP,
                        max_iou_distance=cfg_deep.DEEPSORT.MAX_IOU_DISTANCE,
                        max_age=cfg_deep.DEEPSORT.MAX_AGE,
                        n_init=cfg_deep.DEEPSORT.N_INIT,
                        nn_budget=cfg_deep.DEEPSORT.NN_BUDGET,
                        use_cuda=True)

    set_logging()
    device = select_device(opt.device)
    half = device.type != 'cpu'

    model = attempt_load(weights, map_location=device)
    stride = int(model.stride.max())
    imgsz = check_img_size(imgsz, s=stride)

    if trace:
        model = TracedModel(model, device, opt.img_size)
    if half:
        model.half()

    vid_path, vid_writer = None, None
    dataset = LoadStreams(source, img_size=imgsz, stride=stride) if webcam else LoadImages(source, img_size=imgsz, stride=stride)
    names = load_classes(names)

    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))

    t0 = time.time()
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()
        img /= 255.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        t1 = time_synchronized()
        with torch.no_grad():
            pred = model(img, augment=opt.augment)[0]
        t2 = time_synchronized()

        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=[0], agnostic=opt.agnostic_nms)
        t3 = time_synchronized()

        for i, det in enumerate(pred):
            if webcam:
                p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count
            else:
                p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

            p = Path(p)
            save_path = str(save_dir / p.name)
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]

            if len(det):
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
                xywh_bboxs, confs, oids = [], [], []

                for *xyxy, conf, cls in reversed(det):
                    x_c, y_c, bbox_w, bbox_h = xyxy_to_xywh(*xyxy)
                    xywh_bboxs.append([x_c, y_c, bbox_w, bbox_h])
                    confs.append([conf.item()])
                    oids.append(int(cls))

                xywhs = torch.Tensor(xywh_bboxs)
                confss = torch.Tensor(confs)

                outputs = deepsort.update(xywhs, confss, oids, im0)
                if len(outputs) > 0:
                    bbox_xyxy = outputs[:, :4]
                    identities = outputs[:, -2]
                    frame_num = int(frame) if not webcam else int(time.time() * 1000)
                    video_id = p.stem
                    for i, box in enumerate(bbox_xyxy):
                        id = int(identities[i])
                        x1, y1, x2, y2 = map(int, box)
                        x1 = max(0, x1)
                        y1 = max(0, y1)
                        x2 = min(im0.shape[1], x2)
                        y2 = min(im0.shape[0], y2)
                        crop = im0[y1:y2, x1:x2]
                        crop_dir = os.path.join("outputs/crops", video_id, f"id_{id}")
                        os.makedirs(crop_dir, exist_ok=True)
                        crop_filename = os.path.join(crop_dir, f"frame_{frame_num:06d}.jpg")
                        cv2.imwrite(crop_filename, crop)
                        print(f"[SAVE] id={id} → {crop_filename}")

        if view_img:
            cv2.imshow(str(p), im0)
            cv2.waitKey(1)

    print(f'Done. ({time.time() - t0:.3f}s)')


def run_tracking(weights, source, img_size=640, conf_thres=0.5, iou_thres=0.25,
                 device='', save_dir='outputs/crops', classes=[0], view_img=False):
    import argparse
    opt = argparse.Namespace()
    opt.weights = weights
    opt.source = source
    opt.img_size = img_size
    opt.conf_thres = conf_thres
    opt.iou_thres = iou_thres
    opt.device = device
    opt.save_txt = False
    opt.save_conf = False
    opt.nosave = False
    opt.classes = classes
    opt.agnostic_nms = False
    opt.augment = False
    opt.update = False
    opt.project = save_dir
    opt.name = ''
    opt.exist_ok = True
    opt.no_trace = True
    opt.view_img = view_img
    opt.trailslen = 64
    opt.names = 'data/coco.names'
    detect(opt)
