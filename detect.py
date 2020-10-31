import cv2
import numpy as np
import colorsys
import os
import time
import argparse
import torch
import torch.nn as nn
from Pytorch.photo_detection.YOLO_v4.yolov4_pytorch_train.nets.yolo4 import YoloBody
from Pytorch.photo_detection.YOLO_v4.yolov4_pytorch_train.trains import get_classes, get_anchors
from PIL import Image, ImageFont, ImageDraw
from Pytorch.photo_detection.YOLO_v4.yolov4_pytorch_train.utils.utils import non_max_suppression, DecodeBox, letterbox_image, yolo_correct_boxes
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 加快模型训练的效率


def args_parse():
    parser = argparse.ArgumentParser(description="检测的参数")
    parser.add_argument('-image_size', default=(416, 416), type=int, help="input image size")
    parser.add_argument('-model_path', default="./logs/Epoch27-Total_Loss11.0493-Val_Loss7.9978.pth", type=str, help="model path")
    parser.add_argument('-classes_path', default='./model_data/class_name.txt', type=str, help='classes path')
    parser.add_argument('-anchors_path', default='./model_data/anchors.txt', type=str, help='anchors_path')
    parser.add_argument('-confidence', default=0.2, type=float, help='confidence')
    parser.add_argument('-iou', default=0.2, type=float, help="iou")
    parser.add_argument('-cuda', default=True, type=bool, help="is't use cuda")
    args = parser.parse_args()
    return args


def detect_image(image):
    args = args_parse()
    iou = args.iou
    model_image_size = args.image_size  # 图片尺寸
    model_path = args.model_path  # 模型路径
    confidence = args.confidence  # 置信度

    anchors = args.anchors_path
    anchors = get_anchors(anchors)  # 锚框
    class_names = args.classes_path
    class_names = get_classes(class_names)  # 类别

    net = YoloBody(len(anchors[0]), len(class_names)).eval()
    print('Loading weights into state dict...')
    net.load_state_dict(torch.load(model_path, map_location="cuda:0"), False)

    cuda = True
    if cuda:
        os.environ["CUDA_VISIBLE_DEVICES"] = '0'
        net = nn.DataParallel(net)
        net = net.to(device)

    print('Finished!')

    yolo_decodes = []
    for i in range(3):
        yolo_decodes.append(
            DecodeBox(anchors[i], len(class_names), (model_image_size[1], model_image_size[0])))

    print('{} model, anchors, and classes loaded.'.format(model_path))
    # 画框设置不同的颜色
    hsv_tuples = [(x / len(class_names), 1., 1.)
                  for x in range(len(class_names))]
    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))

    image_shape = np.array(np.shape(image)[0:2])

    # letterbox_image将原始的图片不失真的resize，添加灰度框，使之符合网络输入图片的尺寸
    crop_img = np.array(letterbox_image(image, (args.image_size[0], args.image_size[1])))
    photo = np.array(crop_img, dtype=np.float32)  # 转换成numpy形式
    photo = photo.astype(np.float32) / 255.0  # 将读取的图片矩阵数值从（0~255）归一化到（0~1），得到全黑的图片，*255得彩色图片
    photo = np.transpose(photo, (2, 0, 1))  # 转换图片的维度，通道数放在高和宽的前面
    # batch_size的shape为（batch_size, (channels, height, width)）pytorch要将图片做成一个batch_size的维度才可以训练
    images = []
    images.append(photo)  # 扩充一个维度
    images = np.asarray(images)  # 将图片做成一个batch_size的维度才可以训练  将列表转换为数组，不会复制列表

    with torch.no_grad():
        images = torch.from_numpy(images)
        cuda = True
        if cuda:
            images = images.cuda()  # 调用cuda
            net = net  # 模型
        outputs = net(images)  # 得到模型的预测结果

    output_list = []
    for i in range(3):  # 3个有效特征层
        output_list.append(yolo_decodes[i](outputs[i]))  # 利用预测结果对先验框进行解码，获得最后的预测框，判断先验框内部是否包含物体，及先验框内部物体的种类
    output = torch.cat(output_list, 1)  # 对在给定的1(dim)维的待连接的张量序列进行连接操作，dim在（0, len(output_list[0])）之间
    batch_detections = non_max_suppression(output, len(class_names),  # 筛选出一定区域内得分最大的矩形框
                                           conf_thres=confidence,
                                           nms_thres=iou)
    try:
        batch_detections = batch_detections[0].cpu().numpy()
    except:
        return image

    top_index = batch_detections[:, 4] * batch_detections[:, 5] > confidence  # 先验框内部是否存在目标*属于某个种类
    top_conf = batch_detections[top_index, 4] * batch_detections[top_index, 5]  # 得到最后的置信度
    top_label = np.array(batch_detections[top_index, 6], np.int32)  # 得到最后的类别
    top_bboxes = np.array(batch_detections[top_index, :4])  # 得到最后的边界框
    top_xmin, top_ymin, top_xmax, top_ymax = np.expand_dims(top_bboxes[:, 0], -1), \
                                             np.expand_dims(top_bboxes[:, 1], -1), \
                                             np.expand_dims(top_bboxes[:, 2], -1), \
                                             np.expand_dims(top_bboxes[:, 3], -1)  # 扩展维度

    # 现在得到的检测结果是带灰框的，去掉灰条
    boxes = yolo_correct_boxes(top_ymin, top_xmin, top_ymax, top_xmax,
                               np.array([model_image_size[0], model_image_size[1]]), image_shape)

    font = ImageFont.truetype(font='model_data/simhei.ttf',
                              size=np.floor(3e-2 * np.shape(image)[1] + 0.5).astype('int32'))

    thickness = (np.shape(image)[0] + np.shape(image)[1]) // model_image_size[0]

    for i, c in enumerate(top_label):
        predicted_class = class_names[c]  # 所属的类
        score = top_conf[i]  # 得分

        top, left, bottom, right = boxes[i]
        top = top - 5
        left = left - 5
        bottom = bottom + 5
        right = right + 5

        top = max(0, np.floor(top + 0.5).astype('int32'))
        left = max(0, np.floor(left + 0.5).astype('int32'))
        bottom = min(np.shape(image)[0], np.floor(bottom + 0.5).astype('int32'))
        right = min(np.shape(image)[1], np.floor(right + 0.5).astype('int32'))

        # 画框框
        label = '{} {:.2f}'.format(predicted_class, score)
        draw = ImageDraw.Draw(image)
        label_size = draw.textsize(label, font)
        label = label.encode('utf-8')
        print(label)

        if top - label_size[1] >= 0:
            text_origin = np.array([left, top - label_size[1]])
        else:
            text_origin = np.array([left, top + 1])

        for i in range(thickness):
            draw.rectangle(
                [left + i, top + i, right - i, bottom - i],  # 矩形框
                outline=colors[class_names.index(predicted_class)])
        draw.rectangle(
            [tuple(text_origin), tuple(text_origin + label_size)],
            fill=colors[class_names.index(predicted_class)])
        draw.text(text_origin, str(label, 'UTF-8'), fill=(0, 0, 0), font=font)
        del draw
    return image


if __name__ == "__main__":
    is_image = True
    if is_image:
        image = input("请输入要检测的文件：")
        image = Image.open(image)
        result = detect_image(image)
        result.show()
    else:
        # capture = cv2.VideoCapture(0)
        capture = cv2.VideoCapture("./videos/cars.mp4")
        fps = 0.3
        while (True):
            start = time.time()
            ref, frame = capture.read()  # 读视频帧
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # 格式转变，BGRtoRGB
            frame = Image.fromarray(np.uint8(frame))  # 转变成Image
            frame = np.array(detect_image(frame))  # 进行检测
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)  # RGBtoBGR满足opencv显示格式

            fps = (fps + (1. / (time.time() - start))) / 2
            print("fps= %.2f" % (fps))
            frame = cv2.putText(frame, "fps= %.2f" % (fps), (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv2.namedWindow("videos.mp4", cv2.WINDOW_NORMAL)
            cv2.imshow("videos.mp4", frame)

            key = cv2.waitKey(1)
            if key & 0xFF == "q":  # 视频或摄像头用1，图像使用0或空
                break
        capture.release()
