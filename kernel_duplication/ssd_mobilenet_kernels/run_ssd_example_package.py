from vision.ssd.vgg_ssd import create_vgg_ssd, create_vgg_ssd_predictor
from vision.ssd.mobilenetv1_ssd import create_mobilenetv1_ssd, create_mobilenetv1_ssd_predictor
from vision.ssd.mobilenetv1_ssd_lite import create_mobilenetv1_ssd_lite, create_mobilenetv1_ssd_lite_predictor
from vision.ssd.squeezenet_ssd_lite import create_squeezenet_ssd_lite, create_squeezenet_ssd_lite_predictor
from vision.ssd.mobilenet_v2_ssd_lite import create_mobilenetv2_ssd_lite, create_mobilenetv2_ssd_lite_predictor
from vision.utils.misc import Timer
import cv2
import sys
import os
import time
import os.path          as osp
import torch
import copy

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


if len(sys.argv) < 5:
    print('Usage: python run_ssd_example.py <net type>  <model path> <label path> <image path> <output path>')
    sys.exit(0)
net_type = sys.argv[1]
model_path = sys.argv[2]
label_path = sys.argv[3]
image_path = sys.argv[4]
# out_path = sys.argv[5]
# dup = sys.argv[6]
# err = sys.argv[7]

out_path = "detection/"

if not os.path.exists(out_path):
    os.mkdir(out_path)

class_names = [name.strip() for name in open(label_path).readlines()]

def weight_sum_eval(model):
    weights = model.state_dict()
    evaluation = []
    names = []
    # need to find the connection between conv and fc
    for i in model.all_layer_indices:
        evaluation.append(weights['base_net.' + str(i + 1) + '.3' + '.weight'].detach().clone().abs().sum(dim=3).sum(dim=2).sum(dim=0))
    return evaluation, names

if net_type == 'vgg16-ssd':
    net = create_vgg_ssd(len(class_names), is_test=True)
elif net_type == 'mb1-ssd':
    net = create_mobilenetv1_ssd(len(class_names), is_test=True)
elif net_type == 'mb1-ssd-lite':
    net = create_mobilenetv1_ssd_lite(len(class_names), is_test=True)
elif net_type == 'mb2-ssd-lite':
    net = create_mobilenetv2_ssd_lite(len(class_names), is_test=True)
elif net_type == 'sq-ssd-lite':
    net = create_squeezenet_ssd_lite(len(class_names), is_test=True)
else:
    print("The net type is wrong. It should be one of vgg16-ssd, mb1-ssd and mb1-ssd-lite.")
    sys.exit(1)
net.load(model_path)
net.error = 0.01
net.run_original = False
net.duplicated = False
# net.duplicated = True
net.to(DEVICE)
# for i in net.all_layer_indices:
#     net.weights_copy[i] = copy.deepcopy(net.base_net[i])
#     net.weights_copy[i].eval()
# net.percentage = 0.5
#
# print("d2nn:")
# for k in net.all_layer_indices:
#     index = torch.arange(net.all_width[k - 1]).type(torch.float).to(DEVICE)
#     weight_sum, _ = weight_sum_eval(net)
#     tmp = weight_sum[k - 1]
#     final = torch.stack((tmp, index), axis=0)
#     final = final.sort(dim=1, descending=True)
#     net.all_duplication_indices[k] = final.indices[0]
net.error_injection_weights_all(0.01)

if net_type == 'vgg16-ssd':
    predictor = create_vgg_ssd_predictor(net, candidate_size=200)
elif net_type == 'mb1-ssd':
    predictor = create_mobilenetv1_ssd_predictor(net, candidate_size=200)
elif net_type == 'mb1-ssd-lite':
    predictor = create_mobilenetv1_ssd_lite_predictor(net, candidate_size=200)
elif net_type == 'mb2-ssd-lite':
    predictor = create_mobilenetv2_ssd_lite_predictor(net, candidate_size=200)
elif net_type == 'sq-ssd-lite':
    predictor = create_squeezenet_ssd_lite_predictor(net, candidate_size=200)
else:
    predictor = create_vgg_ssd_predictor(net, candidate_size=200)


# time_start = time.time()
frame_ctr = 0
for i_path in sorted(os.listdir(image_path)):
    # print(image_path+i_path)
    orig_image = cv2.imread(image_path+i_path)

    image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)
    err_t = 0
    time_start = time.time()
    boxes, labels, probs, err_t = predictor.predict(image, 10, 0.4)

    for i in range(boxes.size(0)):
        box = boxes[i, :]
        cv2.rectangle(orig_image, (box[0], box[1]), (box[2], box[3]), (255, 255, 0), 2)
        # label = f"""{voc_dataset.class_names[labels[i]]}: {probs[i]:.2f}"""
        label = f"{class_names[labels[i]]}: {probs[i]:.2f}"
        # print(f"{probs[i]:.2f}")
        # print(class_names[labels[i]], float(probs[i]))
        # label = class_names[labels[i]] + str(probs[i])
        cv2.putText(orig_image, label,
                    (box[0] + 2, box[1] + 4),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,  # font scale
                    (255, 0, 255),
                    1)  # line type

    time_now = time.time()
    # print(time_now - time_start)
    fps = 1 / (time_now - time_start - err_t)
    print(f"fps: {fps:.2f}, {err_t:2f}")
    # time_start = time_now
    cv2.putText(orig_image, f"fps: {fps:.2f}",
                (5, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2,  # font scale
                (255, 255, 255),
                2)  # line type
    #path = out_path + str(image_path.split("/")[-1].split(".")[0])+".jpeg"
    path = out_path + i_path
    cv2.imwrite(path, orig_image)
    #cv2.imshow("file", orig_image)
    # print(f"Found {len(probs)} objects. The output image is {path}")

    # frame_ctr = frame_ctr + 1
    # if frame_ctr == 10:
    #     time_now = time.time()
    #     fps = frame_ctr / (time_now - time_start)
    #     print("fps: %f" % fps)
    #     frame_ctr = 0
    #     time_start = time_now
