import torch
from vision.ssd.config import mobilenetv1_ssd_config
from vision.nn.multibox_loss import MultiboxLoss
from vision.ssd.vgg_ssd import create_vgg_ssd, create_vgg_ssd_predictor
from vision.ssd.mobilenetv1_ssd import create_mobilenetv1_ssd, create_mobilenetv1_ssd_predictor
from vision.ssd.mobilenetv1_ssd_lite import create_mobilenetv1_ssd_lite, create_mobilenetv1_ssd_lite_predictor
from vision.ssd.squeezenet_ssd_lite import create_squeezenet_ssd_lite, create_squeezenet_ssd_lite_predictor
from vision.datasets.voc_dataset import VOCDataset
from vision.datasets.open_images import OpenImagesDataset
from vision.utils import box_utils, measurements
from vision.utils.misc import str2bool, Timer
from torch.utils.data import DataLoader
from vision.ssd.data_preprocessing import TrainAugmentation
from vision.ssd.ssd import MatchPrior
import argparse
import pathlib
import numpy as np
import logging
import sys
import copy
from vision.ssd.mobilenet_v2_ssd_lite import create_mobilenetv2_ssd_lite, create_mobilenetv2_ssd_lite_predictor


parser = argparse.ArgumentParser(description="SSD Evaluation on VOC Dataset.")
parser.add_argument('--net', default="vgg16-ssd",
                    help="The network architecture, it should be of mb1-ssd, mb1-ssd-lite, mb2-ssd-lite or vgg16-ssd.")
parser.add_argument("--trained_model", type=str)

parser.add_argument("--dataset_type", default="voc", type=str,
                    help='Specify dataset type. Currently support voc and open_images.')
parser.add_argument("--dataset", type=str, help="The root directory of the VOC dataset or Open Images dataset.")
parser.add_argument("--label_file", type=str, help="The label file path.")
parser.add_argument("--use_cuda", type=str2bool, default=True)
parser.add_argument("--use_2007_metric", type=str2bool, default=True)
parser.add_argument("--nms_method", type=str, default="hard")
parser.add_argument("--iou_threshold", type=float, default=0.5, help="The threshold of Intersection over Union.")
parser.add_argument("--eval_dir", default="eval_results", type=str, help="The directory to store evaluation results.")
parser.add_argument('--mb2_width_mult', default=1.0, type=float,
                    help='Width Multiplifier for MobilenetV2')
parser.add_argument('--run_original', default=False, type=str2bool, help='train the original model')
parser.add_argument('--duplicated', default=False, type=str2bool, help='make duplication')
parser.add_argument('--error_rate', type=float, default=0, metavar='M', help='error_rate')
parser.add_argument('--num_duplication', type=int, default=0, metavar='M', help='error_rate')
# parser.add_argument('--touch_layer_index', default=1, type=int,
#                     help='how many layers to add attention, maximum 3')
parser.add_argument('--ft_type', default='none', type=str, help='Type of kernel duplication')
args = parser.parse_args()
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() and args.use_cuda else "cpu")


def group_annotation_by_class(dataset):
    true_case_stat = {}
    all_gt_boxes = {}
    all_difficult_cases = {}
    for i in range(len(dataset)):
        image_id, annotation = dataset.get_annotation(i)
        gt_boxes, classes, is_difficult = annotation
        gt_boxes = torch.from_numpy(gt_boxes)
        for i, difficult in enumerate(is_difficult):
            class_index = int(classes[i])
            gt_box = gt_boxes[i]
            if not difficult:
                true_case_stat[class_index] = true_case_stat.get(class_index, 0) + 1

            if class_index not in all_gt_boxes:
                all_gt_boxes[class_index] = {}
            if image_id not in all_gt_boxes[class_index]:
                all_gt_boxes[class_index][image_id] = []
            all_gt_boxes[class_index][image_id].append(gt_box)
            if class_index not in all_difficult_cases:
                all_difficult_cases[class_index]={}
            if image_id not in all_difficult_cases[class_index]:
                all_difficult_cases[class_index][image_id] = []
            all_difficult_cases[class_index][image_id].append(difficult)

    for class_index in all_gt_boxes:
        for image_id in all_gt_boxes[class_index]:
            all_gt_boxes[class_index][image_id] = torch.stack(all_gt_boxes[class_index][image_id])
    for class_index in all_difficult_cases:
        for image_id in all_difficult_cases[class_index]:
            all_gt_boxes[class_index][image_id] = torch.tensor(all_gt_boxes[class_index][image_id])
    return true_case_stat, all_gt_boxes, all_difficult_cases


def compute_average_precision_per_class(num_true_cases, gt_boxes, difficult_cases,
                                        prediction_file, iou_threshold, use_2007_metric):
    with open(prediction_file) as f:
        image_ids = []
        boxes = []
        scores = []
        for line in f:
            t = line.rstrip().split(" ")
            image_ids.append(t[0])
            scores.append(float(t[1]))
            box = torch.tensor([float(v) for v in t[2:]]).unsqueeze(0)
            box -= 1.0  # convert to python format where indexes start from 0
            boxes.append(box)
        scores = np.array(scores)
        sorted_indexes = np.argsort(-scores)
        boxes = [boxes[i] for i in sorted_indexes]
        image_ids = [image_ids[i] for i in sorted_indexes]
        true_positive = np.zeros(len(image_ids))
        false_positive = np.zeros(len(image_ids))
        matched = set()
        for i, image_id in enumerate(image_ids):
            box = boxes[i]
            if image_id not in gt_boxes:
                false_positive[i] = 1
                continue

            gt_box = gt_boxes[image_id]
            ious = box_utils.iou_of(box, gt_box)
            max_iou = torch.max(ious).item()
            max_arg = torch.argmax(ious).item()
            if max_iou > iou_threshold:
                if difficult_cases[image_id][max_arg] == 0:
                    if (image_id, max_arg) not in matched:
                        true_positive[i] = 1
                        matched.add((image_id, max_arg))
                    else:
                        false_positive[i] = 1
            else:
                false_positive[i] = 1

    true_positive = true_positive.cumsum()
    false_positive = false_positive.cumsum()
    precision = true_positive / (true_positive + false_positive)
    recall = true_positive / num_true_cases
    if use_2007_metric:
        return measurements.compute_voc2007_average_precision(precision, recall)
    else:
        return measurements.compute_average_precision(precision, recall)


# SubFlow calculate importance
def cal_importance(model):
    """
    s = o^2 * H

    :param model:
    :param train_data:
    :return:
    """
    model.to(DEVICE)
    importance_list = []
    config = mobilenetv1_ssd_config
    train_transform = TrainAugmentation(config.image_size, config.image_mean, config.image_std)
    target_transform = MatchPrior(config.priors, config.center_variance,
                                  config.size_variance, 0.5)
    imp_dataset = VOCDataset(args.dataset, transform=train_transform, target_transform=target_transform)
    data_loader = DataLoader(imp_dataset, 16, shuffle=True)
    criterion = MultiboxLoss(config.priors, iou_threshold=0.5, neg_pos_ratio=3,
                             center_variance=0.1, size_variance=0.2, device=DEVICE)
    for i, data in enumerate(data_loader):
        images, boxes, labels = data
        images = images.to(DEVICE)
        boxes = boxes.to(DEVICE)
        labels = labels.to(DEVICE)


        # forward pass
        confidence, locations = model(images)
        regression_loss, classification_loss = criterion(confidence, locations, labels, boxes)
        loss = regression_loss + classification_loss
        loss.backward()
        #loss = F.cross_entropy(output, target)
        #loss.backward()

        for j, out in enumerate(model.output):
            hessian_approximate = out.grad ** 2  # grad^2 -> hessian
            importance = hessian_approximate * (out.detach().clone() ** 2)
            importance_list.append(importance.mean(dim=0))
            #print(importance_list[-1].size())

        break

    return importance_list


# D2NN: weight sum evaluation
def weight_sum_eval(model):
    weights = model.state_dict()
    evaluation = []
    names = []
    # need to find the connection between conv and fc
    for name, m in model.named_modules():
        # print(name, m)
        # if name == 'base_net.2.3':
        #     names.append(name)
        #     evaluation.append(weights[name + '.weight'].detach().clone().abs().sum(dim=3).sum(dim=2).sum(dim=0))
        # if name == 'base_net.2.0':
        #     # print(weights[name + '.weight'].size())
        #     names.append(name)
        #     evaluation.append(weights[name + '.weight'].detach().clone().abs().sum(dim=3).sum(dim=2).sum(dim=1))
            # print(evaluation[0].size())
            # exit()
        if name == 'base_net.1.3':
            names.append(name)
            evaluation.append(weights[name + '.weight'].detach().clone().abs().sum(dim=3).sum(dim=2).sum(dim=1))

        # if isinstance(m, torch.nn.Conv2d):
        #     names.append(name)
        #
        #     # output input H W
        #     evaluation.append(weights[name + '.weight'].detach().clone().abs().sum(dim=3).sum(dim=2).sum(dim=0))
        #     # print(name, evaluation[-1].size(), weights[name + '.weight'].size())
        # elif isinstance(m, torch.nn.Linear):
        #     names.append(name)
        #     # output input
        #     evaluation.append(weights[name + '.weight'].detach().clone().abs().sum(dim=0))
        # if evaluation: print(names[-1], type(m), evaluation[-1].size())
    return evaluation, names


if __name__ == '__main__':
    eval_path = pathlib.Path(args.eval_dir)
    eval_path.mkdir(exist_ok=True)
    timer = Timer()
    class_names = [name.strip() for name in open(args.label_file).readlines()]

    if args.dataset_type == "voc":
        dataset = VOCDataset(args.dataset, is_test=True)
    elif args.dataset_type == 'open_images':
        dataset = OpenImagesDataset(args.dataset, dataset_type="test")

    true_case_stat, all_gb_boxes, all_difficult_cases = group_annotation_by_class(dataset)
    if args.net == 'vgg16-ssd':
        net = create_vgg_ssd(len(class_names), is_test=True)
    elif args.net == 'mb1-ssd':
        net = create_mobilenetv1_ssd(len(class_names), is_test=True)
    elif args.net == 'mb1-ssd-lite':
        net = create_mobilenetv1_ssd_lite(len(class_names), is_test=True)
    elif args.net == 'sq-ssd-lite':
        net = create_squeezenet_ssd_lite(len(class_names), is_test=True)
    elif args.net == 'mb2-ssd-lite':
        net = create_mobilenetv2_ssd_lite(len(class_names), width_mult=args.mb2_width_mult, is_test=True)
    else:
        logging.fatal("The net type is wrong. It should be one of vgg16-ssd, mb1-ssd and mb1-ssd-lite.")
        parser.print_help(sys.stderr)
        sys.exit(1)  

    timer.start("Load Model")
    net.load(args.trained_model)
    # net.load_state_dict(torch.load(args.trained_model))

    net.error = args.error_rate
    net.num_duplication = args.num_duplication
    net.run_original = args.run_original
    net.duplicated = args.duplicated

    # stored_weights = torch.load("models/mobilenet-v1-ssd-mp-0_675.pth")
    # curr_weights = torch.load(args.trained_model)
    # for w in stored_weights:
    #     print(w, (curr_weights[w] - stored_weights[w]).sum())
    # exit()

    net = net.to(DEVICE)

    if not args.duplicated:
        print("Evaluating model without duplication...")
        # net.attention_mode = True
    else:
        print("Evaluating model with duplication...")
        # Change to evaluate the model with attention
        # device = torch.device("cuda")

        num_layer_mp = {1: 64, 2: 128, 3: 256}
        # layer_mp = {1: net.fc1.weight, 2: net.fc3.weight, 3: net.fc5.weight}
        # layer_mp = {1: net.fc1.weight}
        layer_mp = {1: net.conv1_attention.weight}

        if args.ft_type == "attention":
            print("attention:")
            for k in {1}:
                index = torch.arange(num_layer_mp[k]).type(torch.float).to(DEVICE)
                # tmp = torch.sum(layer_mp[k], axis=0)
                tmp = layer_mp[k].sum(3).sum(2).sum(1)
                #print(layer_mp[k].shape)
                final = torch.stack((tmp, index), axis=0)
                final = final.sort(dim=1, descending=True)
                if k == 1:
                    net.duplicate_index1 = final.indices[0]
                    # net.duplicate_index1 = torch.tensor([10,30,19,0,55,59,47,29,62,13,20,35,53,37,28,17,26,33,50,5,57,40,32,34,41,18,12,58,45,52,63,56,27,44,16,6,4,43,60,49,46,48,8,2,1,11,21,36,42,24,31,25,51,3,38,54,15,61,7,9,22,39,23,14])
                    # print(net.duplicate_index1)
                elif k == 2:
                    net.duplicate_index2 = final.indices[0]
                if k == 3:
                    net.duplicate_index3 = final.indices[0]
        elif args.ft_type == "importance":
            print("importance:")
            for k in {1}:
                index = torch.arange(num_layer_mp[k]).type(torch.float).to(DEVICE)
                net_imp = create_mobilenetv1_ssd(len(class_names))
                weights_imp = copy.deepcopy(net.state_dict())
                net_imp.load_state_dict(weights_imp)
                net_imp.is_importance = True
                importance = cal_importance(net_imp)
                net_imp.is_importance = False
                print(len(importance))
                tmp = importance[0].sum(2).sum(1)
                # tmp = importance[layer_id[k]].sum(2).sum(1)
                # print(layer_mp[k].shape)
                final = torch.stack((tmp, index), axis=0)
                final = final.sort(dim=1, descending=True)
                if k == 1:
                    net.duplicate_index1 = final.indices[0]
                    print(net.duplicate_index1)
                elif k == 2:
                    net.duplicate_index2 = final.indices[0]
                if k == 3:
                    net.duplicate_index3 = final.indices[0]
        else:
            print("d2nn:")
            for k in {1}:
                index = torch.arange(num_layer_mp[k]).type(torch.float).to(DEVICE)
                weight_sum, _ = weight_sum_eval(net)
                # tmp = torch.sum(layer_mp[k], axis=0)
                # exit()
                #tmp = torch.mul(weight_sum[0], weight_sum[1])
                # tmp = weight_sum[0] + weight_sum[1]
                # print(tmp.size())
                tmp = weight_sum[0]
                # print(layer_mp[k].shape, tmp.size())
                final = torch.stack((tmp, index), axis=0)
                final = final.sort(dim=1, descending=True)
                if k == 1:
                    net.duplicate_index1 = final.indices[0]
                    print(net.duplicate_index1)
                elif k == 2:
                    net.duplicate_index2 = final.indices[0]
                if k == 3:
                    net.duplicate_index3 = final.indices[0]

        net.attention_mode = True

    print(f'It took {timer.end("Load Model")} seconds to load the model.')
    if args.net == 'vgg16-ssd':
        predictor = create_vgg_ssd_predictor(net, nms_method=args.nms_method, device=DEVICE)
    elif args.net == 'mb1-ssd':
        predictor = create_mobilenetv1_ssd_predictor(net, nms_method=args.nms_method, device=DEVICE)
    elif args.net == 'mb1-ssd-lite':
        predictor = create_mobilenetv1_ssd_lite_predictor(net, nms_method=args.nms_method, device=DEVICE)
    elif args.net == 'sq-ssd-lite':
        predictor = create_squeezenet_ssd_lite_predictor(net,nms_method=args.nms_method, device=DEVICE)
    elif args.net == 'mb2-ssd-lite':
        predictor = create_mobilenetv2_ssd_lite_predictor(net, nms_method=args.nms_method, device=DEVICE)
    else:
        logging.fatal("The net type is wrong. It should be one of vgg16-ssd, mb1-ssd and mb1-ssd-lite.")
        parser.print_help(sys.stderr)
        sys.exit(1)

    results = []
    for i in range(len(dataset)):
        if i % 100 == 0:
            print("process image", i, "of", len(dataset))
        timer.start("Load Image")
        image = dataset.get_image(i)
        # print("Load Image: {:4f} seconds.".format(timer.end("Load Image")))
        timer.start("Predict")
        boxes, labels, probs = predictor.predict(image)
        # print("Prediction: {:4f} seconds.".format(timer.end("Predict")))
        indexes = torch.ones(labels.size(0), 1, dtype=torch.float32) * i
        # print(indexes.is_cuda, labels.is_cuda, probs.is_cuda, boxes.is_cuda)
        results.append(torch.cat([
            indexes.reshape(-1, 1),
            labels.reshape(-1, 1).float(),
            probs.reshape(-1, 1),
            boxes + 1.0  # matlab's indexes start from 1
        ], dim=1))
        if results[-1].size()[1] == 3:
            print(indexes, labels, probs, boxes)
            results.pop()
    # for r in results:
    #     print(r.size())
    results = torch.cat(results)
    for class_index, class_name in enumerate(class_names):
        if class_index == 0: continue  # ignore background
        prediction_path = eval_path / f"det_test_{class_name}.txt"
        with open(prediction_path, "w") as f:
            sub = results[results[:, 1] == class_index, :]
            for i in range(sub.size(0)):
                prob_box = sub[i, 2:].numpy()
                image_id = dataset.ids[int(sub[i, 0])]
                print(
                    image_id + " " + " ".join([str(v) for v in prob_box]),
                    file=f
                )
    aps = []
    print("\n\nAverage Precision Per-class:")
    for class_index, class_name in enumerate(class_names):
        if class_index == 0:
            continue
        prediction_path = eval_path / f"det_test_{class_name}.txt"
        ap = compute_average_precision_per_class(
            true_case_stat[class_index],
            all_gb_boxes[class_index],
            all_difficult_cases[class_index],
            prediction_path,
            args.iou_threshold,
            args.use_2007_metric
        )
        aps.append(ap)
        print(f"{class_name}: {ap}")

    print(f"\nAverage Precision Across All Classes:{sum(aps)/len(aps)}")
    with open("result.txt", 'a') as f:
        f.write(str(args.error_rate) + "|" + str(args.duplicated) + "|" + args.ft_type + "|" + str(np.mean(aps)) + "\n")



