import numpy as np
import scipy
from collections import defaultdict
from function import plot_one_box
import os
import cv2
from tabulate import tabulate



def read_label(source, file, file_format, path, names, colors, n_file):
    im = cv2.imread(path)
    imgsize = im.shape[:2]
    label_file = file.replace(file_format, 'txt')
    label_source = source.replace('images', 'labels')

    label_path = label_source + '\\' + label_file
    
    label_list = []
    answer = []
    if os.path.isfile(label_path):
        with open(label_path) as ff:
            labels = ff.readlines()
            if labels != []:
                for lab in labels:
                    lab = lab.split(' ')
                    gt = ground_truth(lab[1:], imgsize)
                    answer.append([lab[0], gt[0], gt[1], gt[2], gt[3]])
                    if lab[0] not in label_list:
                        label_list.append(lab[0])
                    gt_label = f'{names[int(lab[0])]}'
                    gt_result = plot_one_box(gt, im, label=gt_label, color=colors[int(lab[0])], line_thickness=5)
            else:
                gt_result = im

    for i in label_list:
        n_file[f'exist_{i}'] += 1


    return answer, gt_result, n_file




def bbox_iou(boxA, boxB):
    # https://www.pyimagesearch.com/2016/11/07/intersection-over-union-iou-for-object-detection/
    # ^^ corrected.

    # Determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interW = xB - xA + 1
    interH = yB - yA + 1

    # Correction: reject non-overlapping boxes
    if interW <=0 or interH <=0 :
        iou = 0
    else:
        interArea = interW * interH
        boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
        boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
        iou = interArea / float(boxAArea + boxBArea - interArea)

    return iou

def xywh2xyxy(gt):
    
    x, y, w, h = gt
    
    x1 = x - w/2
    y1 = y - h/2
    x2 = x + w/2
    y2 = y+ h/2
    
    return x1, y1, x2, y2


def ground_truth(ground, imgsz):
    
    img_h, img_w = imgsz
    x, y, w, h = ground
    x_ = float(x) * float(img_w)
    y_ = float(y)* float(img_h)
    w_ = float(w) * float(img_w)
    h_ = float(h) * float(img_h)
    
    cord = x_, y_, w_, h_
    
    xmin, ymin, xmax, ymax = xywh2xyxy(cord)

    gt= [int(xmin), int(ymin), int(xmax), int(ymax)]

    return gt


def matching(pred, gt, IOU_THRESH=0.5):
    #### website: https://gist.github.com/AruniRC/c629c2df0e68e23aff7dcaeef87c72d4
        
    n_true = len(gt)
    n_pred = len(pred)
    # MAX_DIST = 1.0
    MIN_IOU = 0.0

    # NUM_GT x NUM_PRED
    iou_matrix = np.zeros((n_true, n_pred))
    for i in range(n_true):
        for j in range(n_pred):
            iou_matrix[i, j] = bbox_iou(gt[i][1:], pred[j][:-2])

    if n_pred > n_true:
    # there are more predictions than ground-truth - add dummy rows
        diff = n_pred - n_true
        iou_matrix = np.concatenate( (iou_matrix, 
                                    np.full((diff, n_pred), MIN_IOU)), 
                                  axis=0)

    if n_true > n_pred:
    # more ground-truth than predictions - add dummy columns
        diff = n_true - n_pred
        iou_matrix = np.concatenate( (iou_matrix, 
                                    np.full((n_true, diff), MIN_IOU)), 
                                  axis=1)
        
    idxs_true, idxs_pred = scipy.optimize.linear_sum_assignment(1 - iou_matrix)
    
    
    if (not idxs_true.size) or (not idxs_pred.size):
        ious = np.array([])
    else:
        ious = iou_matrix[idxs_true, idxs_pred]

    sel_pred = idxs_pred<n_pred
    idx_pred_actual = idxs_pred[sel_pred] 
    idx_gt_actual = idxs_true[sel_pred]
    ious_actual = iou_matrix[idx_gt_actual, idx_pred_actual]
    sel_valid = (ious_actual > IOU_THRESH)
    label = sel_valid.astype(int)
    
    # return idx_gt_actual[sel_valid], idx_pred_actual[sel_valid], ious_actual[sel_valid], label 
    # return idx_gt_actual[sel_valid], idx_pred_actual[sel_valid] 
    
    ### Modified from here
    gt_actual = idx_gt_actual[sel_valid]
    pred_actual = idx_pred_actual[sel_valid]
    
    ground_truth = []
    predicted = []

    a = max(n_true, n_pred)
    b = min(n_true, n_pred)
    
    for i in range(b):
        ground_truth.append(gt[gt_actual[i]])
        predicted.append(pred[pred_actual[i]])
            
    if n_true != n_pred:
        if a == n_true:
            for j in range(n_true):
                if j not in gt_actual.tolist():
                    print('gt')
                    print(j)
                    print(type(j))
                    print(type(gt_actual.tolist()[0]))
                    ground_truth.append(gt[j])
        elif a == n_pred:
            for j in range(n_pred):
                if j not in pred_actual.tolist():
                    print('pred')
                    print(j)
                    print(type(j))
                    print(type(pred_actual.tolist()[0]))
                    predicted.append(pred[j])
        
        

    return ground_truth, predicted





def calculate_metrics(ground_truths, predictions, classes, n_file, iou_threshold=0.5):
    """
    Calculate precision, recall, and accuracy for multi-object detection.
    
    Parameters:
    - ground_truths: List of tuples (label, bounding_box)
    - predictions: List of tuples (label, bounding_box, score)
    - iou_threshold: IoU threshold to determine a true positive
    
    Returns:
    - precision, recall, accuracy

    reference: https://www.evidentlyai.com/classification-metrics/multi-class-metrics#:~:text=To%20calculate%20the%20precision%2C%20divide,True%20Positives%20and%20False%20Negatives.
     https://medium.com/synthesio-engineering/precision-accuracy-and-f1-score-for-multi-label-classification-34ac6bdfb404
    """
    
    tp = defaultdict(int)
    fp = defaultdict(int)
    fn = defaultdict(int)
    gt = defaultdict(int)
    
    exist_label = []

    for pred_idx, (*pred_box, _, pred_label) in enumerate(predictions):
        best_iou = 0

        if pred_label not in exist_label:
            exist_label.append(pred_label)
            
        for gt_label, *gt_box in ground_truths:
            if gt_label not in exist_label:
                exist_label.append(gt_label)

            if pred_idx == 0:
                gt[gt_label] += 1
                n_file[gt_label] += 1
            
            if pred_label == gt_label:
                current_iou = bbox_iou(pred_box, gt_box)
                if current_iou > best_iou:
                    best_iou = current_iou


        if best_iou >= iou_threshold:
            tp[pred_label] += 1
        else:
            fp[pred_label] += 1

    for lab in exist_label:
        n_file[f'existing_{lab}'] += 1


    fn = {label: gt[label] - tp[label] for label in classes}

    precision, recall, accuracy, F1_score = metrics(tp, fp, fn, classes)

    return precision, recall, accuracy, F1_score, n_file


def metrics(tp, fp, fn, classes):
    precision = {}
    recall = {}
    accuracy = {}
    F1_Score = {}


    for label in classes:
        tp_label = tp[label]
        fp_label = fp[label]
        fn_label = fn[label]
        
        precision[label] = tp_label / (tp_label + fp_label) if (tp_label + fp_label) > 0 else 0
        recall[label] = tp_label / (tp_label + fn_label) if (tp_label + fn_label) > 0 else 0
        accuracy[label] = tp_label / (tp_label + fp_label + fn_label) if (tp_label + fp_label + fn_label) > 0 else 0
        F1_Score[label] = 2 * (precision[label] * recall[label]) / (precision[label] + recall[label]) if (precision[label] + recall[label]) > 0 else 0

    return precision, recall, accuracy, F1_Score


def make_table(precision, recall, accuracy, f1_score, names, n_file):
    avg_precision = 0
    avg_recall = 0
    avg_accuracy = 0
    avg_f1_score = 0
    instance = 0
    n = []


    
    for lab in names.keys():
        lab = str(lab)
        avg_precision += precision[lab]
        avg_recall += recall[lab]
        avg_accuracy += accuracy[lab]
        avg_f1_score += f1_score[lab]
        instance += n_file[lab]
        if n_file[lab] != 0:
            n.append(lab)


    # print(n_file)
    avg = avg_precision/len(n), avg_recall/len(n), avg_accuracy/len(n), avg_f1_score/len(n)

    header = [
        'class', 'images', 'Instance', 'Precision', 'Recall', 'Accuracy', 'F1 Score']

    if len(n) > 1:
        table = [
            ['all', n_file['total'], instance, f'{avg[0]}', f'{avg[1]}', f'{avg[2]}', f'{avg[3]}']
            ]
        
        for i in n:
            line = [f'{names[int(i)]}', n_file[f'exist_{i}'], f'{n_file[i]}', f'{precision[i]}', f'{recall[i]}', f'{accuracy[i]}', f'{f1_score[i]}']
            table.append(line)

    elif len(n) == 1:
        table = [
            ['all', n_file['total'], instance, f'{avg[0]}', f'{avg[1]}', f'{avg[2]}', f'{avg[3]}'],
            [f'{names[0]}', n_file['exist_0'], n_file['0'], precision['0'], recall['0'], accuracy['0'], f1_score['0']]
        ]

    
    # Print the table
    print(tabulate(table, header, tablefmt='grid'))
    print('\n')
