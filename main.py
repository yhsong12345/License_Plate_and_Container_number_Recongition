import torch
import cv2
import os
from function import (yolo_load_model, mmocr_load_model, pt_dataset, run_model, 
                      dataset,from_numpy, plot_one_box,
                      align_container, align_plate, scaling_boxes, plate_modification)
from ultralytics.utils.ops import scale_boxes, non_max_suppression
import argparse
import time




def main(args):

    det_weight = args.det_weight
    recog_weight = args.recog_weight
    recog_model_cfg = args.recog_model

    source = args.source
    result = args.result
    imgsz = args.imgsz
    device = args.device
    conf_thres = args.conf_thres
    iou_thres = args.iou_thres
    rec_per_img = []

    det_model, names, weight_format = yolo_load_model(det_weight, device)
    print(names)

    recog_model, rec_weight_format = mmocr_load_model(recog_weight, recog_model_cfg, device)

    print(f"Computation device: {device}\n")

    
    files = os.listdir(source)
    n = len(files)
    for file in files:
        print(file)
        file = file.replace('\n', '')
        path = source + '\\' + file

        #Preprocessing
        im0 = cv2.imread(path)

        if weight_format == 'pt':
            img = pt_dataset(im0, imgsz, device)
            run_model(det_model, img, device, imgsz)
            ### Inference
            with torch.no_grad():
                pred = det_model(img)
            
            if isinstance(pred, dict):
                pred = pred['one2one']

            if isinstance(pred, (list, tuple)):
                pred = pred[0]

        else:
            t1 = time.time()
            img = dataset(im0, imgsz, device)
            img = img.cpu().numpy()
            t2 = time.time()
            if weight_format == 'openvino':
                ### Inference
                pred = list(det_model(img).values())


            elif weight_format == 'onnx':
                output_names = [x.name for x in det_model.get_outputs()]
                ### Inference
                t3 = time.time()
                pred = det_model.run(output_names, {det_model.get_inputs()[0].name: img})
                t4 = time.time()

            t5 = time.time()
            if isinstance(pred, (list, tuple)):
                pred = from_numpy(pred[0], device) if len(pred) == 1 else [from_numpy(x, device) for x in pred]
            else:
                pred = from_numpy(pred, device)
        

        
        if pred.shape[-1] == 6:
            mask = pred[..., 4] > conf_thres
            pred = [p[mask[idx]] for idx, p in enumerate(pred)]

        else:
            pred = non_max_suppression(pred, conf_thres, iou_thres)
        
        t6 = time.time()
        # print(f"preprocess: {(t2-t1) * 1000}ms")
        # print(f"inference: {(t4-t3) * 1000}ms")
        # print(f"postprocess: {(t6-t5) * 1000}ms")

        results = []
        text_score = 0
        for i, det in enumerate(pred):
            if weight_format == 'pt':
                det[:, :4] = scale_boxes(img.shape[2:], det[:, :4], im0.shape)
            else:
                det[:, :4] = scaling_boxes(img.shape[2:], det[:, :4], im0.shape)

            plates = []
            containers = []
            m = 0
            rec_per_single = 0
            for *xyxy, _, cls in reversed(det):
                m += 1
                x1, y1, x2, y2 = list(map(int, xyxy))
                cls = str(int(cls))
                image = im0[int(y1): int(y2), int(x1): int(x2)]
                if cls in ['0', '1']:
                    h, w = image.shape[:2]
                    if h > w:
                        image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
                
                # check time (inference)
                if rec_weight_format == 'pth':
                    text_pred = recog_model(image)
                    text = text_pred['predictions'][0]['text']
                    text_score = text_pred['predictions'][0]['scores']
                    print(text_score)

                elif rec_weight_format == 'onnx':
                    t7 = time.time()
                    text, text_score = recog_model.recognition(image)
                    t8 = time.time()
                    rec_per_single += (t8 - t7)
                    rec_per_single /= m
                    rec_per_img.append(rec_per_single)
                    print(text_score)
                

                if cls in ['0', '1']:
                    containers.append([x1, y1, x2, y2, text.upper(), int(cls)])
                elif cls == '2':
                    plates.append([x1, y1, x2, y2, text.upper(), int(cls)])
            
            print(plates)
            print(containers)
            if containers != []:
                for container in align_container(containers):
                    results.append(container)

            if plates != []:
                for plate in align_plate(plates):
                    results.append(plate_modification(plate))
        
        for result in results:
            if result == None:
                pass
            else:
                *xyxy, text = result
                label = f'{text}'
                gt_result_1 = plot_one_box(xyxy, im0, label=label, color=[255, 0, 0], line_thickness=5)

        gt_result_1 = cv2.resize(gt_result_1, (2000, 1500))
        cv2.imshow('output', gt_result_1)
        cv2.waitKey(0)
    # print(sum(rec_per_img) / n)
    # print(sum(rec_per_img) / len(rec_per_img))
    



                
    
if __name__ == '__main__':
    torch.backends.cudnn.benchmark = True
    parser = argparse.ArgumentParser(
        description="Distributed Arcface Training in Pytorch")
    parser.add_argument('--imgsz', type=int, default=640, help='image size')
    parser.add_argument('--det-weight', type=str, 
                        default=r"D:\\tanzaia\\tanzania\\model\\240909_LP_Conatiner_Detection_v8.onnx",
                        help='weight of detection det_model')
    parser.add_argument('--recog-weight', type=str, 
                        default=r"D:\\tanzaia\\tanzania\\model\\svtr.onnx",
                        help='weight of recognition det_model')
    parser.add_argument('--recog-model', type=str, 
                        default=r"D:\\tanzaia\\mmocr\\configs\\textrecog\\svtr\\svtr-tiny_20e_tanzania.py",
                        help='weight of recognition det_model')
    parser.add_argument('--source', type=str, 
                        help='source directory', 
                        default=r"E:\\tanzania\\test\\images\\")
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--conf-thres', type=float, default=0.5, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.1, help='iou threshold')
    parser.add_argument('--valid', type=bool, default=False, help='Calculate Metrics')
    parser.add_argument('--plot', type=bool, default=False, help='plot results')
    parser.add_argument('--result', type=bool, default=True, help='show results')
    args = parser.parse_args()
    main(args)