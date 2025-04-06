import torch
import cv2
import os
from function import (load_model, pt_dataset, run_model, 
                      dataset,from_numpy, plot_one_box,
                      align_plate, align_container, scaling_boxes)
import random
from ultralytics.utils.ops import scale_boxes, non_max_suppression
import argparse
import time
from mmocr.apis import TextRecInferencer
import matplotlib.pyplot as plt




def main(args):

    det_weight = args.det_weight
    recog_weight = args.recog_weight
    recog_model_dir = args.recog_model

    source = args.source
    result = args.result
    plot = args.plot

    imgsz = args.imgsz
    device = args.device
    conf_thres = args.conf_thres
    iou_thres = args.iou_thres

    print(source)
    print(f"Computation device: {device}\n")


    cap = cv2.VideoCapture(source)
    success, im0 = cap.read()

    ### This is for saving the result
    # frame_width = int(cap.get(3)) 
    # frame_height = int(cap.get(4)) 
   
    # size = (frame_width, frame_height)
    # save_result = cv2.VideoWriter(f"{file_name}.mp4", cv2.VideoWriter_fourcc(*'mp4v'), 10, size)


    det_model, names, weight_format = load_model(det_weight, device)
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]
    print(names)

    recog_model = TextRecInferencer(model=recog_model_dir, weights=recog_weight, device=device)


    print(f"Computation device: {device}\n")


    while success:
        if weight_format == 'pt':
            img = pt_dataset(im0, imgsz, device)
            run_model(det_model, img, device, imgsz)
            ### Inference
            with torch.no_grad():
                det_inf_start = time.time()
                pred = det_model(img)
                det_inf_end = time.time()
            
            if isinstance(pred, dict):
                pred = pred['one2one']

            if isinstance(pred, (list, tuple)):
                pred = pred[0]

        else:
            img = dataset(im0, imgsz, device)
            img = img.cpu().numpy()

            if weight_format == 'openvino':
                ### Inference
                det_inf_start = time.time()
                pred = list(det_model(img).values())
                det_inf_end = time.time()


            elif weight_format == 'onnx':
                output_names = [x.name for x in det_model.get_outputs()]
                ### Inference
                det_inf_start = time.time()
                pred = det_model.run(output_names, {det_model.get_inputs()[0].name: img})
                det_inf_end = time.time()

            if isinstance(pred, (list, tuple)):
                pred = from_numpy(pred[0], device) if len(pred) == 1 else [from_numpy(x, device) for x in pred]
            else:
                pred = from_numpy(pred, device)


        print(f'detection time = {det_inf_end - det_inf_start}')
        if pred.shape[-1] == 6:
            mask = pred[..., 4] > conf_thres
            pred = [p[mask[idx]] for idx, p in enumerate(pred)]

        else:
            pred = non_max_suppression(pred, conf_thres, iou_thres)

        # print(pred)
        results = []
        for i, det in enumerate(pred):
            if weight_format == 'pt':
                det[:, :4] = scale_boxes(img.shape[2:], det[:, :4], im0.shape)
            else:
                det[:, :4] = scaling_boxes(img.shape[2:], det[:, :4], im0.shape)
            # print(det)
            plate = []
            container = []
            for *xyxy, _, cls in reversed(det):
                x1, y1, x2, y2 = list(map(int, xyxy))
                cls = str(int(cls))
                image = im0[int(y1): int(y2), int(x1): int(x2)]
                # plt.imshow(image)
                # plt.show()
                if cls in ['0', '1']:
                    h, w = image.shape[:2]
                    if h > w:
                        image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)

                recog_inf_start = time.time()
                text_pred = recog_model(image)
                recog_inf_end = time.time()
                print(f'recog time = {recog_inf_end - recog_inf_start}')

                text_pred = text_pred['predictions'][0]['text']

                if cls in ['0', '1']:
                    container.append([x1, y1, x2, y2, text_pred.upper(), int(cls)])
                elif cls == '2':
                    plate.append([x1, y1, x2, y2, text_pred.upper(), int(cls)])
                
            if container != []:
                for i in align_container(container):
                    results.append(i)

            if plate != []:
                for i in align_plate(plate):
                    results.append(i)

        
        # print(results)
        for result in results:
            # result = result[0]
            if result == None:
                pass
            else:
                *xyxy, text = result
                label = f'{text}'
                gt_result_1 = plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=5)

        cv2.imshow('output', gt_result_1)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        success, im0 = cap.read()
    cap.release()




    
if __name__ == '__main__':
    torch.backends.cudnn.benchmark = True
    parser = argparse.ArgumentParser(
        description="Distributed Arcface Training in Pytorch")
    parser.add_argument('--imgsz', type=int, default=640, help='image size')
    parser.add_argument('--det-weight', type=str, 
                        default=r"D:\\tanzaia\\tanzania\\model\\240909_LP_Conatiner_Detection_v8.pt",
                        help='weight of detection det_model')
    parser.add_argument('--recog-weight', type=str, 
                        default=r"D:\\tanzaia\\tanzania\\model\\svtr.pth",
                        help='weight of recognition det_model')
    parser.add_argument('--recog-model', type=str, 
                        default=r"D:\\tanzaia\\mmocr\\configs\\textrecog\\svtr\\svtr-tiny_20e_tanzania.py",
                        help='weight of recognition det_model')
    parser.add_argument('--source', type=str, 
                        help='source directory',
                        default=r"D:\\안전모demo1\\ultralytics\\test_img\\video\\KakaoTalk_20231122_165526069.mp4")
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--conf-thres', type=float, default=0.2, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.1, help='iou threshold')
    parser.add_argument('--valid', type=bool, default=False, help='Calculate Metrics')
    parser.add_argument('--plot', type=bool, default=False, help='plot results')
    parser.add_argument('--result', type=bool, default=True, help='show results')
    args = parser.parse_args()
    main(args)
