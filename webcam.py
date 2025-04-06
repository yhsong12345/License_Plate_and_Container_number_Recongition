import torch
import cv2
from function import (load_model, pt_dataset, 
                      run_model, dataset,from_numpy, 
                      plot_one_box, scaling_boxes)
import random
import matplotlib.pyplot as plt
from ultralytics.utils.ops import scale_boxes, non_max_suppression, v10postprocess, xywh2xyxy
import argparse


def main(args):


    weight = args.weight

    result = args.result
    plot = args.plot

    imgsz = args.imgsz
    device = args.device
    conf_thres = args.conf_thres
    iou_thres = args.iou_thres


    model, names, weight_format = load_model(weight, device)
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]
    print(names)

    print(f"Computation device: {device}\n")

    file_name = weight.split('\\')[-1]
    file_name = file_name.split('.')[0]

    cap = cv2.VideoCapture(0)
    success, im0 = cap.read()
    
    frame_width = im0.shape[1]
    frame_height = im0.shape[0]
   
    size = (frame_width, frame_height) 
    save_result = cv2.VideoWriter(f"{file_name}.mp4", cv2.VideoWriter_fourcc(*'mp4v'), 30, size)

    while success:
        if weight_format == 'pt':
            img = pt_dataset(im0, imgsz, device)
            pred = run_model(model, img, device, imgsz)
            ### Inference
            with torch.no_grad():
                pred = model(img)
            
            if isinstance(pred, dict):
                pred = pred['one2one']

            if isinstance(pred, (list, tuple)):
                pred = pred[0]

            if pred.shape[-1] == 6:
                pass
            else:
                pred = pred.transpose(-1, -2)
                bboxes, scores, labels = v10postprocess(pred, max_det=5, nc = pred.shape[-1]-4)
                bboxes = xywh2xyxy(bboxes)
                pred = torch.cat([bboxes, scores.unsqueeze(-1), labels.unsqueeze(-1)], dim=-1)

                mask = pred[..., 4] > conf_thres

                pred = [p[mask[idx]] for idx, p in enumerate(pred)]



        else:
            img = dataset(im0, imgsz, device)
            img = img.cpu().numpy()
            if weight_format == 'openvino':

                ### Inference
                pred = list(model(img).values())


            elif weight_format == 'onnx':
                output_names = [x.name for x in model.get_outputs()]
                ### Inference
                pred = model.run(output_names, {model.get_inputs()[0].name: img})


            if isinstance(pred, (list, tuple)):
                pred = from_numpy(pred[0], device) if len(pred) == 1 else [from_numpy(x, device) for x in pred]
            else:
                pred = from_numpy(pred, device)


        #### Postprocessing
        # pred = non_max_suppression(pred, conf_thres, iou_thres)


        for i, det in enumerate(pred):
            detected_obj = []
            detected_conf = []
            if len(det):
                if weight_format == 'pt':
                    det[:, :4] = scale_boxes(img.shape[2:], det[:, :4], im0.shape)
                else:
                    det[:, :4] = scaling_boxes(img.shape[2:], det[:, :4], im0.shape)
                for *xyxy, conf, cls in reversed(det):
                    conf = conf.detach().cpu().numpy()
                    detected_conf.append(float(conf))
                    cls = cls.detach().cpu().numpy()
                    detected_obj.append(str(int(cls)))
                    label = f'{names[int(cls)]} {conf:.2f}'
                    gt_result = plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3)
                    gt_result = cv2.resize(gt_result, (720, 1280))  #### (Width, Height) original = 720, 1280
                    save_result.write(gt_result)
                    cv2.imshow('output', gt_result)


        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


        if result:
            detected_list = [(obj, conf) for obj, conf in zip(detected_obj, detected_conf)]

            Safety_Helmet = False
            Safety_vest = False
            box = True

            count_0 = 0
            count_1 = 0
            count_2 = 0
            count_3 = 0
            count_4 = 0

            for obj in detected_obj:
                if obj == '0':
                    count_0 += 1

                elif obj == '1':
                    count_1 += 1

                elif obj == '2':
                    count_2 += 1

                elif obj == '3':
                    count_3 += 1

                elif obj == '4':
                    count_4 += 1


            if count_0 > 0:
                if count_2 == 0:
                    Safety_Helmet = True

            if count_1 > 0:
                if count_3 == 0:
                    Safety_vest = True

            if count_4 == 0:
                box = False



            ### Result
            for i in range(len(detected_list)):
                    print(f'{names[int(detected_list[i][0])]} is detected with {detected_list[i][1]:.2f} accuracy')

            # print('Final Result')
            # if (Safety_Helmet == True) and (Safety_vest == True) and (box == False):
            #     print('You are good to go')
            #     for i in range(len(detected_list)):
            #         print(f'{names[int(detected_list[i][0])]} is detected with {detected_list[i][1]:.2f} accuracy')

            # else:
            #     print('Access denied, check your PPE') #PPE = Personal protective equipment
            #     for i in range(len(detected_list)):
            #         if detected_list[i][0] in ['2', '3', '4']:
            #             print(f'{names[int(detected_list[i][0])]} is detected with {detected_list[i][1]:.2f} accuracy')

            print('-----------------------------------------------------------------------------------------')
        success, im0 = cap.read()
    cap.release()


                
    
if __name__ == '__main__':
    torch.backends.cudnn.benchmark = True
    parser = argparse.ArgumentParser(
        description="Distributed Arcface Training in Pytorch")
    parser.add_argument('--imgsz', type=int, default=640, help='image size')
    parser.add_argument('--weight', type=str, 
                        default=r'D:\\yolov10-main\\models\\240724_Construction_Detection_v10.pt', 
                        help='weight of model')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--conf-thres', type=float, default=0.7, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='iou threshold')
    parser.add_argument('--plot', type=bool, default=True, help='Calculate Metrics')
    parser.add_argument('--result', type=bool, default=True, help='show results')
    args = parser.parse_args()
    main(args)


