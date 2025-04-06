import numpy as np
import cv2
import torch
from ultralytics.nn import attempt_load_weights
from ultralytics.utils import Path, yaml_load
import random
import string
import onnxruntime
from mmocr.apis import TextRecInferencer





def pt_dataset(im0s, imgsz, device):
    
    h0, w0 = im0s.shape[:2]
    r = min(imgsz / h0, imgsz / w0)  # ratio
    # r = min(r, 1.0)
    h, w = int(round(h0*r)), int(round(w0*r))
    dh, dw = imgsz-h, imgsz-w
    dw, dh = np.mod(dw, 32), np.mod(dh, 32)
    dw /= 2
    dh /= 2
    if (h0, w0) != (h,w):  # resize
        img = cv2.resize(im0s, (w, h), interpolation=cv2.INTER_LINEAR)
    else:
        img = cv2.resize(im0s, (imgsz, imgsz), interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(
        img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114)
    )

    img = preprocess(img, device)
        
    return img


def preprocess(img, device):
    not_tensor = not isinstance(img, torch.Tensor)
    if not_tensor:
        img = np.stack(img)
        img = img[..., ::-1].transpose((2, 0, 1))  # BGR to RGB, BHWC to BCHW, (n, 3, h, w)
        img = np.ascontiguousarray(img)  # contiguous
        img = torch.from_numpy(img)

    img = img.to(device)
    img = img.float()  # uint8 to fp16/32
    if not_tensor:
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
    
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    return img


def dataset(im0s, imgsz, device):
    
    img = cv2.resize(im0s, (imgsz, imgsz), interpolation=cv2.INTER_LINEAR)

    img = preprocess(img, device)
        
    return img



def run_model(model, img, device, imgsz):
    old_img_w = old_img_h = imgsz
    old_img_b = 1
    if device != 'cpu' and (old_img_b != img.shape[0] or old_img_h != img.shape[2] or old_img_w != img.shape[3]):
        old_img_b = img.shape[0]
        old_img_h = img.shape[2]
        old_img_w = img.shape[3]
        for i in range(3):
            model(img)

    
    
def plot_one_box(x, img, color=None, label=None, line_thickness=3):
    # Plots one bounding box on image img
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    image = cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        image = cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        image = cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)
        
    return image



def yolo_load_model(weight, device):
    print('YOLO')
    
    if weight.endswith('pt'):
        model = attempt_load_weights(weight, device=device)
        names = model.module.names if hasattr(model, 'module') else model.names
        
        weight_format = 'pt'
    
    elif weight.endswith('.onnx'):
        import onnxruntime
        providers = ["CUDAExecutionProvider"] if device == 'cuda' else ["CPUExecutionProvider"]
        print(providers)
        model = onnxruntime.InferenceSession(weight, providers=providers)
        metadata = model.get_modelmeta().custom_metadata_map  # metadata
        for k, v in metadata.items():
            if k in {"stride", "batch"}:
                metadata[k] = int(v)
            elif k in {"imgsz", "names", "kpt_shape"} and isinstance(v, str):
                metadata[k] = eval(v)
        names = metadata["names"]
        weight_format = 'onnx'

        
    else:

        import openvino as ov
    
        core = ov.Core()
        print(f"available_devices: {core.available_devices}")
        w = Path(weight)
        if not w.is_file():  # if not *.xml
            w = next(w.glob("*.xml"))  # get *.xml file from *_openvino_model dir
        ov_model = core.read_model(model=str(w), weights=w.with_suffix(".bin"))

        if ov_model.get_parameters()[0].get_layout().empty:
            ov_model.get_parameters()[0].set_layout(ov.Layout("NCHW"))
        
        inference_mode = "LATENCY"
        model = core.compile_model(ov_model, device_name=f"{device.upper()}", config={"PERFORMANCE_HINT": inference_mode})  # AUTO selects best available device
        metadata = w.parent / "metadata.yaml"
        
        weight_format = 'openvino'
    
        if isinstance(metadata, (str, Path)) and Path(metadata).exists():
            metadata = yaml_load(metadata)

        if metadata:
            names = metadata["names"]

    
    return model, names, weight_format



def mmocr_load_model(weight, cfg, device):
    print('MMOCR')
    
    if weight.endswith('pth'):
        model = TextRecInferencer(model=cfg, weights=weight, device=device)
        weight_format = 'pth'
        
    
    elif weight.endswith('onnx'):
        weight_format = 'onnx'
        providers = ["CUDAExecutionProvider"] if device == 'cuda' else ["CPUExecutionProvider"]
        print(providers)
        model = TextRecognition(weight, f'D:\\tanzaia\\tanzania\\lower_english_digits.txt', providers=providers)


        
    else:

        import openvino as ov
    
        core = ov.Core()
        print(f"available_devices: {core.available_devices}")
        w = Path(weight)
        if not w.is_file():  # if not *.xml
            w = next(w.glob("*.xml"))  # get *.xml file from *_openvino_model dir
        ov_model = core.read_model(model=str(w), weights=w.with_suffix(".bin"))

        if ov_model.get_parameters()[0].get_layout().empty:
            ov_model.get_parameters()[0].set_layout(ov.Layout("NCHW"))
        
        inference_mode = "LATENCY"
        model = core.compile_model(ov_model, device_name=f"{device.upper()}", config={"PERFORMANCE_HINT": inference_mode})  # AUTO selects best available device
    
    return model, weight_format



def from_numpy(x, device):
    """
    Convert a numpy array to a tensor.

    Args:
        x (np.ndarray): The array to be converted.

    Returns:
        (torch.Tensor): The converted tensor
    """
    return torch.tensor(x).to(device) if isinstance(x, np.ndarray) else x


def scaling_boxes(img1, boxes, img0):
    w = img0[1] / img1[1]
    h = img0[0] / img1[0]
    boxes[..., 0] = boxes[..., 0] * w
    boxes[..., 1] = boxes[..., 1] * h
    boxes[..., 2] = boxes[..., 2] * w
    boxes[..., 3] = boxes[..., 3] * h
    
    boxes[..., 0] = boxes[..., 0].clamp(0, img0[1])  # x1
    boxes[..., 1] = boxes[..., 1].clamp(0, img0[0])  # y1
    boxes[..., 2] = boxes[..., 2].clamp(0, img0[1])  # x2
    boxes[..., 3] = boxes[..., 3].clamp(0, img0[0])  # y2
    
    return boxes




def align_plate(datas):
    x1_list = []
    y1_list = []
    x2_list = []
    y2_list = []
    t_text = ''
    
    result = []
    
    if datas != []:
        align_data = sorted(datas, key=lambda x: x[0])
        for i, data in enumerate(align_data):
            x1, y1, x2, y2, text, _ = data
            if len(align_data) == 1:
                result.append([x1, y1, x2, y2, text])
            else:
                if len(text) <=5:
                    x1_list.append(x1)
                    y1_list.append(y1)
                    x2_list.append(x2)
                    y2_list.append(y2)
                    t_text += text

                    if len(t_text) > 5 or len(align_data) == i + 1:
                        x1_ = min(x1_list)
                        y1_ = min(y1_list)
                        x2_ = max(x2_list)
                        y2_ = max(y2_list)
                        result.append([x1_, y1_, x2_, y2_, t_text])
                        x1_list.clear()
                        y1_list.clear()
                        x2_list.clear()
                        y2_list.clear()
                        t_text = ''
                else:
                    result.append([x1, y1, x2, y2, text])

        return result
    else:
        pass


def align_container(datas):
    x1_list = []
    y1_list = []
    x2_list = []
    y2_list = []
    t_text = ''
    
    results = []

    if datas != []:
        align_data = sorted(datas,key=lambda x: x[1])
        i = 0
        for j, data in enumerate(align_data):
            x1, y1, x2, y2, text, cls = data
            if cls == 0:
                if len(text) == 11:
                    results.append([x1, y1, x2, y2, text])
                elif len(text) <= 5 or i == 1:
                    i += 1
                    x1_list.append(x1)
                    y1_list.append(y1)
                    x2_list.append(x2)
                    y2_list.append(y2)
                    t_text += text

                    if i % 2 == 0:
                        x1_ = min(x1_list)
                        y1_ = min(y1_list)
                        x2_ = max(x2_list)
                        y2_ = max(y2_list)
                        results.append([x1_, y1_, x2_, y2_, t_text])
                        x1_list.clear()
                        y1_list.clear()
                        x2_list.clear()
                        y2_list.clear()
                        t_text = ''
                        i = 0
                else:
                    results.append([x1, y1, x2, y2, text])

            
            elif cls == 1:
                results.append([x1, y1, x2, y2, text])
        
        return results
    else:
        pass
                        


def plate_modification(plate):
    num = 0
    char = 0

    word = plate[-1]

    if (len(word) > 9) or (len(word) < 6):
        return None
    
    for i in range(len(word)):
        if word[i] in string.ascii_uppercase:
            char += 1
        elif word[i] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']:
            num += 1

    if (num == 0) or (char == 0):
        return None
    else:
        return plate




class TextRecognition:
    def __init__(self, onnx_model, dict_path, providers):
        self.providers = providers
        self.dict_chars, self.EOS_IDX, self.UKN_IDX = self.read_character_dict(
            dict_path)
        self.session = onnxruntime.InferenceSession(
            onnx_model, providers=self.providers)


    def read_character_dict(self, dict_path):

        with open(dict_path, "r", encoding="utf-8") as f:
            texts = f.readlines()
            dict_chars = [text.strip() for text in texts]

        # Update dict
        dict_chars = dict_chars + ['<BOS/EOS>', '<UKN>']
        eos_idx = len(dict_chars) - 2
        ukn_idx = len(dict_chars) - 1
        return dict_chars, eos_idx, ukn_idx

    def preprocess(self, img):
        target_height, target_width = 64, 256
        resized_img = cv2.resize(img, (target_width, target_height))
        padding_im = resized_img.astype(np.float32)

        # NHWC to NCHW
        x = np.array([padding_im])
        x = torch.Tensor(x)
        x = x.permute(0, 3, 1, 2)

        # Channel conversion
        x = x[:, [2, 1, 0], ...]

        # Normalize
        mean = [127.5, 127.5, 127.5, ]
        mean = torch.tensor(mean).view(-1, 1, 1)
        std = [127.5, 127.5, 127.5, ]
        std = torch.tensor(std).view(-1, 1, 1)
        x = (x - mean) / std

        return x

    def infer(self, x):
        outputs = self.session.run(None, {'input': x.numpy()})
        return torch.from_numpy(outputs[0])

    def postprocess(self, pred):
        max_value, max_idx = torch.max(pred, -1)
        texts = []
        scores = []
        batch_num = pred.shape[0]
        for i in range(batch_num):
            text = ""
            score = 0
            prev_idx = self.EOS_IDX
            for output_score, output_idx in zip(max_value[i], max_idx[i]):
                if output_idx not in (prev_idx, self.EOS_IDX, self.UKN_IDX) and output_score > 0.5:
                    text += self.dict_chars[output_idx]
                    if self.dict_chars[output_idx] == '':
                        text += ' '
                prev_idx = output_idx
            text = text.rstrip()
            score += output_score
            texts.append(text)
            scores.append(output_score)

        return texts, scores

    def recognition(self, image):
        x = self.preprocess(image)
        pred = self.infer(x)
        texts, scores = self.postprocess(pred)
        return texts[0], scores