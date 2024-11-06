import warnings

warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')
import torch, yaml, cv2, os, shutil, sys
import numpy as np

np.random.seed(0)
import matplotlib.pyplot as plt
from tqdm import trange
from PIL import Image
from ultralytics.nn.tasks import attempt_load_weights
from ultralytics.utils.torch_utils import intersect_dicts
from ultralytics.utils.ops import xywh2xyxy, non_max_suppression
from pytorch_grad_cam import GradCAMPlusPlus, GradCAM, XGradCAM, EigenCAM, HiResCAM, LayerCAM, RandomCAM, EigenGradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image, scale_cam_image
# from pytorch_grad_cam.activations_and_gradients import ActivationsAndGradients


def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return im, ratio, (dw, dh)


class ActivationsAndGradients:
    """ Class for extracting activations and
    registering gradients from targetted intermediate layers """

    def __init__(self, model, target_layers, reshape_transform):
        self.model = model
        self.gradients = []
        self.activations = []
        self.reshape_transform = reshape_transform
        self.handles = []
        for target_layer in target_layers:
            self.handles.append(
                target_layer.register_forward_hook(self.save_activation))
            # Because of https://github.com/pytorch/pytorch/issues/61519,
            # we don't use backward hook to record gradients.
            self.handles.append(
                target_layer.register_forward_hook(self.save_gradient))

    def save_activation(self, module, input, output):
        activation = output

        if self.reshape_transform is not None:
            activation = self.reshape_transform(activation)
        self.activations.append(activation.cpu().detach())

    def save_gradient(self, module, input, output):
        if not hasattr(output, "requires_grad") or not output.requires_grad:
            # You can only register hooks on tensor requires grad.
            return

        # Gradients are computed in reverse order
        def _store_grad(grad):
            if self.reshape_transform is not None:
                grad = self.reshape_transform(grad)
            self.gradients = [grad.cpu().detach()] + self.gradients

        output.register_hook(_store_grad)

    def post_process(self, result):
        logits_ = result[:, 4:]
        boxes_ = result[:, :4]
        sorted, indices = torch.sort(logits_.max(1)[0], descending=True)
        return torch.transpose(logits_[0], dim0=0, dim1=1)[indices[0]], torch.transpose(boxes_[0], dim0=0, dim1=1)[
            indices[0]], xywh2xyxy(torch.transpose(boxes_[0], dim0=0, dim1=1)[indices[0]]).cpu().detach().numpy()

    def __call__(self, x):
        self.gradients = []
        self.activations = []
        x_ir = tensor_ir
        model_output = self.model(x, x_ir)
        post_result, pre_post_boxes, post_boxes = self.post_process(model_output['one2one'][0])
        return [[post_result, pre_post_boxes]]

    def release(self):
        for handle in self.handles:
            handle.remove()


class yolov8_target(torch.nn.Module):
    def __init__(self, ouput_type, conf, ratio) -> None:
        super().__init__()
        self.ouput_type = ouput_type
        self.conf = conf
        self.ratio = ratio

    def forward(self, data):
        post_result, pre_post_boxes = data
        result = []
        for i in trange(int(post_result.size(0) * self.ratio)):
            if float(post_result[i].max()) < self.conf:
                break
            if self.ouput_type == 'class' or self.ouput_type == 'all':
                result.append(post_result[i].max())
            elif self.ouput_type == 'box' or self.ouput_type == 'all':
                for j in range(4):
                    result.append(pre_post_boxes[i, j])
        return sum(result)


class yolov8_heatmap:
    def __init__(self, weight, device, method, layer, backward_type, conf_threshold, ratio, show_box, renormalize):
        self.layer = layer
        device = torch.device(device)
        ckpt = torch.load(weight)
        model_names = ckpt['model'].names
        model = attempt_load_weights(weight, device)
        model.info()
        for p in model.parameters():
            p.requires_grad_(True)
        model.eval()

        target = yolov8_target(backward_type, conf_threshold, ratio)
        target_layers = [model.model[l] for l in layer]
        method = eval(method)(model, target_layers)
        method.activations_and_grads = ActivationsAndGradients(model, target_layers, None)

        colors = np.random.uniform(0, 255, size=(len(model_names), 3)).astype(np.int32)
        self.__dict__.update(locals())

    def post_process(self, result):
        result = non_max_suppression(result, conf_thres=self.conf_threshold, iou_thres=0.65)[0]
        return result

    def draw_detections(self, box, color, name, img):
        xmin, ymin, xmax, ymax = list(map(int, list(box)))
        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), tuple(int(x) for x in color), 2)
        cv2.putText(img, str(name), (xmin, ymin - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, tuple(int(x) for x in color), 2,
                    lineType=cv2.LINE_AA)
        return img

    def renormalize_cam_in_bounding_boxes(self, boxes, image_float_np, grayscale_cam):
        """Normalize the CAM to be in the range [0, 1]
        inside every bounding boxes, and zero outside of the bounding boxes. """
        renormalized_cam = np.zeros(grayscale_cam.shape, dtype=np.float32)
        for x1, y1, x2, y2 in boxes:
            x1, y1 = max(x1, 0), max(y1, 0)
            x2, y2 = min(grayscale_cam.shape[1] - 1, x2), min(grayscale_cam.shape[0] - 1, y2)
            renormalized_cam[y1:y2, x1:x2] = scale_cam_image(grayscale_cam[y1:y2, x1:x2].copy())
        renormalized_cam = scale_cam_image(renormalized_cam)
        eigencam_image_renormalized = show_cam_on_image(image_float_np, renormalized_cam, use_rgb=True)
        return eigencam_image_renormalized

    def process(self, img_path, save_path):
        # img process
        ir_path = img2ir_paths([img_path])[0]
        ir_img = cv2.imread(ir_path, cv2.IMREAD_GRAYSCALE)  # 读取红外图像为灰度图像
        ir_img = letterbox(ir_img)[0]  # letterbox操作
        ir_img = np.expand_dims(ir_img, axis=-1)  # 增加一个通道维度
        ir_img = np.repeat(ir_img, 3, axis=-1)  # 将单通道扩展为三通道，如果模型需要RGB输入
        ir_img = np.float32(ir_img) / 255.0  # 归一化
        global  tensor_ir
        tensor_ir = torch.from_numpy(np.transpose(ir_img, axes=[2, 0, 1])).unsqueeze(0).to(self.device)

        img = cv2.imread(img_path)
        img = letterbox(img)[0]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.float32(img) / 255.0
        tensor = torch.from_numpy(np.transpose(img, axes=[2, 0, 1])).unsqueeze(0).to(self.device)

        try:
            grayscale_cam = self.method(tensor, [self.target])
        except AttributeError as e:
            return

        grayscale_cam = grayscale_cam[0, :]
        cam_image = show_cam_on_image(img, grayscale_cam, use_rgb=True)

        pred = self.model(tensor, tensor_ir)['one2one'][0]
        pred = self.post_process(pred)
        if self.renormalize:
            cam_image = self.renormalize_cam_in_bounding_boxes(pred[:, :4].cpu().detach().numpy().astype(np.int32), img,
                                                               grayscale_cam)
        if self.show_box:
            for data in pred:
                data = data.cpu().detach().numpy()
                cam_image = self.draw_detections(data[:4], self.colors[int(data[4:].argmax())],
                                                 f'{self.model_names[int(data[4:].argmax())]} {float(data[4:].max()):.2f}',
                                                 cam_image)

        cam_image = Image.fromarray(cam_image)
        cam_image.save(save_path)

    def __call__(self, img_path, save_path):
        # remove dir if exist
        # if os.path.exists(save_path):
        #     shutil.rmtree(save_path)  #你tm写这玩意搁着？？？害的我文件都没了！你大爷的！
        # make dir if not exist
        os.makedirs(save_path, exist_ok=True)

        if os.path.isdir(img_path):
            for img_path_ in os.listdir(img_path):
                self.process(f'{img_path}/{img_path_}', f'{save_path}/{img_path_}')
        else:
            self.process(img_path, f'{save_path}/vedai_rgb_only'+'_layer'+str(self.layer[0])+'.png')

#/home/qjc/Project/yolov10-main/runs/detect/llvipmul_250ep_ir/weights/best.pt
#/home/qjc/Project/yolov10-main/runs/detect/llvipmul_250ep_rgb/weights/best.pt
#/home/qjc/Project/yolov10-main/runs/detect/llvipmul_250ep_twok2/weights/best.pt

#/home/qjc/Project/yolov10-main/runs/detect/VEDAI_nc9_channelsqueeze/weights/best.pt
def get_params(layer):
    params = {
        'weight': '/home/qjc/Project/yolov10-main/runs/detect/lsknet_rgb_200ep/weights/best.pt',  # 现在只需要指定权重即可,不需要指定cfg
        'device': 'cuda:0',
        'method': 'EigenCAM',
        # GradCAMPlusPlus, GradCAM, XGradCAM, EigenCAM, HiResCAM, LayerCAM, RandomCAM, EigenGradCAM
        'layer': layer,
        'backward_type': 'all',  # class, box, all
        'conf_threshold': 0.4,  # 0.2
        'ratio': 0.02,  # 0.02-0.1
        'show_box': False,
        'renormalize': False
    }
    return params

# def get_params():
#     params = {
#         'weight': '/home/qjc/Project/yolov10-main/runs/detect/llvipmul_250ep_twok2/weights/best.pt', # 这选择想要热力可视化的模型权重路径
#         'cfg': 'lsknet.yaml', # 这里选择与训练上面模型权重相对应的.yaml文件路径
#         'device': 'cuda:0', # 选择设备，其中0表示0号显卡。如果使用CPU可视化 # 'device': 'cpu'
#         'method': 'GradCAM', # GradCAMPlusPlus, GradCAM, XGradCAM
#         'layer': 'model.model[4]',   # 选择特征层
#         'backward_type': 'all', # class, box, all
#         'conf_threshold': 0.65, # 置信度阈值默认0.65， 可根据情况调节
#         'ratio': 0.02 # 取前多少数据，默认是0.02，可根据情况调节
#     }
#     return params

if __name__ == '__main__':
    from ultralytics.data.base import img2ir_paths
    img_path = '/home/qjc/DataSet/DATA/VEDAI_1024/images/00000485_co.png'
    # layer = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
    layer = [7]
    for layer in layer:
        model = yolov8_heatmap(**get_params([layer]))
    # model(r'/home/hjj/Desktop/dataset/dataset_visdrone/VisDrone2019-DET-test-dev/images/9999947_00000_d_0000026.jpg', 'result')
        model(img_path, '/home/qjc/Project/1')

#
# if __name__ == '__main__':
#     model = yolov8_heatmap(**get_params()) # 初始化
#     model('/home/qjc/DataSet/LLvip/LLVIP/visible/train/010001.jpg', '/home/qjc/Project/yolov10-main/热力图/') # 第一个参数是图片的路径，第二个参数是保存路径，比如是result的话，其会创建一个名字为result的文件夹，如果result文件夹不为空，其会先清空文件夹。

#/home/qjc/DataSet/LLvip/LLVIP/visible/train/080869.jpg