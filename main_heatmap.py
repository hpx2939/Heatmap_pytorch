import os
import cv2
import time
import torch
import random
import argparse
import numpy as np
import torch.nn as nn
import torchvision
import matplotlib.pyplot as plt


# from torchvision import models

from pytorch_grad_cam import GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad

from pytorch_grad_cam.utils.image import show_cam_on_image
# from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget





# Arguments
parser = argparse.ArgumentParser()
parser.add_argument('--model-path', type=str, default="torchvision.models.resnet.ResNet50_Weights.DEFAULT", help='Path to the model')
parser.add_argument('--img-path', type=str, default='images', help='input image path')
parser.add_argument('--output-dir', type=str, default='heat_outputs/', help='output dir')
parser.add_argument('--target-layer', type=str, default='[model.layer4[-2]]',
                    help='The layer hierarchical address to which gradcam will applied,'
                         ' the names should be separated by underline')
parser.add_argument('--method', type=str, default='gradcampp', help='gradcam method: gradcam, gradcampp')
parser.add_argument('--device', type=str, default='cuda', help='cuda or cpu')
# parser.add_argument('--names', type=str, default=None,
#                     help='If you want. Provide your custom names as follow: object1,object2,object3')
# parser.add_argument('--img-size', type=int, default=640, help="input image size")
args = parser.parse_args()


def get_res_img(mask, res_img):
    
    mask = mask.squeeze(0).mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).detach().cpu().numpy().astype(
        np.uint8)
    heatmap = cv2.applyColorMap(mask, cv2.COLORMAP_JET)
    n_heatmat = (heatmap / 255).astype(np.float32)
    res_img = res_img / 255
    res_img = cv2.add(res_img, n_heatmat)
    res_img = (res_img / res_img.max())
    return res_img, n_heatmat


def plot_one_box(x, img, color=None, label=None, line_thickness=3):
    # this is a bug in cv2. It does not put box on a converted image from torch unless it's buffered and read again!
    cv2.imwrite('temp.jpg', (img * 255).astype(np.uint8))
    img = cv2.imread('temp.jpg')

    # Plots one bounding box on image img
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    # cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        outside = c1[1] - t_size[1] - 3 >= 0  # label fits outside box up
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3 if outside else c1[1] + t_size[1] + 3
        outsize_right = c2[0] - img.shape[:2][1] > 0  # label fits outside box right
        c1 = c1[0] - (c2[0] - img.shape[:2][1]) if outsize_right else c1[0], c1[1]
        c2 = c2[0] - (c2[0] - img.shape[:2][1]) if outsize_right else c2[0], c2[1]
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(img, label, (c1[0], c1[1] - 2 if outside else c2[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf,
                    lineType=cv2.LINE_AA)
    return img

def preprocessing(img,device):
    if len(img.shape) != 4:
        img = np.expand_dims(img, axis=0)
    im0 = img.astype(np.uint8)
    img = np.array(im0)
    img = img.transpose((0, 3, 1, 2))
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img).to(device)
    img = img / 255.0
    return img

def main(img_path, img_name):
    # detect single image
    device = args.device
    img = cv2.imread(img_path)  # BGR


    model = torchvision.models.resnet50(args.model_path)
    target_layers = args.target_layer

    
    # img[..., ::-1]: BGR --> RGB
    # (480, 640, 3) --> (1, 3, 480, 640)
    torch_img = preprocessing(img[..., ::-1],device)
    
    tic = time.time()
    for target_layer in target_layers:
        if args.method == 'gradcam':
            saliency_method = GradCAM(model=model, target_layers=target_layer)

        elif args.method == 'gradcampp':
            saliency_method = GradCAMPlusPlus(model=model, target_layers=target_layer)

        masks = saliency_method(torch_img)  #get the result
        result = torch_img.squeeze(0).mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).detach().cpu().numpy()
        result = result[..., ::-1]  # convert to bgr
        
        imgae_name = os.path.basename(img_path) 
        save_path = f'{args.output_dir}/{args.method}'
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        print(f'[INFO] Saving the final image at {save_path}')


        # targets = [ClassifierOutputTarget(254)]
        masks = saliency_method(input_tensor=torch_img)
        masks = masks[0, :]
        visualization = show_cam_on_image(img.astype(dtype=np.float32)/255.,
                                        masks, use_rgb=True)

        # plt.imshow(visualization)
        # plt.show()

        visualization = visualization[..., ::-1] #RGB->BGR
        output_path = f'{save_path}/{img_name}'
        print(output_path)
        cv2.imwrite(output_path, visualization)
        print(f'{imgae_name[:-4]}_{img_name} done!!')

    print(f'Total time : {round(time.time() - tic, 4)} s')


if __name__ == '__main__':
    if os.path.isdir(args.img_path):
        img_list = os.listdir(args.img_path)
        print(img_list)
        for item in img_list:
            print(item)
            main(os.path.join(args.img_path, item), item)
    else:
        main(args.img_path)
