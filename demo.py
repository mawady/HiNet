import math
import torch
import torch.nn
import torchvision
import numpy as np
from PIL import Image
from datasets import to_rgb, transform_test
from model import *
from modules import Unet_common as common
# import config as c

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
channels_in = 3
device_ids = [0]

def load_model(model_path):
    net = Model()
    net = net.to(device)
    init_model(net, device=device)
    net = torch.nn.DataParallel(net, device_ids=device_ids)
    state_dicts = torch.load(model_path, map_location=device)
    network_state_dict = {
        k: v for k, v in state_dicts["net"].items() if "tmp_var" not in k
    }
    net.load_state_dict(network_state_dict)
    return net


def gauss_noise(shape):
    noise = torch.zeros(shape, device=device)
    for i in range(noise.shape[0]):
        noise[i] = torch.randn(noise[i].shape, device=device)
    return noise


def encode_watermark(model_path, cover, secret):
    net = load_model(model_path)
    net.eval()
    dwt = common.DWT()
    iwt = common.IWT()
    with torch.no_grad():
        cover_input = dwt(cover)
        secret_input = dwt(secret)
        input_img = torch.cat((cover_input, secret_input), 1)
        output = net(input_img)
        output_steg = output.narrow(1, 0, 4 * channels_in)
        output_z = output.narrow(
            1, 4 * channels_in, output.shape[1] - 4 * channels_in
        )
        steg_img = iwt(output_steg)
    return steg_img, output_z


def decode_watermark(model_path, steg_img, output_z=None):
    net = load_model(model_path)
    net.eval()
    dwt = common.DWT()
    iwt = common.IWT()
    if output_z is not None:
        z_shape = output_z.shape
    else:
        z_shape = (1,12,512,512)
    backward_z = gauss_noise(z_shape)
    # print(backward_z.shape)
    output_steg = dwt(steg_img)
    with torch.no_grad():
        output_rev = torch.cat((output_steg, backward_z), 1)
        bacward_img = net(output_rev, rev=True)
        secret_rev = bacward_img.narrow(
            1, 4 * channels_in, bacward_img.shape[1] - 4 * channels_in
        )
        secret_rev = iwt(secret_rev)
    return secret_rev


def process_img(imgPath):
    img = Image.open(imgPath)
    img = to_rgb(img)
    img = transform_test(img)
    img = img[None, :]
    return img


def process_out(tensor_obj):
    numpy_obj =  (
        tensor_obj.squeeze()
        .mul(255)
        .add_(0.5)
        .clamp_(0, 255)
        .permute(1, 2, 0)
        .to("cpu", torch.uint8)
        .numpy()
    )
    return Image.fromarray(numpy_obj)


if __name__ == "__main__":
    imgSrcPath = "/Users/mohamedelawady/##Research##/repos/ICASSP22-JPEG-AI-watermarking/data/kodak_imgs/kodim01.png"
    imgWtrPath = "/Users/mohamedelawady/##Research##/repos/ICASSP22-JPEG-AI-watermarking/data/kodak_imgs/kodim02.png"
    model_path = "/Users/mohamedelawady/##Research##/repos/ICASSP22-JPEG-AI-watermarking/models/HiNet/model.pt"
    imgSrc = process_img(imgSrcPath)
    imgWtr = process_img(imgWtrPath)
    steg_img, output_z = encode_watermark(model_path, imgSrc, imgWtr)
    im_enc = process_out(steg_img)
    im_enc.save("temp_enc.jpg")
    img_src = process_out(imgSrc)
    img_src.save("img_src.jpg")

    img_wtr = process_out(imgWtr)
    img_wtr.save("img_wtr.jpg")

    print(output_z.shape)
    print(steg_img.shape)

    secret_rev = decode_watermark(model_path, steg_img, output_z=None)
    img_rec = process_out(secret_rev)
    img_rec.save("img_rec.jpg")