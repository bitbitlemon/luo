import torch
from basicsr.utils.misc import gpu_is_available
import gradio as gr
import cv2
from run import main1
pretrain_model_url = {
    'restoration': 'https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/codeformer.pth',
}
def set_realesrgan():
    from basicsr.archs.rrdbnet_arch import RRDBNet
    from basicsr.utils.realesrgan_utils import RealESRGANer

    use_half = False
    if torch.cuda.is_available():
        no_half_gpu_list = ['1650', '1660']
        if not True in [gpu in torch.cuda.get_device_name(0) for gpu in no_half_gpu_list]:
            use_half = True

    model = RRDBNet(
        num_in_ch=3,
        num_out_ch=3,
        num_feat=64,
        num_block=23,
        num_grow_ch=32,
        scale=2,
    )
    upsampler = RealESRGANer(
        scale=2,
        model_path="https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/RealESRGAN_x2plus.pth",
        model=model,
        tile_pad=40,
        pre_pad=0,
        half=use_half
    )
    if not gpu_is_available():
        import warnings
        warnings.warn('Running on CPU now! Make sure your PyTorch version matches your CUDA.'
                        'The unoptimized RealESRGAN is slow on CPU. '
                        'If you want to disable it, please remove `--bg_upsampler` and `--face_upsample` in command.',
                        category=RuntimeWarning)
    return upsampler
def process_image(input_image, fidelity_weight=0.5, upscale=2):
    # Save input image to ./inputs directory
    input_path = "./inputs/input_image.jpg"
    cv2.imwrite(input_path, input_image)

    # Call face restoration function
    main1()

    # Load restored image from restored_faces directory
    output_path = "./results/restored_faces/input_image_00.png"
    restored_image = cv2.imread(output_path)

    # Convert BGR to RGB
    restored_image_rgb = cv2.cvtColor(restored_image, cv2.COLOR_BGR2RGB)



    return restored_image_rgb

# Define Gradio interface
inputs = gr.Image(label="Input Image")
outputs = gr.Image(label="Restored Image")

title = "人脸修复"
description = "请上传修复图片"

gr.Interface(
    fn=process_image,
    inputs=inputs,
    outputs=outputs,
    title=title,
    # examples=[['./inputs/0240.png']],
    description=description
).launch()


