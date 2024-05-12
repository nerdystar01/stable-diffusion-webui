import re
import io
import base64
import json
from contextlib import closing

import modules.scripts
from modules import processing, infotext_utils
from modules.infotext_utils import create_override_settings_dict, parse_generation_parameters
from modules.shared import opts
import modules.shared as shared
from modules.ui import plaintext_to_html
from PIL import Image
import gradio as gr

# 작업 
import sys
import os

# 현재 파일의 절대 경로를 기준으로 상위 디렉토리 경로를 구함
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(os.path.dirname(current_dir))  # 두 단계 상위 디렉토리

# extensions 디렉토리 경로 추가
sys.path.append(root_dir)

# 이제 절대 경로로 모듈을 임포트할 수 있습니다.
from extensions.sd_webui_controlnet.internal_controlnet.args import ControlNetUnit


def txt2img_create_processing(id_task: str, request: gr.Request, prompt: str, negative_prompt: str, prompt_styles, n_iter: int, batch_size: int, cfg_scale: float, height: int, width: int, enable_hr: bool, denoising_strength: float, hr_scale: float, hr_upscaler: str, hr_second_pass_steps: int, hr_resize_x: int, hr_resize_y: int, hr_checkpoint_name: str, hr_sampler_name: str, hr_scheduler: str, hr_prompt: str, hr_negative_prompt, override_settings_texts, *args, force_enable_hr=False):
    override_settings = create_override_settings_dict(override_settings_texts)

    if force_enable_hr:
        enable_hr = True

    p = processing.StableDiffusionProcessingTxt2Img(
        sd_model=shared.sd_model,
        outpath_samples=opts.outdir_samples or opts.outdir_txt2img_samples,
        outpath_grids=opts.outdir_grids or opts.outdir_txt2img_grids,
        prompt=prompt,
        styles=prompt_styles,
        negative_prompt=negative_prompt,
        batch_size=batch_size,
        n_iter=n_iter,
        cfg_scale=cfg_scale,
        width=width,
        height=height,
        enable_hr=enable_hr,
        denoising_strength=denoising_strength,
        hr_scale=hr_scale,
        hr_upscaler=hr_upscaler,
        hr_second_pass_steps=hr_second_pass_steps,
        hr_resize_x=hr_resize_x,
        hr_resize_y=hr_resize_y,
        hr_checkpoint_name=None if hr_checkpoint_name == 'Use same checkpoint' else hr_checkpoint_name,
        hr_sampler_name=None if hr_sampler_name == 'Use same sampler' else hr_sampler_name,
        hr_scheduler=None if hr_scheduler == 'Use same scheduler' else hr_scheduler,
        hr_prompt=hr_prompt,
        hr_negative_prompt=hr_negative_prompt,
        override_settings=override_settings,
    )

    p.scripts = modules.scripts.scripts_txt2img
    p.script_args = args

    p.user = request.username

    if shared.opts.enable_console_prompts:
        print(f"\ntxt2img: {prompt}", file=shared.progress_print_out)

    return p


def txt2img_upscale(id_task: str, request: gr.Request, gallery, gallery_index, generation_info, *args):
    assert len(gallery) > 0, 'No image to upscale'
    assert 0 <= gallery_index < len(gallery), f'Bad image index: {gallery_index}'

    p = txt2img_create_processing(id_task, request, *args, force_enable_hr=True)
    p.batch_size = 1
    p.n_iter = 1
    # txt2img_upscale attribute that signifies this is called by txt2img_upscale
    p.txt2img_upscale = True

    geninfo = json.loads(generation_info)

    image_info = gallery[gallery_index] if 0 <= gallery_index < len(gallery) else gallery[0]
    p.firstpass_image = infotext_utils.image_from_url_text(image_info)

    parameters = parse_generation_parameters(geninfo.get('infotexts')[gallery_index], [])
    p.seed = parameters.get('Seed', -1)
    p.subseed = parameters.get('Variation seed', -1)

    p.override_settings['save_images_before_highres_fix'] = False

    with closing(p):
        processed = modules.scripts.scripts_txt2img.run(p, *p.script_args)

        if processed is None:
            processed = processing.process_images(p)

    shared.total_tqdm.clear()

    new_gallery = []
    for i, image in enumerate(gallery):
        if i == gallery_index:
            geninfo["infotexts"][gallery_index: gallery_index+1] = processed.infotexts
            new_gallery.extend(processed.images)
        else:
            fake_image = Image.new(mode="RGB", size=(1, 1))
            fake_image.already_saved_as = image["name"].rsplit('?', 1)[0]
            new_gallery.append(fake_image)

    geninfo["infotexts"][gallery_index] = processed.info

    return new_gallery, json.dumps(geninfo), plaintext_to_html(processed.info), plaintext_to_html(processed.comments, classname="comments")

def txt2img(id_task: str, request: gr.Request, *args):
    p = txt2img_create_processing(id_task, request, *args)
    print("확인해볼게 있습니다.")
    print(p.script_args)
    print("확인해볼게 있습니다.")
    with closing(p):
        processed = modules.scripts.scripts_txt2img.run(p, *p.script_args)

        if processed is None:
            processed = processing.process_images(p)

    shared.total_tqdm.clear()

    generation_info_js = processed.js()
    if opts.samples_log_stdout:
        print(generation_info_js)

    if opts.do_not_show_images:
        processed.images = []

    return processed.images, generation_info_js, plaintext_to_html(processed.info), plaintext_to_html(processed.comments, classname="comments")

def txt2img_with_server(id_task: str, request: gr.Request, *args):
    

    email_input = args[0]
    print("이메일 등록 확인2")
    print(email_input)

    generate_params = args[1:]
    print("실제 생성 파라미티 확인")
    print(generate_params)

    p = txt2img_create_processing(id_task, request, *generate_params)
    
    sd_dict = {
        "model_hash" : p.sd_model_hash, 
        "model_name" : p.sd_model_name,
        "sampler" : p.sampler,
        "prompt" : p.prompt,
        "negative_prompt" : p.negative_prompt,
        "width" : p.width,
        "height" : p.height,
        "steps" : p.steps,
        "cfg_scale" : p.cfg_scale,
        "seed" : p.seed,
        "sd_vae" : p.sd_vae_name
    }
    
    if p.enable_hr == True:
        hires_dict = {
            "is_highres" : True,
            "hr_upscaler" : p.hr_upscaler,
            "hr_steps" : p.hr_second_pass_steps,
            "hr_denoising_strength" : p.denoising_strength,
            "hr_upscale_by" : p.hr_scale,
        }
        sd_dict.update(hires_dict)
        
    controlnet_unit_list = []
    
    for obj in p.script_args:
        if not isinstance(obj, (float, int, str, bool, type(None))) and not obj == []:
            this_instance = obj
            controlnet_dict = this_instance.dict()
            server_controlnet_dict = {}
            if controlnet_dict['enabled'] == True:
                print("controlnet 존재합니다.")
                server_controlnet_dict["threshold_a"] = controlnet_dict['threshold_a']
                server_controlnet_dict["threshold_b"] = controlnet_dict['threshold_b']
                server_controlnet_dict["guidance_start"] = controlnet_dict['guidance_start']
                server_controlnet_dict["guidance_end"] = controlnet_dict['guidance_end']
                server_controlnet_dict["lowvram"] = controlnet_dict['low_vram']
                server_controlnet_dict["is_pixel_perfect"] = controlnet_dict['pixel_perfect']
                server_controlnet_dict["processor_res"] = controlnet_dict['processor_res']
                server_controlnet_dict["weight"] = controlnet_dict['weight']
                server_controlnet_dict["module"] = controlnet_dict['module']


                if not controlnet_dict["image"] == None:
                    image_data = controlnet_dict["image"]["image"]
                    pil_img = Image.fromarray(image_data)
                    buffer = io.BytesIO()
                    pil_img.save(buffer, format='PNG')
                    img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
                    server_controlnet_dict["image"] = img_base64

                    mask_data = controlnet_dict["image"]["mask"]
                    pil_img = Image.fromarray(mask_data)
                    buffer = io.BytesIO()
                    pil_img.save(buffer, format='PNG')
                    img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
                    server_controlnet_dict["mask"] = img_base64
                
                else:
                    server_controlnet_dict["image"] = ""
                    server_controlnet_dict["mask"] = ""

                controlnet_unit_list.append(server_controlnet_dict)    
    
    return sd_dict, controlnet_unit_list
            