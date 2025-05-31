import os.path
import cv2
import logging

import numpy as np
import torch
from datetime import datetime
from collections import OrderedDict

from utils import utils_model
from utils import utils_logger
from utils import utils_image as util
from utils.utils_inpaint import mask_generator

# from guided_diffusion import dist_util
from guided_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
)


def inpaint_service_demo(mask_type='box',
                         input_image_path='/Users/zhiyuzhang/Downloads/DiffPIR/testsets/demo_test/mingrui.png',
                         output_path='/Users/zhiyuzhang/Downloads/DiffPIR/results'):

    path_to_return = {}

    # ----------------------------------------
    # Preparation
    # ----------------------------------------

    noise_level_img = 15 / 255.0  # set AWGN noise level for LR image, default: 0
    noise_level_model = noise_level_img  # set noise level of model, default: 0
    model_name = '256x256_diffusion_uncond'  # 256x256_diffusion_uncond, diffusion_ffhq_10m; set diffusino model
    # testset_name = 'demo_test'  # set testing set, 'imagenet_val' | 'ffhq_val'
    num_train_timesteps = 1000
    iter_num = 50  # set number of iterations
    iter_num_U = 1  # set number of inner iterations, default: 1
    skip = num_train_timesteps // iter_num  # skip interval

    mask_name = 'gt_keep_masks/face/000000.png'  # mask path for loading mask img
    load_mask = False
    # mask_type = 'box'  # ['box', 'random', 'both', 'extreme']
    # mask_len_range          = [128, 129]
    mask_len_range = [128, 129]
    mask_prob_range = [0.5, 0.5]

    show_img = False  # default: False
    save_L = True  # save LR image
    save_E = True  # save estimated image
    save_LEH = True  # save zoomed LR, E and H images
    save_progressive = True  # save generation process
    save_progressive_mask = True  # save generation process

    sigma = max(0.05, noise_level_img)  # noise level associated with condition y
    lambda_ = 1.  # key parameter lambda
    sub_1_analytic = True  # use analytical solution
    eta = 0.0  # eta for ddim samplingn
    zeta = 1.0
    guidance_scale = 1.0

    model_out_type = 'pred_xstart'  # model output type: pred_x_prev; pred_xstart; epsilon; score
    generate_mode = 'DiffPIR'  # repaint; vanilla; DiffPIR
    skip_type = 'quad'  # uniform, quad
    ddim_sample = False  # sampling method

    log_process = False
    task_current = 'ip'  # 'ip' for inpainting
    n_channels = 3  # fixed
    # cwd = os.path.dirname(os.path.abspath(__file__))
    cwd = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_zoo = os.path.join(cwd, 'model_zoo')  # fixed
    # testsets = os.path.join(cwd, 'testsets')  # fixed
    # results = os.path.join(cwd, 'results')  # fixed
    result_name = f'{task_current}_{generate_mode}_{mask_type}_{model_name}_sigma{noise_level_img}_NFE{iter_num}_eta{eta}_zeta{zeta}_lambda{lambda_}'
    model_path = os.path.join(model_zoo, model_name + '.pt')
    device = torch.device(
        'cuda' if torch.cuda.is_available() else
        'mps' if torch.backends.mps.is_available() else
        'cpu'
    )
    torch.cuda.empty_cache()

    calc_LPIPS = False

    # noise schedule
    beta_start = 0.1 / 1000
    beta_end = 20 / 1000
    betas = np.linspace(beta_start, beta_end, num_train_timesteps, dtype=np.float32)
    betas = torch.from_numpy(betas).to(device)
    alphas = 1.0 - betas
    alphas_cumprod = np.cumprod(alphas.cpu(), axis=0)
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_1m_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)
    reduced_alpha_cumprod = torch.div(sqrt_1m_alphas_cumprod, sqrt_alphas_cumprod)  # equivalent noise sigma on image

    noise_model_t = utils_model.find_nearest(reduced_alpha_cumprod, 2 * noise_level_model)
    noise_model_t = 0

    noise_inti_img = 50 / 255
    t_start = utils_model.find_nearest(reduced_alpha_cumprod,
                                       2 * noise_inti_img)  # start timestep of the diffusion process
    t_start = num_train_timesteps - 1

    # ----------------------------------------
    # L_path, E_path, H_path, mask_path
    # ----------------------------------------

    # L_path = os.path.join(testsets, testset_name)  # L_path, for Low-quality images
    E_path = os.path.join(output_path, result_name)  # E_path, for Estimated images
    # mask_path = os.path.join(testsets, mask_name)  # mask_path, for mask images
    # mask_path = ''

    util.mkdir(E_path)

    logger_name = result_name
    utils_logger.logger_info(logger_name, log_path=os.path.join(E_path, logger_name + '.log'))
    logger = logging.getLogger(logger_name)

    # ----------------------------------------
    # load model
    # ----------------------------------------

    model_config = dict(
        model_path=model_path,
        num_channels=128,
        num_res_blocks=1,
        attention_resolutions="16",
    ) if model_name == 'diffusion_ffhq_10m' \
        else dict(
        model_path=model_path,
        num_channels=256,
        num_res_blocks=2,
        attention_resolutions="8,16,32",
    )
    args = utils_model.create_argparser(model_config).parse_args([])
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys()))
    # model.load_state_dict(
    #     dist_util.load_state_dict(args.model_path, map_location="cpu")
    # )
    model.load_state_dict(torch.load(args.model_path, map_location="cpu"))
    model.eval()
    for k, v in model.named_parameters():
        v.requires_grad = False
    model = model.to(device)

    logger.info('model_name:{}, mask_type:{}, image sigma:{:.3f}, model sigma:{:.3f}'.format(model_name, mask_type,
                                                                                             noise_level_img,
                                                                                             noise_level_model))
    logger.info(
        'eta:{:.3f}, zeta:{:.3f}, lambda:{:.3f}, guidance_scale:{:.2f} '.format(eta, zeta, lambda_, guidance_scale))
    logger.info(
        'start step:{}, skip_type:{}, skip interval:{}, skipstep analytic steps:{}'.format(t_start, skip_type, skip,
                                                                                           noise_model_t))
    logger.info('Model path: {:s}'.format(model_path))
    logger.info(input_image_path)

    if calc_LPIPS:
        import lpips
        loss_fn_vgg = lpips.LPIPS(net='vgg').to(device)

    def test_rho(lambda_=lambda_, model_out_type_=model_out_type, zeta=zeta):
        model_out_type = model_out_type_
        logger.info(
            'eta:{:.3f}, zeta:{:.3f}, lambda:{:.3f}, guidance_scale:{:.2f}'.format(eta, zeta, lambda_, guidance_scale))
        test_results = OrderedDict()
        test_results['psnr'] = []
        if calc_LPIPS:
            test_results['lpips'] = []

        # --------------------------------
        # (1) get img_H and img_L from input_image_path
        # --------------------------------
        img = input_image_path
        img_name, ext = os.path.splitext(os.path.basename(img))
        img_H = util.imread_uint(img, n_channels=n_channels)

        # --------------------------------
        # (2) initialize x
        # --------------------------------

        mask_gen = mask_generator(mask_type=mask_type, mask_len_range=mask_len_range,
                                  mask_prob_range=mask_prob_range)
        np.random.seed(seed=0)  # for reproducibility
        mask = mask_gen(util.uint2tensor4(img_H)).numpy()
        mask = np.squeeze(mask)
        mask = np.transpose(mask, (1, 2, 0))

        img_L = img_H * mask / 255.  # (256,256,3)         [0,1]

        np.random.seed(seed=0)  # for reproducibility
        img_L = img_L * 2 - 1
        img_L += np.random.normal(0, noise_level_img * 2, img_L.shape)  # add AWGN
        img_L = img_L / 2 + 0.5
        img_L = img_L * mask

        y = util.single2tensor4(img_L).to(device)  # (1,3,256,256)
        y = y * 2 - 1  # [-1,1]
        mask = util.single2tensor4(mask.astype(np.float32)).to(device)

        # for y with given noise level, add noise from t_y
        t_y = utils_model.find_nearest(reduced_alpha_cumprod, 2 * noise_level_img)
        sqrt_alpha_effective = sqrt_alphas_cumprod[t_start] / sqrt_alphas_cumprod[t_y]
        x = sqrt_alpha_effective * y + torch.sqrt(sqrt_1m_alphas_cumprod[t_start] ** 2 - \
                                                  sqrt_alpha_effective ** 2 * sqrt_1m_alphas_cumprod[
                                                      t_y] ** 2) * torch.randn_like(y)
        # x = sqrt_alphas_cumprod[t_start] * y + sqrt_1m_alphas_cumprod[t_start] * torch.randn_like(y)

        # --------------------------------
        # (3) get rhos and sigmas
        # --------------------------------

        sigmas = []
        sigma_ks = []
        rhos = []
        for i in range(num_train_timesteps):
            sigmas.append(reduced_alpha_cumprod[num_train_timesteps - 1 - i])
            if model_out_type == 'pred_xstart':
                sigma_ks.append((sqrt_1m_alphas_cumprod[i] / sqrt_alphas_cumprod[i]))
            elif model_out_type == 'pred_x_prev':
                sigma_ks.append(torch.sqrt(betas[i] / alphas[i]))
            rhos.append(lambda_ * (sigma ** 2) / (sigma_ks[i] ** 2))

        rhos, sigmas, sigma_ks = torch.tensor(rhos).to(device), torch.tensor(sigmas).to(device), torch.tensor(
            sigma_ks).to(device)

        # --------------------------------
        # (4) main iterations
        # --------------------------------

        progress_img = []
        # create sequence of timestep for sampling
        if skip_type == 'uniform':
            seq = [i * skip for i in range(iter_num)]
            if skip > 1:
                seq.append(num_train_timesteps - 1)
        elif skip_type == "quad":
            seq = np.sqrt(np.linspace(0, num_train_timesteps ** 2, iter_num))
            seq = [int(s) for s in list(seq)]
            seq[-1] = seq[-1] - 1
        progress_seq = seq[::(len(seq) // 10)]
        progress_seq.append(seq[-1])

        # reverse diffusion for one image from random noise
        for i in range(len(seq)):
            curr_sigma = sigmas[seq[i]].cpu().numpy()
            # time step associated with the noise level sigmas[i]
            t_i = utils_model.find_nearest(reduced_alpha_cumprod, curr_sigma)
            # skip iters
            if t_i > t_start:
                continue
            for u in range(iter_num_U):
                # --------------------------------
                # step 1, reverse diffsuion step
                # --------------------------------

                # add noise, make the image noise level consistent in pixel level
                if generate_mode == 'repaint':
                    x = (sqrt_alphas_cumprod[t_i] * y + sqrt_1m_alphas_cumprod[t_i] * torch.randn_like(x)) * mask \
                        + (1 - mask) * x

                # solve equation 6b with one reverse diffusion step
                if model_out_type == 'pred_xstart':
                    x0 = utils_model.model_fn(x, noise_level=curr_sigma * 255, model_out_type=model_out_type, \
                                              model_diffusion=model, diffusion=diffusion, ddim_sample=ddim_sample,
                                              alphas_cumprod=alphas_cumprod)
                else:
                    x = utils_model.model_fn(x, noise_level=curr_sigma * 255, model_out_type=model_out_type, \
                                             model_diffusion=model, diffusion=diffusion, ddim_sample=ddim_sample,
                                             alphas_cumprod=alphas_cumprod)
                # x = utils_model.test_mode(model_fn, x, mode=0, refield=32, min_size=256, modulo=16, noise_level=sigmas[i].cpu().numpy()*255)

                # --------------------------------
                # step 2, closed-form solution
                # --------------------------------

                # analytic solution
                if (generate_mode == 'DiffPIR') and not (seq[i] == seq[-1]):
                    # solve sub-problem
                    if sub_1_analytic:
                        if model_out_type == 'pred_xstart':
                            # when noise level less than given image noise, skip
                            if i < num_train_timesteps - noise_model_t:
                                x0_p = (mask * y + rhos[t_i].float() * x0).div(mask + rhos[t_i])
                                x0 = x0 + guidance_scale * (x0_p - x0)
                            else:
                                model_out_type = 'pred_x_prev'
                                x0 = utils_model.model_fn(x, noise_level=curr_sigma * 255,
                                                          model_out_type=model_out_type, \
                                                          model_diffusion=model, diffusion=diffusion,
                                                          ddim_sample=ddim_sample, alphas_cumprod=alphas_cumprod)
                                pass
                        elif model_out_type == 'pred_x_prev':
                            # when noise level less than given image noise, skip
                            if i < num_train_timesteps - noise_model_t:
                                x = (mask * y + rhos[t_i].float() * x).div(mask + rhos[t_i])  # y-->yt ?
                            else:
                                pass
                    else:
                        # TODO: first order solver
                        # x = x - 1 / (2*rhos[t_i]) * (x - y_t) * mask
                        pass

                if (model_out_type == 'pred_xstart') and not (seq[i] == seq[-1]):
                    # x = sqrt_alphas_cumprod[t_i] * (x) + (sqrt_1m_alphas_cumprod[t_i]) *  torch.randn_like(x) # x = sqrt_alphas_cumprod[t_i] * (x) + (sqrt_1m_alphas_cumprod[t_i]) *  torch.randn_like(x)

                    t_im1 = utils_model.find_nearest(reduced_alpha_cumprod, sigmas[seq[i + 1]].cpu().numpy())
                    # calculate \hat{\eposilon}
                    eps = (x - sqrt_alphas_cumprod[t_i] * x0) / sqrt_1m_alphas_cumprod[t_i]
                    eta_sigma = eta * sqrt_1m_alphas_cumprod[t_im1] / sqrt_1m_alphas_cumprod[t_i] * torch.sqrt(
                        betas[t_i])
                    x = sqrt_alphas_cumprod[t_im1] * x0 + np.sqrt(1 - zeta) * (
                                torch.sqrt(sqrt_1m_alphas_cumprod[t_im1] ** 2 - eta_sigma ** 2) * eps \
                                + eta_sigma * torch.randn_like(x)) + np.sqrt(zeta) * sqrt_1m_alphas_cumprod[
                            t_im1] * torch.randn_like(x)

                # set back to x_t from x_{t-1}
                if u < iter_num_U - 1 and seq[i] != seq[-1]:
                    # x = torch.sqrt(alphas[t_i]) * x + torch.sqrt(betas[t_i]) * torch.randn_like(x)
                    sqrt_alpha_effective = sqrt_alphas_cumprod[t_i] / sqrt_alphas_cumprod[t_im1]
                    x = sqrt_alpha_effective * x + torch.sqrt(sqrt_1m_alphas_cumprod[t_i] ** 2 - \
                                                              sqrt_alpha_effective ** 2 * sqrt_1m_alphas_cumprod[
                                                                  t_im1] ** 2) * torch.randn_like(x)

            # save the process
            x_0 = (x / 2 + 0.5)
            if save_progressive and (seq[i] in progress_seq):
                x_show = x_0.clone().detach().cpu().numpy()  # [0,1]
                x_show = np.squeeze(x_show)
                if x_show.ndim == 3:
                    x_show = np.transpose(x_show, (1, 2, 0))
                progress_img.append(x_show)
                if log_process:
                    logger.info(
                        '{:>4d}, steps: {:>4d}, np.max(x_show): {:.4f}, np.min(x_show): {:.4f}'.format(seq[i], t_i,
                                                                                                       np.max(
                                                                                                           x_show),
                                                                                                       np.min(
                                                                                                           x_show)))
                if show_img:
                    util.imshow(x_show)

        # recover conditional part
        if generate_mode in ['repaint', 'DiffPIR']:
            x[mask.to(torch.bool)] = y[mask.to(torch.bool)]

        # --------------------------------
        # (4) save process
        # --------------------------------

        img_E = util.tensor2uint(x_0)

        psnr = util.calculate_psnr(img_E, img_H, border=0)  # change with your own border
        test_results['psnr'].append(psnr)

        if calc_LPIPS:
            img_H_tensor = np.transpose(img_H, (2, 0, 1))
            img_H_tensor = torch.from_numpy(img_H_tensor)[None, :, :, :].to(device)
            img_H_tensor = img_H_tensor / 255 * 2 - 1
            lpips_score = loss_fn_vgg(x_0.detach() * 2 - 1, img_H_tensor)
            lpips_score = lpips_score.cpu().detach().numpy()[0][0][0][0]
            test_results['lpips'].append(lpips_score)
            logger.info(
                '{:>10s} PSNR: {:.4f}dB LPIPS: {:.4f} ave LPIPS: {:.4f}'.format(img_name + ext,
                                                                                           psnr, lpips_score, sum(
                        test_results['lpips']) / len(test_results['lpips'])))
        else:
            logger.info('{:>10s} PSNR: {:.4f}dB'.format(img_name + ext, psnr))
            pass

        if save_E:
            estimated_path = os.path.join(E_path, img_name + '_' + model_name + ext)
            util.imsave(img_E, estimated_path)
            path_to_return['estimated'] = estimated_path

        if save_L:
            low_path = os.path.join(E_path, img_name + '_L' + ext)
            util.imsave(util.single2uint(img_L), low_path)
            path_to_return['low'] = low_path

        if save_LEH:
            leh_path = os.path.join(E_path, img_name + model_name + '_LEH' + ext)
            util.imsave(np.concatenate([util.single2uint(img_L), img_E, img_H], axis=1), leh_path)
            path_to_return['leh'] = leh_path

        if save_progressive:
            now = datetime.now()
            current_time = now.strftime("%Y_%m_%d_%H_%M_%S")
            if generate_mode in ['repaint', 'DiffPIR']:
                mask = np.squeeze(mask.cpu().numpy())
                if mask.ndim == 3:
                    mask = np.transpose(mask, (1, 2, 0))

            process_path = os.path.join(E_path,
                                        img_name + '_process_lambda_{:.3f}_{}{}'.format(lambda_, current_time, ext))
            img_total = cv2.hconcat(progress_img)
            if show_img:
                util.imshow(img_total, figsize=(80, 4))
            util.imsave(img_total * 255., process_path)
            path_to_return['process'] = process_path


            images = []
            y_t = np.squeeze((y / 2 + 0.5).cpu().numpy())
            if y_t.ndim == 3:
                y_t = np.transpose(y_t, (1, 2, 0))
            if generate_mode in ['repaint', 'DiffPIR']:
                for x in progress_img:
                    images.append((y_t) * mask + (1 - mask) * x)
                process_mask_path = os.path.join(E_path, img_name + '_process_mask_lambda_{:.3f}_{}{}'.format(lambda_,
                                                                                                              current_time,
                                                                                                              ext))
                img_total = cv2.hconcat(images)
                if show_img:
                    util.imshow(img_total, figsize=(80, 4))
                if save_progressive_mask:
                    util.imsave(img_total * 255., process_mask_path)
                    path_to_return['process_mask'] = process_mask_path



        # --------------------------------
        # Average PSNR and LPIPS
        # --------------------------------

        ave_psnr = sum(test_results['psnr']) / len(test_results['psnr'])
        logger.info('------> Average PSNR of ({}), sigma: ({:.3f}): {:.4f} dB'.format(input_image_path, noise_level_model,
                                                                                      ave_psnr))

        if calc_LPIPS:
            ave_lpips = sum(test_results['lpips']) / len(test_results['lpips'])
            logger.info('------> Average LPIPS of ({}), sigma: ({:.3f}): {:.4f}'.format(input_image_path, noise_level_model,
                                                                                        ave_lpips))

    # experiments
    lambdas = [lambda_ * i for i in range(1, 2)]
    for lambda_ in lambdas:
        # for zeta_i in [0,0.3,0.8,0.9,1.0]:
        for zeta_i in [zeta * i for i in range(1, 2)]:
            test_rho(lambda_, zeta=zeta_i)

    return path_to_return


if __name__ == '__main__':
    result_path = inpaint_service_demo()
    print(result_path)
