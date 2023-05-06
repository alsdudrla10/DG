import torch
from guided_diffusion.script_util import create_classifier
import os
import numpy as np

def get_discriminator(latent_extractor_ckpt, discriminator_ckpt, condition, img_resolution=32, device='cuda', enable_grad=True):
    classifier = load_classifier(latent_extractor_ckpt, img_resolution, device, eval=True)
    discriminator = load_discriminator(discriminator_ckpt, device, condition, eval=True)
    def evaluate(perturbed_inputs, timesteps=None, condition=None):
        with torch.enable_grad() if enable_grad else torch.no_grad():
            adm_features = classifier(perturbed_inputs, timesteps=timesteps, feature=True)
            prediction = discriminator(adm_features, timesteps, sigmoid=True, condition=condition).view(-1)
        return prediction
    return evaluate

def load_classifier(ckpt_path, img_resolution, device, eval=True):
    classifier_args = dict(
      image_size=img_resolution,
      classifier_use_fp16=False,
      classifier_width=128,
      classifier_depth=4 if img_resolution in [64, 32] else 2,
      classifier_attention_resolutions="32,16,8",
      classifier_use_scale_shift_norm=True,
      classifier_resblock_updown=True,
      classifier_pool="attention",
      out_channels=1000,
    )
    classifier = create_classifier(**classifier_args)
    classifier.to(device)
    if ckpt_path is not None:
        ckpt_path = os.getcwd() + ckpt_path
        classifier_state = torch.load(ckpt_path, map_location="cpu")
        classifier.load_state_dict(classifier_state)
    if eval:
      classifier.eval()
    return classifier

def load_discriminator(ckpt_path, device, condition, eval=False, channel=512):
    discriminator_args = dict(
      image_size=8,
      classifier_use_fp16=False,
      classifier_width=128,
      classifier_depth=2,
      classifier_attention_resolutions="32,16,8",
      classifier_use_scale_shift_norm=True,
      classifier_resblock_updown=True,
      classifier_pool="attention",
      out_channels=1,
      in_channels=channel,
      condition=condition,
    )
    discriminator = create_classifier(**discriminator_args)
    discriminator.to(device)
    if ckpt_path is not None:
        ckpt_path = os.getcwd() + ckpt_path
        discriminator_state = torch.load(ckpt_path, map_location="cpu")
        discriminator.load_state_dict(discriminator_state)
    if eval:
        discriminator.eval()
    return discriminator

def get_grad_log_ratio(discriminator, vpsde, unnormalized_input, std_wve_t, img_resolution, time_min, time_max, class_labels, log=False):
    mean_vp_tau, tau = vpsde.transform_unnormalized_wve_to_normalized_vp(std_wve_t) ## VP pretrained classifier
    if tau.min() > time_max or tau.min() < time_min or discriminator == None:
        if log:
          return torch.zeros_like(unnormalized_input), 10000000. * torch.ones(unnormalized_input.shape[0], device=unnormalized_input.device)
        return torch.zeros_like(unnormalized_input)
    else:
        input = mean_vp_tau[:,None,None,None] * unnormalized_input
    with torch.enable_grad():
        x_ = input.float().clone().detach().requires_grad_()
        if img_resolution == 64: # ADM trained UNet classifier for 64x64 with Cosine VPSDE
            tau = vpsde.compute_t_cos_from_t_lin(tau)
        tau = torch.ones(input.shape[0], device=tau.device) * tau
        log_ratio = get_log_ratio(discriminator, x_, tau, class_labels)
        discriminator_guidance_score = torch.autograd.grad(outputs=log_ratio.sum(), inputs=x_, retain_graph=False)[0]
        # print(mean_vp_tau.shape)
        # print(std_wve_t.shape)
        # print(discriminator_guidance_score.shape)
        discriminator_guidance_score *= - ((std_wve_t[:,None,None,None] ** 2) * mean_vp_tau[:,None,None,None])
    if log:
      return discriminator_guidance_score, log_ratio
    return discriminator_guidance_score

def get_log_ratio(discriminator, input, time, class_labels):
    if discriminator == None:
        return torch.zeros(input.shape[0], device=input.device)
    else:
        logits = discriminator(input, timesteps=time, condition=class_labels)
        prediction = torch.clip(logits, 1e-5, 1. - 1e-5)
        log_ratio = torch.log(prediction / (1. - prediction))
        return log_ratio

class vpsde():
    def __init__(self):
        self.beta_0 = 0.1
        self.beta_1 = 20.
        self.s = 0.008
        self.f_0 = np.cos(self.s / (1. + self.s) * np.pi / 2.) ** 2

    @property
    def T(self):
        return 1

    def compute_tau(self, std_wve_t):
        tau = -self.beta_0 + torch.sqrt(self.beta_0 ** 2 + 2. * (self.beta_1 - self.beta_0) * torch.log(1. + std_wve_t ** 2))
        tau /= self.beta_1 - self.beta_0
        return tau

    def marginal_prob(self, t):
        log_mean_coeff = -0.25 * t ** 2 * (self.beta_1 - self.beta_0) - 0.5 * t * self.beta_0
        mean = torch.exp(log_mean_coeff)
        std = torch.sqrt(1. - torch.exp(2. * log_mean_coeff))
        return mean, std

    def transform_unnormalized_wve_to_normalized_vp(self, t, std_out=False):
        tau = self.compute_tau(t)
        mean_vp_tau, std_vp_tau = self.marginal_prob(tau)
        if std_out:
            return mean_vp_tau, std_vp_tau, tau
        return mean_vp_tau, tau

    def compute_t_cos_from_t_lin(self, t_lin):
        sqrt_alpha_t_bar = torch.exp(-0.25 * t_lin ** 2 * (self.beta_1 - self.beta_0) - 0.5 * t_lin * self.beta_0)
        time = torch.arccos(np.sqrt(self.f_0) * sqrt_alpha_t_bar)
        t_cos = self.T * ((1. + self.s) * 2. / np.pi * time - self.s)
        return t_cos

    def get_diffusion_time(self, batch_size, batch_device, t_min=1e-5, importance_sampling=True):
        if importance_sampling:
            Z = self.normalizing_constant(t_min)
            u = torch.rand(batch_size, device=batch_device)
            return (-self.beta_0 + torch.sqrt(self.beta_0 ** 2 + 2 * (self.beta_1 - self.beta_0) *
                    torch.log(1. + torch.exp(Z * u + self.antiderivative(t_min))))) / (self.beta_1 - self.beta_0), Z.detach()
        else:
            return torch.rand(batch_size, device=batch_device) * (self.T - t_min) + t_min, 1

    def antiderivative(self, t, stabilizing_constant=0.):
        if isinstance(t, float) or isinstance(t, int):
            t = torch.tensor(t).float()
        return torch.log(1. - torch.exp(- self.integral_beta(t)) + stabilizing_constant) + self.integral_beta(t)

    def normalizing_constant(self, t_min):
        return self.antiderivative(self.T) - self.antiderivative(t_min)

    def integral_beta(self, t):
        return 0.5 * t ** 2 * (self.beta_1 - self.beta_0) + t * self.beta_0