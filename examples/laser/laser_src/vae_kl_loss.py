import math
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from omegaconf import II
from torch.distributions import Normal

from fairseq import metrics
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.dataclass import FairseqDataclass

eps = 1e-6


@dataclass
class VaeKLCriterionConfig(FairseqDataclass):
    sentence_avg: bool = II("optimization.sentence_avg")
    alpha: float = field(
        default=1.0,
        metadata={"help": "hyper-parameters for Beta-tcvae"},
    )
    beta: float = field(
        default=1.0,
        metadata={"help": "hyper-parameters for Beta-tcvae"},
    )
    gamma: float = field(
        default=1.0,
        metadata={"help": "hyper-parameters for Beta-tcvae"},
    )
    anneal_steps: int = field(
        default=1000,
        metadata={"help": "hyper-parameters for Beta-tcvae"},
    )


@register_criterion("vae_kl", dataclass=VaeKLCriterionConfig)
class VaeKLCriterion(FairseqCriterion):
    def __init__(self, task, sentence_avg, alpha, beta, gamma, anneal_steps):
        super().__init__(task)
        self.sentence_avg = sentence_avg
        self.training = True

        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.anneal_steps = anneal_steps

        max_capacity = 25
        Capaticy_max_iter = int(1e5)
        self.C_max = torch.Tensor([max_capacity])
        self.C_stop_iter = Capaticy_max_iter
        self.loss_type = "B"

    def forward(self, model, sample, reduce=True):
        """
        Args:
            model:
            sample: see BilanguagePairDataset, MultitaskDatasetWrapper
            reduce:
        """
        bsz = sample["source_lang_batch"]["nsentences"]
        ntokens = sample["source_lang_batch"]["ntokens"] + sample["target_lang_batch"]["ntokens"]
        sample_size = bsz

        net_out = model(
            sample["source_lang_batch"]["net_input"], sample["target_lang_batch"]["net_input"]
        )

        source_losses = self.compute_single_lang_losses(
            sample["source_lang_batch"],
            net_out["source_encoder_out"],
            net_out["source_decoder_out"],
            model.update_num,
            reduce=reduce,
        )
        target_losses = self.compute_single_lang_losses(
            sample["target_lang_batch"],
            net_out["target_encoder_out"],
            net_out["target_decoder_out"],
            model.update_num,
            reduce=reduce,
        )
        vae_loss = source_losses["loss"] + target_losses["loss"]

        kl_loss = self.compute_kl_loss(
            net_out["source_encoder_out"]["controller_out"]["mu"],
            net_out["source_encoder_out"]["controller_out"]["log_var"],
            net_out["target_encoder_out"]["controller_out"]["mu"],
            net_out["target_encoder_out"]["controller_out"]["log_var"],
        )
        kl_loss = kl_loss.sum()

        loss = vae_loss + kl_loss

        logging_output = {
            "loss": loss.data,
            "vae_loss": vae_loss.data,
            "kl_loss": kl_loss.data,
            "sample_size": sample_size,
            "ntokens": ntokens,
            "nsentences": bsz * 2,
            "source_recons": source_losses["reconstruction"].data / bsz,
            "source_KLD": source_losses["KLD"].data,
            "source_TC_loss": source_losses["TC_loss"].data,
            "source_MI_loss": source_losses["MI_loss"].data,
            "target_recons": target_losses["reconstruction"].data / bsz,
            "target_KLD": target_losses["KLD"].data,
            "target_TC_loss": target_losses["TC_loss"].data,
            "target_MI_loss": target_losses["MI_loss"].data,
        }

        return loss, sample_size, logging_output

    def loss_function(self, sample, encoder_out, decoder_out, update_num, reduce=True) -> dict:
        dataset_len = self.task.dataset_size

        mu = encoder_out["controller_out"]["mu"]
        log_var = encoder_out["controller_out"]["log_var"]
        tgt_tokens = sample["tgt_tokens"]

        # decoder_out[0]: B x T x C
        log_probs = F.log_softmax(decoder_out[0], dim=-1, dtype=torch.float32)
        log_probs = log_probs.view(-1, log_probs.size(-1))
        # B x T -> B * T,
        tgt_tokens = tgt_tokens.view(-1)
        recons_loss = F.nll_loss(
            log_probs,
            tgt_tokens,
            ignore_index=self.padding_idx,
            reduction="sum" if reduce else "none",
        )

        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1), dim=0)

        kld_weight = mu.size(0) / dataset_len

        if self.loss_type == "H":  # https://openreview.net/forum?id=Sy2fzU9gl
            loss = recons_loss + self.beta * kld_weight * kld_loss
        elif self.loss_type == "B":  # https://arxiv.org/pdf/1804.03599.pdf
            self.C_max = self.C_max.to(tgt_tokens.device)
            C = torch.clamp(self.C_max / self.C_stop_iter * self.num_iter, 0, self.C_max.data[0])
            loss = recons_loss + self.gamma * kld_weight * (kld_loss - C).abs()
        else:
            raise ValueError("Undefined loss type.")

        return {"loss": loss, "Reconstruction_Loss": recons_loss, "KLD": kld_loss}

    def compute_single_lang_losses(self, sample, encoder_out, decoder_out, update_num, reduce=True):
        """
        Args:
            model:
            sample:
            encoder_out: see LSTMEncoder
            decoder_out: see LSTMDecoder
            reduce:
        """
        dataset_len = self.task.dataset_size

        z = encoder_out["controller_out"]["z"]
        mu = encoder_out["controller_out"]["mu"]
        log_var = encoder_out["controller_out"]["log_var"]
        tgt_tokens = sample["tgt_tokens"]

        weight = 1  # kwargs['M_N']  # Account for the minibatch samples from the dataset

        # decoder_out[0]: B x T x C
        log_probs = F.log_softmax(decoder_out[0], dim=-1, dtype=torch.float32)
        log_probs = log_probs.view(-1, log_probs.size(-1))
        # B x T -> B * T,
        tgt_tokens = tgt_tokens.view(-1)
        recons_loss = F.nll_loss(
            log_probs,
            tgt_tokens,
            ignore_index=self.padding_idx,
            reduction="sum" if reduce else "none",
        )

        log_q_zx = self.log_density_gaussian(z, mu, log_var).sum(dim=1)

        zeros = torch.zeros_like(z)
        log_p_z = self.log_density_gaussian(z, zeros, zeros).sum(dim=1)

        batch_size, latent_dim = z.shape
        M_N = batch_size / dataset_len
        mat_log_q_z = self.log_density_gaussian(
            z.view(batch_size, 1, latent_dim),
            mu.view(1, batch_size, latent_dim),
            log_var.view(1, batch_size, latent_dim),
        )

        # Reference
        # https://github.com/YannDubs/disentangling-vae/blob/535bbd2e9aeb5a200663a4f82f1d34e084c4ba8d/disvae/utils/math.py#L54 # noqa
        dataset_size = (1 / M_N) * batch_size  # dataset size
        strat_weight = (dataset_size - batch_size + 1) / (dataset_size * (batch_size - 1))
        importance_weights = (
            torch.Tensor(batch_size, batch_size).fill_(1 / (batch_size - 1)).to(tgt_tokens.device)
        )
        importance_weights.view(-1)[::batch_size] = 1 / dataset_size
        importance_weights.view(-1)[1::batch_size] = strat_weight
        importance_weights[batch_size - 2, 0] = strat_weight
        log_importance_weights = (importance_weights + eps).log()

        mat_log_q_z += log_importance_weights.view(batch_size, batch_size, 1)

        log_q_z = torch.logsumexp(mat_log_q_z.sum(2), dim=1, keepdim=False)
        log_prod_q_z = torch.logsumexp(mat_log_q_z, dim=1, keepdim=False).sum(1)

        mi_loss = (log_q_zx - log_q_z).sum()
        tc_loss = (log_q_z - log_prod_q_z).sum()
        kld_loss = (log_prod_q_z - log_p_z).sum()

        if self.training:
            anneal_rate = min(0 + 1 * update_num / self.anneal_steps, 1)
        else:
            anneal_rate = 1.0

        loss = (
            recons_loss
            + self.alpha * mi_loss
            + weight * (self.beta * tc_loss + anneal_rate * self.gamma * kld_loss)
        )

        return {
            "loss": loss,
            "reconstruction": recons_loss,
            "KLD": kld_loss,
            "TC_loss": tc_loss,
            "MI_loss": mi_loss,
        }

    def log_density_gaussian(self, x: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor):
        """
        Computes the log pdf of the Gaussian with parameters mu and logvar at x
        :param x: (Tensor) Point at whichGaussian PDF is to be evaluated
        :param mu: (Tensor) Mean of the Gaussian distribution
        :param logvar: (Tensor) Log variance of the Gaussian distribution
        :return:
        """
        norm = -0.5 * (math.log(2 * math.pi) + logvar)
        log_density = norm - 0.5 * ((x - mu) ** 2 * torch.exp(-logvar))
        return log_density

    def compute_kl_loss(self, source_mu, source_log_var, target_mu, target_log_var):
        # \log \frac{\sigma_{2}}{\sigma_{1}} +
        # \frac{\sigma_{1}^{2}+\left(\mu_{1}-\mu_{2}\right)^{2}}{2 \sigma_{2}^{2}} -
        # \frac{1}{2}

        # \log \frac{\sigma_{2}}{\sigma_{1}}
        kl_1 = 0.5 * (target_log_var - source_log_var)
        # \frac{\sigma_{1}^{2}+\left(\mu_{1}-\mu_{2}\right)^{2}}{2 \sigma_{2}^{2}}
        kl_2 = (torch.exp(source_log_var) + (source_mu - target_mu) ** 2) / (
            2 * torch.exp(target_log_var) + eps
        )

        return kl_1 + kl_2 - 0.5

    def compute_js_loss(self, source_mu, source_log_var, target_mu, target_log_var):
        def get_prob(mu, log_var):
            dist = Normal(mu, torch.exp(0.5 * log_var))
            val = dist.sample()
            return dist.log_prob(val).exp()

        def kl_loss(p, q):
            return F.kl_div(p, q, reduction="batchmean", log_target=False)

        source_prob = get_prob(source_mu, source_log_var)
        target_prob = get_prob(target_mu, target_log_var)

        log_mean_prob = (0.5 * (source_prob + target_prob)).log()
        js_loss = 0.5 * (kl_loss(log_mean_prob, source_prob) + kl_loss(log_mean_prob, target_prob))
        return js_loss

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        vae_loss_sum = sum(log.get("vae_loss", 0) for log in logging_outputs)
        kl_loss_sum = sum(log.get("kl_loss", 0) for log in logging_outputs)
        sample_sum = sum(log.get("sample_size", 0) for log in logging_outputs)

        source_recons_sum = sum(log.get("source_recons", 0) for log in logging_outputs)
        source_KLD_sum = sum(log.get("source_KLD", 0) for log in logging_outputs)
        source_TC_loss_sum = sum(log.get("source_TC_loss", 0) for log in logging_outputs)
        source_MI_loss_sum = sum(log.get("source_MI_loss", 0) for log in logging_outputs)

        target_recons_sum = sum(log.get("target_recons", 0) for log in logging_outputs)
        target_KLD_sum = sum(log.get("target_KLD", 0) for log in logging_outputs)
        target_TC_loss_sum = sum(log.get("target_TC_loss", 0) for log in logging_outputs)
        target_MI_loss_sum = sum(log.get("target_MI_loss", 0) for log in logging_outputs)

        # ntokens = sum(log.get("ntokens", 0) for log in logging_outputs)
        # sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)

        # sample_size = len(logging_outputs)
        metrics.log_scalar("loss", loss_sum / sample_sum, round=3)
        metrics.log_scalar("vae_loss", vae_loss_sum / sample_sum, round=3)
        metrics.log_scalar("kl_loss", kl_loss_sum / sample_sum, round=3)

        metrics.log_scalar("source_recons", source_recons_sum / sample_sum, round=3)
        metrics.log_scalar("source_KLD", source_KLD_sum / sample_sum, round=3)
        metrics.log_scalar("source_TC_loss", source_TC_loss_sum / sample_sum, round=3)
        metrics.log_scalar("source_MI_loss", source_MI_loss_sum / sample_sum, round=3)

        metrics.log_scalar("target_recons", target_recons_sum / sample_sum, round=3)
        metrics.log_scalar("target_KLD", target_KLD_sum / sample_sum, round=3)
        metrics.log_scalar("target_TC_loss", target_TC_loss_sum / sample_sum, round=3)
        metrics.log_scalar("target_MI_loss", target_MI_loss_sum / sample_sum, round=3)

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True
