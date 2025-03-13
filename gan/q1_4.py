import os

import torch
import torch.nn.functional as F
from utils import get_args

from networks import Discriminator, Generator
from train import train_model


def compute_discriminator_loss(
    discrim_real, discrim_fake, discrim_interp, interp, lamb
):
    ##################################################################
    # TODO: 1.4: Implement LSGAN loss for discriminator.
    # Do not use discrim_interp, interp, lamb. They are placeholders
    # for Q1.5.
    ##################################################################
    true_labels=torch.ones_like(discrim_real)
    false_labels=torch.zeros_like(discrim_real)
    mse_dis=torch.nn.MSELoss()
    loss = (mse_dis(discrim_real,true_labels)+mse_dis(discrim_fake,false_labels))/2
    ##################################################################
    #                          END OF YOUR CODE                      #
    ##################################################################
    return loss


def compute_generator_loss(discrim_fake):
    ##################################################################
    # TODO: 1.4: Implement LSGAN loss for generator.
    ##################################################################
    true_labels=torch.ones_like(discrim_fake)
    mse_gen=torch.nn.MSELoss()
    loss = mse_gen(discrim_fake,true_labels)
    ##################################################################
    #                          END OF YOUR CODE                      #
    ##################################################################
    return loss

if __name__ == "__main__":
    args = get_args()
    gen = Generator().cuda()
    disc = Discriminator().cuda()
    prefix = "data_ls_gan/"
    os.makedirs(prefix, exist_ok=True)

    train_model(
        gen,
        disc,
        num_iterations=int(3e4),
        batch_size=256,
        prefix=prefix,
        gen_loss_fn=compute_generator_loss,
        disc_loss_fn=compute_discriminator_loss,
        log_period=1000,
        amp_enabled=not args.disable_amp,
    )
