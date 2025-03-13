import argparse
import os
from utils import get_args

import torch

from networks import Discriminator, Generator
import torch.nn.functional as F
from train import train_model



def compute_discriminator_loss(
    discrim_real, discrim_fake, discrim_interp, interp, lamb
):
    ##################################################################
    # TODO 1.3: Implement GAN loss for discriminator.
    # Do not use discrim_interp, interp, lamb. They are placeholders
    # for Q1.5.
    ##################################################################
    cpu_discrim_fake=discrim_fake.to(device="cpu")
    cpu_discrim_real=discrim_real.to(device="cpu")
    true_labels=torch.ones_like(discrim_real)
    false_labels=torch.zeros_like(discrim_real)
    bce=torch.nn.BCEWithLogitsLoss()
    loss=(bce(discrim_real,true_labels)+bce(discrim_fake,false_labels))/2
    #loss = -(torch.log(discrim_real)+torch.log(torch.ones_like(discrim_fake)-discrim_fake)).mean()
    ##################################################################
    #                          END OF YOUR CODE                      #
    #################################################################
    return loss


def compute_generator_loss(discrim_fake):
    ##################################################################
    # TODO 1.3: Implement GAN loss for the generator.
    ##################################################################
    bce_gen=torch.nn.BCEWithLogitsLoss()
    # cpu_discrim_fake=discrim_fake.to(device="cpu")   
    true_labels=torch.ones_like(discrim_fake)
    loss=bce_gen(discrim_fake,true_labels)
    #loss = torch.log(torch.ones_like(discrim_fake)-discrim_fake).mean()
    ##################################################################
    #                          END OF YOUR CODE                      #
    ##################################################################
    return loss


if __name__ == "__main__":
    args = get_args()
    gen = Generator().cuda()
    disc = Discriminator().cuda()
    prefix = "data_gan/"
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
