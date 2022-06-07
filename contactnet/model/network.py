# Pytorch related imports
import torch
from neptune.new.types import File

# We use the implementation of Pix2Pix from lightning bolts as base model
from pl_bolts.models.gans import Pix2Pix

# Some utilities
from contactnet.model.utils import grid_of_imgs

class ContactNet(Pix2Pix):
    def __init__(self,
        in_channels,
        out_channels,
        learning_rate=0.0002,
        lambda_recon=200,
        log_visuals_rate=5):
        super().__init__(in_channels=in_channels, out_channels=out_channels,
                         learning_rate=learning_rate, lambda_recon=lambda_recon)

        self.log_visuals_rate = log_visuals_rate

    def configure_optimizers(self):
        return super().configure_optimizers()

    def gen_forward(self, inp):
        return self.gen(inp)

    def _gen_step(self, real_images, conditioned_images, mask):
        fake_images = self.gen(conditioned_images)

        # We mask the generated images to focus the network on the object
        fake_images = fake_images * mask + real_images * (1 - mask)

        disc_logits = self.patch_gan(fake_images, conditioned_images)
        adversarial_loss = self.adversarial_criterion(disc_logits, torch.ones_like(disc_logits))

        # calculate reconstruction loss
        recon_loss = self.recon_criterion(fake_images, real_images)
        lambda_recon = self.hparams.lambda_recon

        return adversarial_loss + lambda_recon * recon_loss

    def _disc_step(self, real_images, conditioned_images, mask):
        fake_images = self.gen(conditioned_images).detach()
        
        # We mask the generated images to focus the network on the object
        fake_images = fake_images * mask + real_images * (1 - mask)

        fake_logits = self.patch_gan(fake_images, conditioned_images)
        real_logits = self.patch_gan(real_images, conditioned_images)

        fake_loss = self.adversarial_criterion(fake_logits, torch.zeros_like(fake_logits))
        real_loss = self.adversarial_criterion(real_logits, torch.ones_like(real_logits))
        return (real_loss + fake_loss) / 2

    def _common_step(self, batch, batch_idx, optimizer_idx):
        condition, real, mask = batch

        loss = None
        if optimizer_idx == 0:
            loss = self._disc_step(real, condition, mask)
            self.logger.experiment["train/discriminator_loss"].log(loss)
        elif optimizer_idx == 1:
            loss = self._gen_step(real, condition, mask)
            self.logger.experiment["train/generator_loss"].log(loss)
            self.log('gen_loss', loss)

        return loss

    def training_step(self, batch, batch_idx, optimizer_idx):
        return self._common_step(batch, batch_idx=batch_idx, optimizer_idx=optimizer_idx)

    def validation_step(self, batch, batch_idx):
        if self.current_epoch % self.log_visuals_rate == 0:
            condition, _, mask = batch
            output = self.gen_forward(condition)
            output = output * mask + (1 - mask)
            output = grid_of_imgs(output)
            self.logger.experiment["visuals/"].log(File.as_image(output))
