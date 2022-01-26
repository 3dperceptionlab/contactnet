from pl_bolts.models.gans import Pix2Pix

class ContactNet(Pix2Pix):
    def __init__(self, in_channels, out_channels, learning_rate=0.0002):
        super().__init__(in_channels=in_channels, out_channels=out_channels, learning_rate=learning_rate)

    def forward(self, x):
        return self.gen(x)

    def validation_step(self, batch, batch_idx):
        raise NotImplementedError

    def test_step(self, batch, batch_idx):
        raise NotImplementedError
