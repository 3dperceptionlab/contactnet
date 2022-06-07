import os
from PIL import Image
from torchvision import transforms

# Pytorch imports
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import NeptuneLogger

# DataModule
from contactnet.datasets.syncontact import SynContactDataModule

# SynContact model
from contactnet.model.network import ContactNet


def _forward():
    # transforms
    resize = transforms.Resize(256)
    to_tensor = transforms.ToTensor()
    to_pil = transforms.ToPILImage()
    
    #load model
    model = ContactNet.load_from_checkpoint("./checkpoints/predictive-epoch=264-val_loss=0.0000.ckpt")
    model.eval()
    
    # load images
    paths_to_imgs = []
    for _root, _, _files in os.walk('/src/datasets/syncontact/test'):
        for _file in _files:
            if _file.endswith("rgb.png"):
                paths_to_imgs.append((os.path.join(_root, "rgb.png"),
                                     (os.path.join(_root, "mask.png"))))
    
    num = 0    
    for img_pth, mask_pth in paths_to_imgs:
        _img = Image.open(img_pth)
        _mask = Image.open(mask_pth)
        _mask = to_tensor(resize(_mask)).unsqueeze(0)
        _img = to_tensor(resize(_img)).unsqueeze(0)
        
        # [-1,1]
        _img = _img * 2 - 1
        
        _prediction = model.gen_forward(_img) * _mask + (1 - _mask)
        
        # save prediction
        # to [0,1] range
        _prediction = (_prediction + 1) / 2
        # permute axes
        #_prediction = _prediction[0].permute(1,2,0)
        # toPIL
        pil_img = to_pil(_prediction[0])
        
        #save to disk
        pil_img.save('./unseen/' + str(num) + ".png")
        
        num += 1
        
        
        
        
        
        
                

def _main():
    # Neptune logging
    try:
        # Neptune credentials. User-specific. You need to create you Neptune accout!
        import contactnet.neptune_credentials as nc
        neptune = NeptuneLogger(api_key=nc.key,
                                project=nc.project,
                                log_model_checkpoints=False)
    except ImportError: # no neptune credentials, no logger
        print("No Neptune logging")
        neptune = NeptuneLogger()

    chkpoint_cb = ModelCheckpoint(monitor='gen_loss', dirpath='./checkpoints',
                                  filename='predictive-{epoch:02d}-{val_loss:.4f}',
                                  mode='min')

    model = ContactNet(3,3)
    syncontact_dm = SynContactDataModule()
    syncontact_dm.prepare_data()

    trainer = Trainer(gpus=1, logger=neptune, log_every_n_steps=10, max_epochs=1000, callbacks=[chkpoint_cb])
    trainer.fit(model, datamodule=syncontact_dm)
