from torchvision.utils import make_grid

def grid_of_imgs(images):
    images = (images + 1) / 2 # [0,1] range 
    return make_grid(images, nrow=4).cpu().permute(1,2,0)
