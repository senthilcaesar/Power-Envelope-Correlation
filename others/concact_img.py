from PIL import Image

def get_concat_h(im1, im2, im3, im4, im5, im6):
    dst = Image.new('RGB', (im1.width + im2.width + im3.width + im4.width + im5.width + im6.width, im1.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (im1.width*1, 0))
    dst.paste(im3, (im2.width*2, 0))
    dst.paste(im4, (im3.width*3, 0))
    dst.paste(im5, (im4.width*4, 0))
    dst.paste(im6, (im5.width*5, 0))
    return dst

def get_concat_v(im1, im2):
    dst = Image.new('RGB', (im1.width, im1.height + im2.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (0, im1.height))
    return dst

age1 = '18to35'
im1 = Image.open(f'linkplot_4Hz_{age1}.png')
im2 = Image.open(f'linkplot_8Hz_{age1}.png')
im3 = Image.open(f'linkplot_12Hz_{age1}.png')
im4 = Image.open(f'linkplot_16Hz_{age1}.png')
im5 = Image.open(f'linkplot_20Hz_{age1}.png')
im6 = Image.open(f'linkplot_24Hz_{age1}.png')
get_concat_h(im1, im2, im3, im4, im5, im6).save(f'{age1}_horizontal.png')
del im1, im2, im3, im4, im5, im6

age2 = '65to88'
im1 = Image.open(f'linkplot_4Hz_{age2}.png')
im2 = Image.open(f'linkplot_8Hz_{age2}.png')
im3 = Image.open(f'linkplot_12Hz_{age2}.png')
im4 = Image.open(f'linkplot_16Hz_{age2}.png')
im5 = Image.open(f'linkplot_20Hz_{age2}.png')
im6 = Image.open(f'linkplot_24Hz_{age2}.png')
get_concat_h(im1, im2, im3, im4, im5, im6).save(f'{age2}_horizontal.png')
del im1, im2, im3, im4, im5, im6

import PIL.Image
PIL.Image.MAX_IMAGE_PIXELS = None
im1 = Image.open(f'{age1}_horizontal.png')
im2 = Image.open(f'{age2}_horizontal.png')
get_concat_v(im1, im2).save('final.jpg')
