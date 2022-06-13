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

def get_concat_v(im1, im2, im3, im4, im5, im6):
    dst = Image.new('RGB', (im1.width, im1.height + im2.height + im3.height + im4.height + im5.height + im6.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (0, im1.height*1))
    dst.paste(im3, (0, im2.height*2))
    dst.paste(im4, (0, im3.height*3))
    dst.paste(im5, (0, im4.height*4))
    dst.paste(im6, (0, im5.height*5))
    return dst

im1 = Image.open('1lh1.jpg')
im2 = Image.open('2lh1.jpg')
im3 = Image.open('3lh1.jpg')
im4 = Image.open('4lh1.jpg')
im5 = Image.open('5lh1.jpg')
im6 = Image.open('6lh1.jpg')
get_concat_h(im1, im2, im3, im4, im5, im6).save('lh1_horizontal.png')
del im1, im2, im3, im4, im5, im6

im1 = Image.open('1lh2.jpg')
im2 = Image.open('2lh2.jpg')
im3 = Image.open('3lh2.jpg')
im4 = Image.open('4lh2.jpg')
im5 = Image.open('5lh2.jpg')
im6 = Image.open('6lh2.jpg')
get_concat_h(im1, im2, im3, im4, im5, im6).save('lh2_horizontal.png')
del im1, im2, im3, im4, im5, im6

im1 = Image.open(f'1lh3.jpg')
im2 = Image.open(f'2lh3.jpg')
im3 = Image.open(f'3lh3.jpg')
im4 = Image.open(f'4lh3.jpg')
im5 = Image.open(f'5lh3.jpg')
im6 = Image.open(f'6lh3.jpg')
get_concat_h(im1, im2, im3, im4, im5, im6).save('lh3_horizontal.png')
del im1, im2, im3, im4, im5, im6

im1 = Image.open('1rh1.jpg')
im2 = Image.open('2rh1.jpg')
im3 = Image.open('3rh1.jpg')
im4 = Image.open('4rh1.jpg')
im5 = Image.open('5rh1.jpg')
im6 = Image.open('6rh1.jpg')
get_concat_h(im1, im2, im3, im4, im5, im6).save('rh1_horizontal.png')
del im1, im2, im3, im4, im5, im6

im1 = Image.open('1rh2.jpg')
im2 = Image.open('2rh2.jpg')
im3 = Image.open('3rh2.jpg')
im4 = Image.open('4rh2.jpg')
im5 = Image.open('5rh2.jpg')
im6 = Image.open('6rh2.jpg')
get_concat_h(im1, im2, im3, im4, im5, im6).save('rh2_horizontal.png')
del im1, im2, im3, im4, im5, im6

im1 = Image.open('1rh3.jpg')
im2 = Image.open('2rh3.jpg')
im3 = Image.open('3rh3.jpg')
im4 = Image.open('4rh3.jpg')
im5 = Image.open('5rh3.jpg')
im6 = Image.open('6rh3.jpg')
get_concat_h(im1, im2, im3, im4, im5, im6).save('rh3_horizontal.png')
del im1, im2, im3, im4, im5, im6

import PIL.Image
PIL.Image.MAX_IMAGE_PIXELS = None
im1 = Image.open('lh1_horizontal.png')
im2 = Image.open('lh2_horizontal.png')
im3 = Image.open('lh3_horizontal.png')
im4 = Image.open('rh2_horizontal.png')
im5 = Image.open('rh1_horizontal.png')
im6 = Image.open('rh3_horizontal.png')
get_concat_v(im1, im2, im3, im4, im5, im6).save('cluster_corr.png')
