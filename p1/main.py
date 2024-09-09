# CS180 (CS280A): Project 1

import numpy as np
import skimage as sk
import skimage.io as skio

from pathlib import Path

path = Path.cwd()

# cropping helper function
def crop(im, ratio):
    h, w = im.shape[0], im.shape[1]
    h_crop = int(h * ratio)
    w_crop = int(w * ratio)
    
    return im[h_crop:h - h_crop, w_crop:w - w_crop]


# exhaustive search algorithm
def align(im1, im2, dispacement_window_size=15, border_ratio=0.1):
    im1_cropped = crop(im1, border_ratio)
    im2_cropped = crop(im2, border_ratio)

    dispacement_window = range(-dispacement_window_size, dispacement_window_size + 1)
    
    score_min = float('inf')
    displacement = (0, 0)

    for x in dispacement_window:
        for y in dispacement_window:
            im1_rolled = np.roll(im1_cropped, (x, y), axis=(0, 1))
            score = np.sqrt(np.sum(np.sum((im1_rolled - im2_cropped) ** 2)))
            if score < score_min:
                score_min = score
                displacement = (x, y)

    # print(f"{h}, {w}")
    # print(score_min)
    return displacement


# image pyramid
def subsample(im1, im2, dispacement_window_size=4, layer=1, factor=2):
    h, w = im1.shape
    if w <= 500:
        return align(im1, im2)

    im1_rescaled = sk.transform.rescale(im1, 1 / factor)
    im2_rescaled = sk.transform.rescale(im2, 1 / factor)

    # print(f"{h}, {w}")
    displacement = subsample(im1_rescaled, im2_rescaled, dispacement_window_size, layer + 1)
    displacement_rescaled = (displacement[0] * factor, displacement[1] * factor)

    # print(displacement_rescaled)
    im1_rolled = np.roll(im1, displacement_rescaled, axis=(0, 1))
    displacement = align(im1_rolled, im2, dispacement_window_size * layer)
    return (displacement[0] + displacement_rescaled[0], displacement[1] + displacement_rescaled[1])


# automatic contrasting
def contrast_auto(im, linear=False):
    im_cropped = crop(im, 0.05)

    # for i in range(im_cropped.shape[2]):
    #     pixels = im_cropped[:, :, i]
    #     min_pixel = pixels.min()
    #     diff = pixels.max() - min_pixel
    #     # print(min_pixel)
    #     # print(diff)
    #     im[:, :, i] = np.maximum(np.minimum((im[:, :, i] - min_pixel) / diff, 1), 0)

    # return im

    # pixels = im_cropped.sum(axis=2)
    # min_pixel = np.unravel_index(np.argmin(pixels), pixels.shape)
    # max_pixel = np.unravel_index(np.argmax(pixels), pixels.shape)

    # min_value = np.min(im_cropped[min_pixel])
    # delta = np.max(im_cropped[max_pixel]) - min_value

    min_value = np.min(im_cropped)
    delta = np.max(im_cropped) - min_value

    im = (im - min_value) / delta

    if not linear:
        im = np.sin((im - 0.5) * np.pi) / 2 + 0.5

    im = np.maximum(np.minimum(im, 1), 0)

    return im


# automatic cropping
def crop_auto(im, upper_threshold=0.945, lower_threshold=0.12):
    h, w = im.shape[0], im.shape[1]

    for j in range(0, w):
        pixel_sum = im[:, j, :].sum(axis=0) / h
        if not (np.any(pixel_sum < lower_threshold) or np.any(pixel_sum > upper_threshold)):
            break

    im = im[:, j:, :]
    h, w = im.shape[0], im.shape[1]

    for j in range(w - 1, 0, -1):
        pixel_sum = im[:, j, :].sum(axis=0) / h
        if not (np.any(pixel_sum < lower_threshold) or np.any(pixel_sum > upper_threshold)):
            break

    im = im[:, :j, :]
    h, w = im.shape[0], im.shape[1]

    for i in range(0, h):
        pixel_sum = im[i, :, :].sum(axis=0) / w
        if not (np.any(pixel_sum < lower_threshold) or np.any(pixel_sum > upper_threshold)):
            break
    
    im = im[i:, :, :]
    h, w = im.shape[0], im.shape[1]

    for i in range(h - 1, 0, -1):
        pixel_sum = im[i, :, :].sum(axis=0) / w
        if not (np.any(pixel_sum < lower_threshold) or np.any(pixel_sum > upper_threshold)):
            break

    im = im[:i, :, :]

    return im
    

# get all images under the same folder
files = [f for f in Path('.').iterdir() if f.is_file() and f.suffix in [".jpg", ".tif"]]
# files = ["custom_stantsiia_soroka.tif"]
print(files)

Path('./output').mkdir(parents=True, exist_ok=True)
Path('./output_crop').mkdir(parents=True, exist_ok=True)
Path('./output_crop_contrast_linear').mkdir(parents=True, exist_ok=True)
Path('./output_crop_contrast_nonlinear').mkdir(parents=True, exist_ok=True)

for imname in files:
    # if not "custom" in str(imname):
    #     continue

    # name of the input file
    print(imname)

    # read in the image
    im = skio.imread(imname)

    # convert to double (might want to do this later on to save memory)    
    im = sk.img_as_float(im)
        
    # compute the height of each part (just 1/3 of total)
    height = np.floor(im.shape[0] / 3.0).astype(int)

    # separate color channels
    b = im[:height]
    g = im[height: 2*height]
    r = im[2*height: 3*height]

    # align the images
    # functions that might be useful for aligning the images include:
    # np.roll, np.sum, sk.transform.rescale (for multiscale)

    g_d = subsample(g, b)
    r_d = subsample(r, b)

    ag = np.roll(g, g_d, axis=(0, 1))
    ar = np.roll(r, r_d, axis=(0, 1))

    # create a color image
    im_out = np.dstack([ar, ag, b])

    
    fname = f'{path}\output\{imname}_out.jpg'
    skio.imsave(fname, (im_out * 255).astype(np.uint8))

    # automatically crop the image
    im_out_crop = crop_auto(im_out, upper_threshold=0.945, lower_threshold=0.12)

    
    fname = f'{path}\output_crop\{imname}_out.jpg'
    skio.imsave(fname, (im_out_crop * 255).astype(np.uint8))

    # automatically adjust the contrast
    im_out_crop_linear = contrast_auto(im_out_crop, linear=True)

    fname = f'{path}\output_crop_contrast_linear\{imname}_out.jpg'
    skio.imsave(fname, (im_out_crop_linear * 255).astype(np.uint8))

    im_out_crop_nonlinear = contrast_auto(im_out_crop, linear=False)
    
    fname = f'{path}\output_crop_contrast_nonlinear\{imname}_out.jpg'
    skio.imsave(fname, (im_out_crop_nonlinear * 255).astype(np.uint8))

    # save the image

    # print the displacements
    print(f"r_d={str(r_d)}")
    print(f"g_d={str(g_d)}")

    # # display the image
    # skio.imshow(im_out)
    # skio.show()
