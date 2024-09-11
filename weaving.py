import argparse
import matplotlib.image as img
import numpy as np
import math
import sys
from PIL import Image

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])

def convert_image_to_set(image_path, num_layers, loom_width=3520):

    img = Image.open(image_path)
    img_gr = img.convert('L')

    nrow, ncol = img.size
    a_rat = loom_width / float(ncol)
    img_rs = img_gr.resize((int(a_rat * nrow), loom_width), Image.NEAREST)

    return np.vectorize(lambda x:int(num_layers * x/256))(np.array(img_rs).T)

def shaded_satin_square(num_ends=16, weft_face_cnt=4, weft_build=False):
    
    step_sz = 3
    struct_sz = (num_ends,num_ends)

    x,y,z = step_sz, weft_face_cnt, num_ends

    ss = np.resize([int((i + x*j + y) % z < y) for i in range(z) for j in range(z)], struct_sz)
    if weft_build:
        ss = ss.transpose()
    return ss

# satin_tuple = (1,7) => 8x8 structure
def shaded_satin_layer(pick_sz, warp_sz, satin_tuple):

    num_ends = sum(satin_tuple)
    ss = shaded_satin_square(num_ends, satin_tuple[0])

    mask = np.tile(ss.transpose(), math.ceil(pick_sz / num_ends)).transpose()
    mask = np.tile(mask, math.ceil(warp_sz / num_ends))

    return mask[:pick_sz,:warp_sz]

def append_middle_layer(weft_face_cnt, num_layers, num_ends=16):
    if len(weft_face_cnt) < num_layers:
        weft_face_cnt = sorted(weft_face_cnt + [int(num_ends/2)])
    return weft_face_cnt

def generate_ratios(num_layers, num_ends=None):

    if not num_ends:
        num_ends = 2 * num_layers

    ratios = list()
    for idx in range(1, int(num_ends/2), math.floor(num_ends / (num_layers-1))):
        ratios += [idx, num_ends - idx]
    ratios = sorted(ratios)

    ratios = append_middle_layer(ratios, num_layers, num_ends)
    return [(i, num_ends - i) for i in ratios]

def apply_layers(image_set, layers):

    LRG_PRM = 422353
    image_set[image_set == 0] = LRG_PRM

    res = np.zeros(image_set.shape)

    for idx in range(len(layers)):
        if idx == 0:
            mask = np.vectorize(lambda x:int(x == LRG_PRM))(image_set)
        else:
            mask = np.vectorize(lambda x:int(x == idx))(image_set)
        res += np.multiply(mask, layers[idx])

    return res.astype(np.uint8)

def convert_to_shaded_satin(image_path, num_layers=6, loom_width=3520):

    ratios = generate_ratios(num_layers)

    image_set = convert_image_to_set(image_path, num_layers)
    num_picks, num_warp_ends = image_set.shape

    satins = [shaded_satin_layer(num_picks, num_warp_ends, ratio) for ratio in ratios]

    return apply_layers(image_set, satins)

def plain_weave(num_picks, num_warp_ends):

    res = np.zeros(num_picks * num_warp_ends, dtype=np.uint8)
    res[np.arange(0,res.size,2)] = 1
    res = res.reshape(num_picks, num_warp_ends)
    res[1::2] = res[1::2] ^ 1
    return res

def interleave_columns(a,b):
    assert(a.shape == b.shape)
    return np.dstack((a,b)).reshape(a.shape[0],-1)

def double_warp_mask(a, ratio):

    x,y = ratio

    if x < y:
        mask = (np.arange(a.size) % (x+y) <  x).reshape(a.shape)
    else:
        mask = (np.arange(a.size) % (x+y) >= y).reshape(a.shape)

    mask = mask.astype(np.uint8)
    return a * mask

def double_cloth_alternating(a, b, ratio_a, ratio_b):

    assert(a.shape == b.shape)

    mask_a = double_warp_mask(b, ratio_a)
    mask_b = double_warp_mask(a, ratio_b)

    a_dbl = interleave_columns(a, mask_a)
    b_dbl = interleave_columns(mask_b, b)

    return interleave_columns(a_dbl.transpose(), b_dbl.transpose()).reshape(a_dbl.shape[1],-1).transpose()

def plain_weave_double_warp_layer(num_picks, num_warp_ends, ratio_a, ratio_b):

    a = plain_weave(num_picks, num_warp_ends)
    b = a ^ 1

    return double_cloth_alternating(a, b, ratio_a, ratio_b)

def plain_weave_double_warp(image_path, num_layers=5, loom_width=3520):

    #ratios = generate_ratios(num_layers, num_ends = num_layers)
    ratios = [(0,16),(4,12),(8,8),(12,4),(16,0)]

    image_set = convert_image_to_set(image_path, num_layers, loom_width)

    num_picks, num_warp_ends = image_set.shape
    num_warp_ends //= 2

    image_set = np.repeat(image_set, repeats=2, axis=0)

    plain_layers = [plain_weave_double_warp_layer(num_picks, num_warp_ends, ratio, ratio[::-1]) for ratio in ratios]

    return apply_layers(image_set, plain_layers)

def game_of_life(grid):

    print("Running Game Of Life")
    grid = grid.astype(int)
    newGrid = np.zeros(grid.shape)
    rows,cols = grid.shape

    for i in range(rows):
        for j in range(cols):
            total = grid[(i-1) % rows:(i+2) % rows, (j-1) % cols:(j+2) % cols].sum() - grid[i, j]
            if grid[i, j] == 1:
                newGrid[i,j] = int(not ((total < 2) or (total > 3)))
            elif total == 3:
                newGrid[i, j] = 1

    newGrid = newGrid.astype(np.uint8)
    return newGrid

def write_to_file(weave_array, file_path):

    im = Image.new('1', weave_array.shape)

    rows,cols = im.size

    pixels = im.load()
    for i in range(rows):
        for j in range(cols):
            pixels[i,j] = weave_array[i,j],

    im.save(file_path)

def main(args):

    parser = argparse.ArgumentParser()    
    
    parser.add_argument('-i','--input', help='input file path')
    parser.add_argument('--output-shaded-satin', default=None, help='outut file path')
    parser.add_argument('--output-double-warp', default=None, help='outut file path')
    parser.add_argument('-g','--game-of-life', help='game of life iteration cound [default=0]', type=np.uint8, default=0)

    args = parser.parse_args()

    if args.output_shaded_satin:
        weaving_ss = convert_to_shaded_satin(args.input, 4)
        write_to_file(weaving_ss, args.output_shaded_satin)
   
    if args.output_double_warp:
        weaving_dbl_warp = plain_weave_double_warp(args.input, 3)
        write_to_file(weaving_dbl_warp, args.output_double_warp)

if __name__ == '__main__':

    main(sys.argv[1:])
