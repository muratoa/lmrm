import argparse
import matplotlib.image as img
import numpy as np
import math
import random
import sys
from PIL import Image

KEY = ["",""]
KEY[0] = "RGRGRGBG"
KEY[1] = "BGRGRGRG"

WARP_FACE_0 = KEY[0]*8
WARP_FACE_1 = KEY[1]*8
WEFT_FACE = ''.join([_*8 for _ in KEY[0]])

def convolution(input_matrix, kernel):
    input_h, input_w = input_matrix.shape
    kernel_h, kernel_w = kernel.shape
    
    output_h = input_h - kernel_h + 1
    output_w = input_w - kernel_w + 1
                                    
    output = np.zeros((output_h, output_w))
    
    for i in range(output_h):
        for j in range(output_w):
            region = input_matrix[i:i+kernel_h, j:j+kernel_w]
            output[i, j] = np.sum(region * kernel)

    return output

def blur(weave_a, dim=16):

    kernel = np.ones((dim,dim))
    for i in range(dim):
        for j in range(dim):
            kernel[i,j] *= (i+j)**2
    kernel /= dim**2

    return convolution(weave_a, kernel)

def double_cloth_interleave(weave_a, weave_b):

    assert(weave_a.shape == weave_b.shape)
    weave_a = weave_a.astype(np.bool)
    weave_b = ~(weave_b.astype(np.bool))

    n_picks,n_ends = weave_a.shape
    weave_c = np.zeros((2*n_picks,2*n_ends))

    for i in range(2*n_picks):

        odd_pick = (i % 2)
        pick = weave_a[i//2,] if not odd_pick else weave_b[i//2,]

        for j in range(2*n_ends):

            odd_end = (j % 2)

            if odd_pick ^ odd_end:
                weave_c[i,j] = odd_pick
            else:
                weave_c[i,j] = pick[j//2]

    return weave_c

def rescale_picks(strct_a, strct_b):
    nr_a = strct_a.shape[0]
    nr_b = strct_b.shape[0]
    return np.tile(strct_b, (int(np.ceil(nr_a / nr_b)), 1))

def rescale_ends(strct_a, strct_b):
    nc_a = strct_a.shape[1]
    nc_b = strct_b.shape[1]
    return np.tile(strct_b, (1, int(np.ceil(nc_a / nc_b))))

def resize_structures_to_match(strct_a, strct_b):

    nr_a,nc_a = strct_a.shape
    nr_b,nc_b = strct_b.shape

    if nr_a != nr_b:
        if nr_a > nr_b:
            strct_b = rescale_picks(strct_a, strct_b)
        else:
            strct_a = rescale_picks(strct_b, strct_a)

    if nc_a != nc_b:
        if nc_a > nc_b:
            strct_b = rescale_ends(strct_a, strct_b)
        else:
            strct_a = rescale_ends(strct_b, strct_a)

    mx_r = min(strct_a.shape[0], strct_b.shape[0])
    mx_c = min(strct_a.shape[1], strct_b.shape[1])
    return strct_a[:mx_r,:mx_c], strct_b[:mx_r,:mx_c]

def make_double_cloth(strct_a, strct_b):

    weave_a, weave_b = resize_structures_to_match(strct_a, strct_b)

    return double_cloth_interleave(weave_a, weave_b)

def plain_sc(nrow,ncol):
    return np.tile(np.array([[1,0,1,0],[0,1,0,1],[1,0,1,0],[0,1,0,1]]), (int(nrow/4),int(ncol/4)))

def plain_sc_extend(rep):
    return np.array([[1]+[0]*rep+[1]+[0]*rep,[0]*rep+[1]+[0]*rep+[1],
                     [1]+[0]*rep+[1]+[0]*rep,[0]*rep+[1]+[0]*rep+[1]])

def plain_dc(nrow,ncol):
    return np.tile(np.array([[1,0,0,0],[1,1,1,0],[0,0,1,0],[1,0,1,1]]), (int(nrow/4),int(ncol/4)))

def recursive_weav_dbl(wv_s, r_cnt):
    for _ in range(r_cnt):
        wv_s = make_double_cloth(wv_s, wv_s)
    return wv_s


def twill_dc(out, a_pat, b_pat):

    for i in range(out.shape[1]):
        out[2*i] = a_pat
        out[2*i+1] = b_pat
        a_pat = np.roll(a_pat,-1)
        b_pat = np.roll(b_pat,-1)

    nrep = int(np.ceil(28 / out.shape[1]))
    return np.tile(out, (nrep,nrep))[:56,:28]

def C1():
    out = np.zeros((8,4))

    a_pat = [0,1,1,1]
    b_pat = [0,1,0,1]

    return twill_dc(out, a_pat, b_pat)

def C2():
    out = np.zeros((24,12))

    a_pat = [0,1,1,1]*3
    b_pat = [0,0,1,0,0,1,0,0,1,0,0,1]

    return twill_dc(out, a_pat, b_pat)

def C3():
    out = np.zeros((8,4))

    a_pat = [1,0,1,1]
    b_pat = [0,0,0,1]

    return twill_dc(out, a_pat, b_pat)

def C4():
    out = np.zeros((40,20))

    a_pat = [1,0,1,1]*5
    b_pat = [0,0,0,0,1]*4

    return twill_dc(out, a_pat, b_pat)

def C5():
    out = np.zeros((24,12))

    a_pat = [0,1,1,1]*3
    b_pat = [0,0,0,0,0,1]*2

    return twill_dc(out, a_pat, b_pat)

def C6():
    out = np.zeros((56,28))

    a_pat = [1,0,1,1]*7
    b_pat = [0,0,0,0,0,0,1]*4

    return twill_dc(out, a_pat, b_pat)

def cornell_sample():

    cct = [[C1(),C2(),C3(),C4(),C5(),C6()], 
           [np.concat([_,_],axis=1) for _ in [C1(),C2(),C3(),C4(),C5(),C6()]],
           [np.concat([_,_,_],axis=1) for _ in [C1(),C2(),C3(),C4(),C5(),C6()]]]
    pw = plain_dc(56,16)

    s_type = [[0,1,2,3,4,5],[5,4,3,2,1,0],[0,2,4,1,3,5],[3,1,5,0,4,2]]
    r_type = [[0,1,2,2,1,0],[0,2,1,1,2,0],[2,1,0,0,1,2],[1,0,2,2,0,1]]
    
    weav = np.array([])
    for i in range(4):

        subm=plain_dc(56,8)
        for j in range(6):
            rr = r_type[i][j]
            ss = s_type[i][j]
            tmp = np.hstack([cct[rr][ss],pw])
            subm = np.hstack([subm,tmp]) if subm.size else tmp

        weav = np.vstack([weav,subm]) if weav.size else subm
    
    weav2 = np.array([])
    ii=[2,3,0,1]
    ii2=[1,0,3,2]
    for i in range(4):

        subm=plain_dc(56,8)
        for j in range(6):
            rr = r_type[ii[i]][j]
            ss = s_type[ii2[i]][j]
            tmp = np.hstack([cct[rr][ss],pw])
            subm = np.hstack([subm,tmp]) if subm.size else tmp

        weav2 = np.vstack([weav2,subm]) if weav2.size else subm
    
    return np.hstack([weav,weav2])

def max_color_count(warp_ends_key, val):
    sz = len(warp_ends_key)
    cnt = warp_ends_key.count(val)
    return cnt * sz + (sz - cnt) * cnt

TOTR = max_color_count(KEY[0],"R")
TOTG = max_color_count(KEY[0],"G")
TOTB = max_color_count(KEY[0],"B")

def create_color_count_cache(warp_face_key, weft_face_key):
    num_rr = [0]*64 
    num_gg = [0]*64 
    num_bb = [0]*64

    union_rr = [warp_face_key[i] == 'R' or weft_face_key[i] == 'R' for i in range(64)]
    union_gg = [warp_face_key[i] == 'G' or weft_face_key[i] == 'G' for i in range(64)]
    union_bb = [warp_face_key[i] == 'B' or weft_face_key[i] == 'B' for i in range(64)]

    for i in range(8):
        for j in range(8):
            num_rr[j+i*8] = sum(union_rr[j+i*8:])
            num_gg[j+i*8] = sum(union_gg[j+i*8:])
            num_bb[j+i*8] = sum(union_bb[j+i*8:])

    return {"R":num_rr, "G":num_gg, "B":num_bb}

ENDS_REMAIN_0 = create_color_count_cache(WARP_FACE_0, WEFT_FACE)
ENDS_REMAIN_1 = create_color_count_cache(WARP_FACE_0, WEFT_FACE)

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])

def convert_image_to_rgb(image_path, loom_width=3520):

    img = Image.open(image_path)
    nrow, ncol = img.size
    a_rat = loom_width / float(ncol)

    nrow = 8 * math.floor(int(a_rat * nrow) / 8)

    img_rs = img.resize((nrow, loom_width), Image.Resampling.LANCZOS)
    img_rgb = np.array(img_rs)

    nrow, ncol = img_rs.size
    bmp = np.zeros((nrow, ncol))
    first_end_group = True

    for i in range(0, nrow, 8):
        for j in range(0, ncol, 8):
            idx = j + i*ncol
            rgb8_desired = np.average(img_rgb[i:(i+1)*8,j:(j+1)*8], axis=(0,1))
            ws, rgb8_selected = convert_by_pixel_rgb8(rgb8_desired,
                                                      WARP_FACE_0 if first_end_group else WARP_FACE_1,
                                                      WEFT_FACE,
                                                      ENDS_REMAIN_0 if first_end_group else ENDS_REMAIN_1)
            bmp[i:(i+8), j:(j+8)] = ws.reshape(8,8)
 
            first_end_group = not first_end_group

    return bmp

def convert_to_indexed_shading(image_path, structures, num_ends_loom=3520, blur_dim=0, trim_max_perc=0.95):

    num_layers = len(structures)

    image_set = convert_image_to_set(image_path, num_layers, num_ends_loom)

    if blur_dim > 0:
        image_set = convert_array_to_set(blur(image_set, blur_dim), num_layers, transpose=False, trim_max_perc=trim_max_perc)

    num_picks, num_ends = image_set.shape

    structs = [resize_strct_to_layer(_, num_picks, num_ends) for _ in structures]

    return apply_layers(image_set, structs)

def convert_image_to_set(image_path, num_layers, num_ends=3520):

    tmp = np.asarray(Image.open(image_path))

    img = Image.open(image_path)
    img_gr = img.convert('L')
    img_gr = img_gr.transpose(Image.ROTATE_90)

    # jpg images are columns x rows
    nrow, ncol = img_gr.size
 
    a_rat = num_ends / float(ncol)
    img_rs = img_gr.resize((int(a_rat * nrow), num_ends), Image.NEAREST)

    return convert_array_to_set(img_rs, num_layers, transpose=True)

def convert_array_to_set(img_arr, num_layers, transpose=False, trim_max_perc=1.0):
    xx = np.array(img_arr)
    if transpose:
        xx = xx.T
    mx_v = (xx.max() + 1) * trim_max_perc
    return np.vectorize(lambda x:int(num_layers * x/mx_v))(xx)

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

def shaded_satin_square(num_ends=16, weft_face_cnt=4, weft_build=False):

    invert = False
    struct_sz = (num_ends,num_ends)
 
    if weft_face_cnt == 0:
        return np.zeros(struct_sz)
    if weft_face_cnt >= num_ends:
        return np.ones(struct_sz)
    if 2*weft_face_cnt > num_ends:
        invert = True
        weft_face_cnt = num_ends - weft_face_cnt 

    step_sz = num_ends // weft_face_cnt - 1

    x,y,z = step_sz, weft_face_cnt, num_ends

    ss = np.resize([int((i + x*j + y) % z < y) for i in range(z) for j in range(z)], struct_sz).astype(np.bool)
    if weft_build:
        ss = ss.transpose()
    return ~ss if invert else ss

def shaded_satin_square_float_dbl(num_ends=16, weft_face_cnt=4, weft_build=False):

    ss = shaded_satin_square(num_ends, weft_face_cnt, weft_build)
    ss_d = np.resize(np.zeros(4*(num_ends**2)),(2*num_ends,2*num_ends))

    for i in range(num_ends):
        for j in range(num_ends):
            ss_d[2*i,2*j] = ss[i,j]
            ss_d[2*i+1,2*j] = (j + (i % 2 == 0)) % 2 == 0
            ss_d[2*i,2*j+1] = 0
            ss_d[2*i+1,2*j+1] = 0

    return ss_d

def resize_strct_to_layer(wv_strct, num_picks, num_ends):

    nn = wv_strct.shape[1]

    mask = np.tile(wv_strct.transpose(), math.ceil(num_picks / nn)).transpose()
    mask = np.tile(mask, math.ceil(num_ends / nn))

    return mask[:num_picks,:num_ends]

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
    ratios = [1,16,16,16,16,16,16,16]
    return [(i, num_ends - i) for i in ratios]

def convert_by_pixel_rgb8(rgb8, warp_face_arr, weft_face_arr, ends_remain):

    rr,gg,bb = rgb8[:3] / 4

    rr = min(rr,TOTR)
    gg = min(gg,TOTG)
    bb = min(bb,TOTB)
    
    ends_select = {'R':0, 'G':0, 'B':0}
    ends_rgb = {'R':rr, 'G':gg, 'B':bb}

    ws = np.zeros(8*8, dtype=np.uint8)

    for i in range(8):
        for j in range(8):
            
            idx = j+i*8
            
            if warp_face_arr[idx] == weft_face_arr[idx]:
                ws[idx] = random.random() < .5
                color_select2 = weft_face_arr[idx] if ws[idx] else warp_face_arr[idx]
                #print(idx,0.5,0.5, color_select2, ends_select[color_select2], ends_rgb[color_select2], ends_remain[color_select2][idx])
            else:
                warp_color = warp_face_arr[idx]
                weft_color = weft_face_arr[idx]

                weft_pr = min(1, max(0,ends_rgb[weft_color] - ends_select[weft_color]))
                warp_pr = min(1, max(0,ends_rgb[warp_color] - ends_select[warp_color]))

                if warp_pr != weft_pr:
                    ws[idx] = random.random() * warp_pr * ends_remain[warp_color][idx] < weft_pr * ends_remain[weft_color][idx]
                else:
                    ws[idx] = random.random() < .5

                #color_select2 = weft_face_arr[idx] if ws[idx] else warp_face_arr[idx]
                #print(idx,weft_pr,warp_pr, color_select2, ends_select[color_select2], ends_rgb[color_select2], ends_remain[color_select2][idx])

            color_select = weft_face_arr[idx] if ws[idx] else warp_face_arr[idx]
            ends_select[color_select] += 1

    return ws, ends_select

def convert_to_shaded_satin(image_path, num_layers, num_ends_struct, loom_width=3520):

    ratios = generate_ratios(num_layers, num_ends_struct)

    image_set = convert_image_to_set(image_path, num_layers, loom_width)
    num_picks, num_warp_ends = image_set.shape

    satins = [shaded_satin_layer(num_picks, num_warp_ends, ratio) for ratio in ratios]

    return apply_layers(image_set, satins)

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

def write_to_file(weave_array, file_path, invert, transpose=False):

    im = Image.new('1', weave_array.shape)

    if invert:
        weave_array = (weave_array + 1) % 2

    rows,cols = im.size

    pixels = im.load()
    for i in range(rows):
        for j in range(cols):
            pixels[i,j] = int(weave_array[i,j]),

    if transpose:
        im = im.transpose(Image.ROTATE_270)
    im.save(file_path)

def main(args):

    parser = argparse.ArgumentParser()    
    
    parser.add_argument('-i','--input', help='input file path')
    parser.add_argument('--output-shaded-satin', default=None, help='outut file path')
    parser.add_argument('--output-double-warp', default=None, help='outut file path')
    parser.add_argument('--output-rgb-weaving', default=None, help='outut file path')
    parser.add_argument('--reverse', action="store_true", help='swap warp/weft facing')
    parser.add_argument('--warp-ends', default=3520, type=np.uint16, help='number of ends')
    parser.add_argument('--num-layers', default=4, type=np.uint8, help='number of layers')
    parser.add_argument('--num-ends-structure', default=16, type=np.uint8, help='number of ends for structures')
    parser.add_argument('-g','--game-of-life', help='game of life iteration cound [default=0]', type=np.uint8, default=0)

    args = parser.parse_args()

    ss_0 = list()
    ss_1 = list()
    for i in range(0,32):
        ss_0.append(shaded_satin_square_float_dbl(32,i,False).transpose())
        ss_1.append(shaded_satin_square_float_dbl(32,i,True))
        write_to_file(ss_1[-1], "/Users/muratahmed/Desktop/ss_16_{}_weft.bmp".format(i), False)

    if args.output_rgb_weaving:
        weaving_ss0 = convert_image_to_rgb(args.input, loom_width=1168)
        weaving_ss1 = convert_image_to_rgb(args.input, loom_width=1168)
        weaving_ss2 = convert_image_to_rgb(args.input, loom_width=1168)
        
        selvedge = np.resize([0, 1], (weaving_ss0.shape[0],8))
        weaving_ss = np.hstack((selvedge,weaving_ss0,weaving_ss1,weaving_ss2,selvedge))

    if args.output_shaded_satin:

        structures = [ss_1[idx] for idx in [0,4,29]]
        w_sc = convert_to_indexed_shading(args.input, structures, args.warp_ends, blur_dim=24, trim_max_perc=0.95)

        #_,pw_sc  = resize_structures_to_match(w_sc, plain_sc_extend(16))
        #w_dc     = double_cloth_interleave(w_sc, pw_sc)
        slvdg    = np.tile(plain_dc(4,4), ((w_sc.shape[0] // 4) + 1, 4))[:w_sc.shape[0],:]

        w_out = np.hstack((w_sc,slvdg))
        #w_out = np.hstack((slvdg,w_sc))
        w_out = w_out[:,:args.warp_ends]

        reverse_weaving = False
        if args.reverse:
            reverse_weaving = True

        write_to_file(w_out, args.output_shaded_satin, reverse_weaving, transpose=True)
 
if __name__ == '__main__':

    main(sys.argv[1:])
