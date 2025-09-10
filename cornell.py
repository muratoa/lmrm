import argparse
import matplotlib.image as img
import numpy as np
import math
import random
import sys
from PIL import Image

def plain_dc(nrow,ncol):
    return np.tile(np.array([[1,0,0,0],[1,1,1,0],[0,0,1,0],[1,0,1,1]]), (int(nrow/4),int(ncol/4)))

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

def write_to_file(weave_array, file_path, invert):

    im = Image.new('1', weave_array.shape)

    if invert:
        weave_array = (weave_array + 1) % 2

    rows,cols = im.size

    pixels = im.load()
    for i in range(rows):
        for j in range(cols):
            pixels[i,j] = int(weave_array[i,j]),

    im.save(file_path)

def main(args):

    parser = argparse.ArgumentParser()    
    
    parser.add_argument('-o','--output', help='output file path')
    parser.add_argument('--reverse', action="store_true", help='swap warp/weft facing')

    args = parser.parse_args()

    reverse_weaving = False
    if args.reverse:
        reverse_weaving = True

    weav = cornell_sample()
    write_to_file(np.transpose(np.hstack([weav,weav])), args.output, reverse_weaving)

if __name__ == '__main__':

    main(sys.argv[1:])
