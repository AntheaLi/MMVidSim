import os 
import sys
from glob import glob
from subprocess import call


exp_names = os.listdir ('trained_ckpt')
for exp_name in exp_names:
    print(exp_name)
    exp_dir = os.path.join('trained_ckpt', exp_name)
    results_dir = glob(exp_dir + f'/visu_*')

    for rd in results_dir:
        rdf = rd.split('/')[-1]
        print(rdf)
        gif_dirs = glob(rd+f'/visu*')
        if len(gif_dirs) > 0:
            for gd in gif_dirs:
                cmd = f'cp -r {gd} ./gifs/{exp_name}_{rdf}/'
                call(cmd, shell=True)
