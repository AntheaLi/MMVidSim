import sys
from glob import glob

activity = sys.argv[1] 
samples = glob(f"holo/video/*-{activity}")
print(len(samples))
file_names = [f.split('/')[-1] for f in samples]


with open(f"holo_gen/gen_{activity}.txt", 'w') as f:
    for file in file_names:
        f.write(f'python holo_action_project.py --save_video --video_name {file} \n')
