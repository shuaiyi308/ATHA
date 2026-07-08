root = 'output/checkpoints'
import os
import subprocess as cmd

for d in os.listdir(root):
    for f in os.listdir(os.path.join(root, d)):
        if '.tar' in f:
            if 'ave' not in f or 'zero-shot' not in f:
                f = os.path.join(root, d, f)
                cmd.getoutput('rm %s'%f)

