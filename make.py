import os
import sys
make = int(sys.argv[1])
cur_dir = os.getcwd().split('/')[-1]
if make == 0:
	os.system('mkdir -p ../checkpoints/%s'%(cur_dir))
	os.system('ln -s ../checkpoints/%s/ checkpoints'%(cur_dir))
	os.system('mkdir -p ../result/%s'%(cur_dir))
	os.system('ln -s ../result/%s/ result'%(cur_dir))
else:
	os.system('rm -rf ../checkpoints/%s'%(cur_dir))
	os.system('rm -rf checkpoints')
	os.system('rm -rf ../result/%s'%(cur_dir))
	os.system('rm -rf result')

