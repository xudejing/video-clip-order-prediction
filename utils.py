"""Utils"""
import skvideo.io
import os
import glob


from tqdm import tqdm
import ffmpeg
import tensorflow as tf
from tensorboardX import SummaryWriter

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def sample_video_tuples(video_fname,  output_dir, clip_length=16, interval_length=8, tuple_length=3):
    videodata = skvideo.io.vread(video_fname)
    videoname = os.path.basename(video_fname)
    videoname = videoname[:videoname.rfind('.')]

    clip_start = 0
    for i in range(tuple_length):
        clipname = os.path.join(output_dir, '{}_{}.avi'.format(videoname, i+1))
        clipdata = videodata[clip_start:clip_start+clip_length]        
        clip_start = clip_start+clip_length+interval_length
        skvideo.io.vwrite(clipname, clipdata)

def skvideo_io_vread_compatiable(dir):
	filenames = glob.glob(os.path.join(dir, '**/*.avi'), recursive=True)
	for filename in tqdm(filenames):
		dirname, basename = os.path.split(filename)
		tmp_filename = os.path.join(dirname, 'tmp.avi')
		try:
			videodata = skvideo.io.vread(filename)
		except Exception:
			print('Convert ', filename)
			ffmpeg.input(filename).output(tmp_filename).run(quiet=True)
			os.remove(filename)
			os.rename(tmp_filename, filename)


if __name__ == "__main__":
	# sample_video_tuples('data/ucf101/video/BasketballDunk/v_BasketballDunk_g01_c01.avi', '.')
	skvideo_io_vread_compatiable('data/ucf101/video')