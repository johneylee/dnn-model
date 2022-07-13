#!/bin/sh

#echo "argc=$#"
#echo "all=$@"

argc=$#

echo $argc
if [ $argc -eq 6 ]; then
  gpu_id=$1
  name=$2
  struct=$3
  mode=$4
  opt=$5
  rnn_type=$6
  weights=[1]
elif [ $argc -eq 7 ]; then
  gpu_id=$1
  name=$2
  struct=$3
  mode=$4
  opt=$5
  rnn_type=$6
  weights=$7
else
  echo "[Error] argument"
  exit
fi

dir_fea="features_"$struct"_"$mode
rnn_type="$rnn_type"blockfused


if [ $gpu_id -eq 0 ]
then
  echo "1st GPU"
elif [ $gpu_id -eq 1 ]
then
  echo "2nd GPU"
else
  echo "[Error] gpu_id"
  exit
fi

if [ $rnn_type = "lstmblockfused" ]
then
  echo
elif [ $rnn_type = "blstmblockfused" ]
then
  echo
else
  echo "[Error] rnn_type"
  exit
fi

if [ $struct = "pcm_pha" ]; then
  featype_inp="logmag_noisy+pha_noisy+bpd_noisy"
else
  featype_inp="logmag_noisy+pha_noisy"
fi

if [ $mode = "ma" ]; then
  featype_ref="irm"

elif [ $mode = "cma" ]; then
  featype_ref="cirm_real+cirm_imag"

elif [ $mode = "msa" ]; then
  featype_ref="mag_clean"

elif [ $mode = "tdr_old" ]; then
  featype_ref="frm_hann_clean"

elif [ $mode = "tdr" ]; then
  featype_ref="sig_max+frm_hann_norm_clean"

elif [ $mode = "etdr_old" ]; then
  featype_ref="frm_rect_clean"

elif [ $mode = "etdr" ]; then
  featype_ref="sig_max+frm_rect_norm_clean"

elif [ $mode = "params" ]; then
  featype_ref="sig_max+mag_norm_warp_clean+mag_norm_warp_noise+cos_xn+cos_xy+sin_xy"

elif [ $mode = "params+etdr" ]; then
  featype_ref="sig_max+frm_rect_norm_clean+mag_norm_warp_clean+mag_norm_warp_noise+cos_xn+cos_xy+sin_xy"

elif [ $mode = "params+etdr+sc" ]; then
  featype_ref="sig_max+frm_rect_norm_clean+mag_norm_warp_clean+mag_norm_warp_noise+cos_xn+cos_xy+sin_xy"

else
  ehco "[Error] mode"
fi

echo "------------------------------------------------"
echo "name:      "$name
echo "mode:      "$mode
echo "dir_fea:   "$dir_fea
echo "inp:       "$featype_inp
echo "ref:       "$featype_ref
echo "gpu_id:    "$gpu_id
echo "rnn_type:  "$rnn_type
echo "weights:   "$weights
echo "------------------------------------------------"

# Extract features.
if [ -d "$dir_fea" ]; then
  echo "Skip feature extraction process."
else
  python extract_enh_features.py $featype_inp $featype_ref data_wav/train $dir_fea train
  python extract_enh_features.py $featype_inp $featype_ref data_wav/devel $dir_fea devel
fi

# Network Train
python train_network.py $name $struct $mode $opt $dir_fea $featype_inp $featype_ref --gpu_id=$gpu_id --rnn_type=$rnn_type --weights=$weights --fc_type=tanh --learning_rate=0.0005 --thr_clip=0.5
