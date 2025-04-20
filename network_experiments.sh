#!/bin/bash

sleep 1 && CUDA_VISIBLE_DEVICES="1" python ./train_CNN.py \
 --n_unroll 1 --R 6 --dataset kirby21 --early_stop_patience 40 --ReduceLROnPlateau_patience 20&
sleep 150 && CUDA_VISIBLE_DEVICES="2" python ./train_CNN.py \
--n_unroll 2 --R 6 --dataset kirby21 --early_stop_patience 40 --ReduceLROnPlateau_patience 20&
wait
sleep 1 && CUDA_VISIBLE_DEVICES="1" python ./train_CNN.py \
--n_unroll 3 --R 6 --dataset kirby21 --early_stop_patience 40 --ReduceLROnPlateau_patience 20&
sleep 150 && CUDA_VISIBLE_DEVICES="2" python ./train_CNN.py \
--n_unroll 1 --dc_step_size 1 --numpoints 3 --R 6 --dataset kirby21  --early_stop_patience 40 --ReduceLROnPlateau_patience 20&
wait
sleep 1 && CUDA_VISIBLE_DEVICES="1" python ./train_CNN.py \
--n_unroll 1 --dc_step_size 1 --numpoints 4 --R 6 --dataset kirby21 --early_stop_patience 40 --ReduceLROnPlateau_patience 20 &
sleep 60 && CUDA_VISIBLE_DEVICES="2" python ./train_CNN.py \
--n_unroll 1 --dc_step_size 1 --R 6 --grid_size 1.5 --dataset kirby21 --early_stop_patience 40 --ReduceLROnPlateau_patience 20&
wait
sleep 1 && CUDA_VISIBLE_DEVICES="1" python ./train_CNN.py \
--n_unroll 1 --dc_step_size 1 --R 6 --grid_size 1.125 --dataset kirby21 --early_stop_patience 40 --ReduceLROnPlateau_patience 20&
sleep 60 && CUDA_VISIBLE_DEVICES="2" python ./train_CNN.py \
--n_unroll 2 --dc_step_size 1 --R 6 --dataset kirby21 --early_stop_patience 40 --ReduceLROnPlateau_patience 20&
wait
sleep 1 && CUDA_VISIBLE_DEVICES="1" python ./train_CNN.py \
--n_unroll 3 --dc_step_size 1 --R 6 --dataset kirby21 --early_stop_patience 40 --ReduceLROnPlateau_patience 20&
sleep 60 && CUDA_VISIBLE_DEVICES="2" python ./train_CNN.py \
 --n_unroll 4 --dc_step_size 1 --R 6 --dataset kirby21 --early_stop_patience 40 --ReduceLROnPlateau_patience 20&
wait
sleep 1 && CUDA_VISIBLE_DEVICES="1" python ./train_CNN.py \
--n_unroll 1 --dc_step_size 1 --R 6 --dataset kirby21 --early_stop_patience 40 --ReduceLROnPlateau_patience 20&
wait
sleep 1 && CUDA_VISIBLE_DEVICES="0" python ./train_CNN.py \
 --n_unroll 1 --R 3 --dataset kirby21 --early_stop_patience 40 --ReduceLROnPlateau_patience 20&
sleep 150 && CUDA_VISIBLE_DEVICES="1" python ./train_CNN.py \
 --n_unroll 2 --R 3 --dataset kirby21 --early_stop_patience 40 --ReduceLROnPlateau_patience 20&
wait
sleep 1 && CUDA_VISIBLE_DEVICES="0" python ./train_CNN.py \
 --n_unroll 3 --R 3 --dataset kirby21 --early_stop_patience 40 --ReduceLROnPlateau_patience 20&
sleep 60 && CUDA_VISIBLE_DEVICES="2" python ./train_CNN.py \
 --n_unroll 2 --dc_step_size 1 --R 3 --dataset kirby21 --early_stop_patience 40 --ReduceLROnPlateau_patience 20&
wait
sleep 1 && CUDA_VISIBLE_DEVICES="2" python ./train_CNN.py \
 --n_unroll 3 --dc_step_size 1 --R 3 --dataset kirby21 --early_stop_patience 40 --ReduceLROnPlateau_patience 20&
sleep 150 && CUDA_VISIBLE_DEVICES="1" python ./train_CNN.py \
 --n_unroll 4 --dc_step_size 1 --R 3 --dataset kirby21 --early_stop_patience 40 --ReduceLROnPlateau_patience 20&
wait
sleep 1 && CUDA_VISIBLE_DEVICES="2" python ./train_CNN.py \
--n_unroll 1 --dc_step_size 1 --R 3 --dataset kirby21 --early_stop_patience 40 --ReduceLROnPlateau_patience 20&
wait
## p_loss
sleep 1 && CUDA_VISIBLE_DEVICES="1" python ./train_CNN.py \
 --n_unroll 3 --dc_step_size 1 --R 3 --dataset kirby21 --early_stop_patience 40 --ReduceLROnPlateau_patience 20 --p_loss --beta 1&
sleep 150 && CUDA_VISIBLE_DEVICES="2" python ./train_CNN.py \
 --n_unroll 3 --dc_step_size 1 --R 6 --dataset kirby21 --early_stop_patience 40 --ReduceLROnPlateau_patience 20 --p_loss --beta 1&
wait
### rot angle
sleep 1 && CUDA_VISIBLE_DEVICES="2" python ./train_CNN.py \
 --n_unroll 3 --dc_step_size 1 --R 3 --dataset kirby21 --early_stop_patience 40 \
 --ReduceLROnPlateau_patience 20 --rotation_angle&
sleep 150 && CUDA_VISIBLE_DEVICES="0" python ./train_CNN.py \
 --n_unroll 3 --dc_step_size 1 --R 6 --dataset kirby21 --early_stop_patience 40 \
 --ReduceLROnPlateau_patience 20 --rotation_angle&
wait
### deli_cs
sleep 1 && CUDA_VISIBLE_DEVICES="0" python ./train_CNN.py --n_unroll 3 \
 --dc_step_size 1 --R 3 --dataset deli_cs --early_stop_patience 40 --ReduceLROnPlateau_patience 20 --nufft_split_channel&
sleep 150 && CUDA_VISIBLE_DEVICES="2" python ./train_CNN.py --n_unroll 3 \
 --dc_step_size 1 --R 6 --dataset deli_cs --early_stop_patience 40 --ReduceLROnPlateau_patience 20 --nufft_split_batch&

