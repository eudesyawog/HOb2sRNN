#!/bin/sh

path_ts_radar="./reunion/radarTimeSeries"
path_ts_opt="./reunion/optTimeSeries"
path_gt="./reunion/hierarchical_gt"

for i in 0
do
    python HOb2sRNN.py $path_ts_radar/train_x$i\_50.npy $path_ts_opt/train_x$i\_50.npy $path_gt/train_y$i\_50.npy \
    $path_ts_radar/valid_x$i\_50.npy $path_ts_opt/valid_x$i\_50.npy $path_gt/valid_y$i\_50.npy \
    $path_ts_radar/test_x$i\_50.npy $path_ts_opt/test_x$i\_50.npy $path_gt/test_y$i\_50.npy \
    $i model 26 21 1
done