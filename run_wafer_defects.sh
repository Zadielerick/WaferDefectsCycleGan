#!/bin/bash

TRAINY_DIR=datasets/wafer/trainB
TESTY_DIR=datasets/wafer/testB
DEFECT_DIRS=datasets/wafer_backup/trainY/*

DEFECTS=("Center  Donut  Edge-Loc  Edge-Ring  Loc  Near-full  Random  Scratch")

for def in $DEFECTS
do
	echo $def
	rm -rf $TRAINY_DIR $TESTY_DIR
	cp -r datasets/wafer_backup/trainY/$def $TRAINY_DIR
	cp -r datasets/wafer_backup/testY/$def $TESTY_DIR

	python3 train.py --dataroot ./datasets/wafer --name wafer_lr01_$def --model cycle_gan --ngf 32 --batch_size 5 --save_epoch_freq 25 --lr 0.01
	python3 test.py --dataroot ./datasets/wafer --name wafer_lr01_$def --model cycle_gan --ngf 32 --batch_size 5 --epoch 25
	python3 test.py --dataroot ./datasets/wafer --name wafer_lr01_$def --model cycle_gan --ngf 32 --batch_size 5 --epoch 50
	python3 test.py --dataroot ./datasets/wafer --name wafer_lr01_$def --model cycle_gan --ngf 32 --batch_size 5 --epoch 75
	python3 test.py --dataroot ./datasets/wafer --name wafer_lr01_$def --model cycle_gan --ngf 32 --batch_size 5 --epoch 100
	python3 test.py --dataroot ./datasets/wafer --name wafer_lr01_$def --model cycle_gan --ngf 32 --batch_size 5 --epoch 125
	python3 test.py --dataroot ./datasets/wafer --name wafer_lr01_$def --model cycle_gan --ngf 32 --batch_size 5 --epoch 150
	python3 test.py --dataroot ./datasets/wafer --name wafer_lr01_$def --model cycle_gan --ngf 32 --batch_size 5 --epoch 175
	python3 test.py --dataroot ./datasets/wafer --name wafer_lr01_$def --model cycle_gan --ngf 32 --batch_size 5 --epoch 200

	python3 fid_kid_calc/fid-kid-calc.py -t ./results/wafer_lr01_$def
done
