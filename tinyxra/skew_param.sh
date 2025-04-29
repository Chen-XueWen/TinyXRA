#!/bin/bash
python main.py --risk_metric "skew" --test_year 2024 --encoder_lr 1e-6 --gpu cuda:1 --seed 21 &
python main.py --risk_metric "skew" --test_year 2024 --encoder_lr 5e-7 --gpu cuda:2 --seed 21 &
python main.py --risk_metric "skew" --test_year 2024 --encoder_lr 1e-7 --gpu cuda:3 --seed 21 &
wait
