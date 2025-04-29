#!/bin/bash
python main.py --risk_metric "skew" --encoder_lr 1e-6 --test_year 2024 --gpu cuda:0 --seed 21 &
python main.py --risk_metric "skew" --encoder_lr 1e-6 --test_year 2023 --gpu cuda:1 --seed 21 &
python main.py --risk_metric "skew" --encoder_lr 1e-6 --test_year 2022 --gpu cuda:2 --seed 21 &
wait
