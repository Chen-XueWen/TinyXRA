#!/bin/bash
python main.py --risk_metric "std" --test_year 2024 --gpu cuda:0 &
python main.py --risk_metric "skew" --test_year 2024 --gpu cuda:1 &
python main.py --risk_metric "kurt" --test_year 2024 --gpu cuda:2 &
python main.py --risk_metric "sortino" --test_year 2024 --gpu cuda:3 &
wait