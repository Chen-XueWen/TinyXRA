#!/bin/bash
python main.py --risk_metric "std" --test_year 2024 --gpu cuda:4 &
python main.py --risk_metric "skew" --test_year 2024 --gpu cuda:5 &
python main.py --risk_metric "kurt" --test_year 2024 --gpu cuda:6 &
python main.py --risk_metric "sortino" --test_year 2024 --gpu cuda:7 &
wait