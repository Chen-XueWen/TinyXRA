#!/bin/bash
python main.py --risk_metric "std" --test_year 2024 --gpu cuda:0 --seed 62 &
python main.py --risk_metric "skew" --test_year 2024 --gpu cuda:1 --seed 62 &
python main.py --risk_metric "kurt" --test_year 2024 --gpu cuda:2 --seed 62 &
python main.py --risk_metric "sortino" --test_year 2024 --gpu cuda:3 --seed 62 &
wait
python main.py --risk_metric "std" --test_year 2023 --gpu cuda:0 --seed 62 &
python main.py --risk_metric "skew" --test_year 2023 --gpu cuda:1 --seed 62 &
python main.py --risk_metric "kurt" --test_year 2023 --gpu cuda:2 --seed 62 &
python main.py --risk_metric "sortino" --test_year 2023 --gpu cuda:3 --seed 62 &
wait
python main.py --risk_metric "std" --test_year 2022 --gpu cuda:0 --seed 62 &
python main.py --risk_metric "skew" --test_year 2022 --gpu cuda:1 --seed 62 &
python main.py --risk_metric "kurt" --test_year 2022 --gpu cuda:2 --seed 62 &
python main.py --risk_metric "sortino" --test_year 2022 --gpu cuda:3 --seed 62 &
wait
python main.py --risk_metric "std" --test_year 2021 --gpu cuda:0 --seed 62 &
python main.py --risk_metric "skew" --test_year 2021 --gpu cuda:1 --seed 62 &
python main.py --risk_metric "kurt" --test_year 2021 --gpu cuda:2 --seed 62 &
python main.py --risk_metric "sortino" --test_year 2021 --gpu cuda:3 --seed 62 &
wait
python main.py --risk_metric "std" --test_year 2020 --gpu cuda:0 --seed 62 &
python main.py --risk_metric "skew" --test_year 2020 --gpu cuda:1 --seed 62 &
python main.py --risk_metric "kurt" --test_year 2020 --gpu cuda:2 --seed 62 &
python main.py --risk_metric "sortino" --test_year 2020 --gpu cuda:3 --seed 62 &
wait
python main.py --risk_metric "std" --test_year 2019 --gpu cuda:0 --seed 62 &
python main.py --risk_metric "skew" --test_year 2019 --gpu cuda:1 --seed 62 &
python main.py --risk_metric "kurt" --test_year 2019 --gpu cuda:2 --seed 62 &
python main.py --risk_metric "sortino" --test_year 2019 --gpu cuda:3 --seed 62 &
wait
python main.py --risk_metric "std" --test_year 2018 --gpu cuda:0 --seed 62 &
python main.py --risk_metric "skew" --test_year 2018 --gpu cuda:1 --seed 62 &
python main.py --risk_metric "kurt" --test_year 2018 --gpu cuda:2 --seed 62 &
python main.py --risk_metric "sortino" --test_year 2018 --gpu cuda:3 --seed 62 &
wait