#!/bin/bash
python main.py --risk_metric "std" --test_year 2024 --gpu cuda:7 --seed 21
python main.py --risk_metric "std" --test_year 2023 --gpu cuda:7 --seed 21
python main.py --risk_metric "std" --test_year 2022 --gpu cuda:7 --seed 21
python main.py --risk_metric "std" --test_year 2021 --gpu cuda:7 --seed 21
python main.py --risk_metric "std" --test_year 2020 --gpu cuda:7 --seed 21
python main.py --risk_metric "std" --test_year 2019 --gpu cuda:7 --seed 21
python main.py --risk_metric "std" --test_year 2018 --gpu cuda:7 --seed 21

python main.py --risk_metric "skew" --test_year 2024 --gpu cuda:7 --seed 21
python main.py --risk_metric "skew" --test_year 2023 --gpu cuda:7 --seed 21
python main.py --risk_metric "skew" --test_year 2022 --gpu cuda:7 --seed 21
python main.py --risk_metric "skew" --test_year 2021 --gpu cuda:7 --seed 21
python main.py --risk_metric "skew" --test_year 2020 --gpu cuda:7 --seed 21
python main.py --risk_metric "skew" --test_year 2019 --gpu cuda:7 --seed 21
python main.py --risk_metric "skew" --test_year 2018 --gpu cuda:7 --seed 21

python main.py --risk_metric "kurt" --test_year 2024 --gpu cuda:7 --seed 21
python main.py --risk_metric "kurt" --test_year 2023 --gpu cuda:7 --seed 21
python main.py --risk_metric "kurt" --test_year 2022 --gpu cuda:7 --seed 21
python main.py --risk_metric "kurt" --test_year 2021 --gpu cuda:7 --seed 21
python main.py --risk_metric "kurt" --test_year 2020 --gpu cuda:7 --seed 21
python main.py --risk_metric "kurt" --test_year 2019 --gpu cuda:7 --seed 21
python main.py --risk_metric "kurt" --test_year 2018 --gpu cuda:7 --seed 21

python main.py --risk_metric "sortino" --test_year 2024 --gpu cuda:7 --seed 21
python main.py --risk_metric "sortino" --test_year 2023 --gpu cuda:7 --seed 21
python main.py --risk_metric "sortino" --test_year 2022 --gpu cuda:7 --seed 21
python main.py --risk_metric "sortino" --test_year 2021 --gpu cuda:7 --seed 21
python main.py --risk_metric "sortino" --test_year 2020 --gpu cuda:7 --seed 21
python main.py --risk_metric "sortino" --test_year 2019 --gpu cuda:7 --seed 21
python main.py --risk_metric "sortino" --test_year 2018 --gpu cuda:7 --seed 21
