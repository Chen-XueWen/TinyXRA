#!/bin/bash
python main-eval.py --risk_metric "kurt" --test_year 2024 --gpu cuda:7 --seed 42
python main-eval.py --risk_metric "kurt" --test_year 2023 --gpu cuda:7 --seed 42
python main-eval.py --risk_metric "kurt" --test_year 2022 --gpu cuda:7 --seed 42
python main-eval.py --risk_metric "kurt" --test_year 2021 --gpu cuda:7 --seed 42
python main-eval.py --risk_metric "kurt" --test_year 2020 --gpu cuda:7 --seed 42
python main-eval.py --risk_metric "kurt" --test_year 2019 --gpu cuda:7 --seed 42
python main-eval.py --risk_metric "kurt" --test_year 2018 --gpu cuda:7 --seed 42
