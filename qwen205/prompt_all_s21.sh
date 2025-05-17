#!/bin/bash
python main.py --risk_metric std --test_year 2024 --seed 21 --gpu cuda:0 > results/std/2024s21.txt &
python main.py --risk_metric skew --test_year 2024 --seed 21 --gpu cuda:1 > results/skew/2024s21.txt &
python main.py --risk_metric kurt --test_year 2024 --seed 21 --gpu cuda:2 > results/kurt/2024s21.txt &
python main.py --risk_metric sortino --test_year 2024 --seed 21 --gpu cuda:3 > results/sort/2024s21.txt &
wait

python main.py --risk_metric std --test_year 2023 --seed 21 --gpu cuda:0 > results/std/2023s21.txt &
python main.py --risk_metric skew --test_year 2023 --seed 21 --gpu cuda:1 > results/skew/2023s21.txt &
python main.py --risk_metric kurt --test_year 2023 --seed 21 --gpu cuda:2 > results/kurt/2023s21.txt &
python main.py --risk_metric sortino --test_year 2023 --seed 21 --gpu cuda:3 > results/sort/2023s21.txt &
wait

python main.py --risk_metric std --test_year 2022 --seed 21 --gpu cuda:0 > results/std/2022s21.txt &
python main.py --risk_metric skew --test_year 2022 --seed 21 --gpu cuda:1 > results/skew/2022s21.txt &
python main.py --risk_metric kurt --test_year 2022 --seed 21 --gpu cuda:2 > results/kurt/2022s21.txt &
python main.py --risk_metric sortino --test_year 2022 --seed 21 --gpu cuda:3 > results/sort/2022s21.txt &
wait

python main.py --risk_metric std --test_year 2021 --seed 21 --gpu cuda:0 > results/std/2021s21.txt &
python main.py --risk_metric skew --test_year 2021 --seed 21 --gpu cuda:1 > results/skew/2021s21.txt &
python main.py --risk_metric kurt --test_year 2021 --seed 21 --gpu cuda:2 > results/kurt/2021s21.txt &
python main.py --risk_metric sortino --test_year 2021 --seed 21 --gpu cuda:3 > results/sort/2021s21.txt &
wait

python main.py --risk_metric std --test_year 2020 --seed 21 --gpu cuda:0 > results/std/2020s21.txt &
python main.py --risk_metric skew --test_year 2020 --seed 21 --gpu cuda:1 > results/skew/2020s21.txt &
python main.py --risk_metric kurt --test_year 2020 --seed 21 --gpu cuda:2 > results/kurt/2020s21.txt &
python main.py --risk_metric sortino --test_year 2020 --seed 21 --gpu cuda:3 > results/sort/2020s21.txt &
wait

python main.py --risk_metric std --test_year 2019 --seed 21 --gpu cuda:0 > results/std/2019s21.txt &
python main.py --risk_metric skew --test_year 2019 --seed 21 --gpu cuda:1 > results/skew/2019s21.txt &
python main.py --risk_metric kurt --test_year 2019 --seed 21 --gpu cuda:2 > results/kurt/2019s21.txt &
python main.py --risk_metric sortino --test_year 2019 --seed 21 --gpu cuda:3 > results/sort/2019s21.txt &
wait

python main.py --risk_metric std --test_year 2018 --seed 21 --gpu cuda:0 > results/std/2018s21.txt &
python main.py --risk_metric skew --test_year 2018 --seed 21 --gpu cuda:1 > results/skew/2018s21.txt &
python main.py --risk_metric kurt --test_year 2018 --seed 21 --gpu cuda:2 > results/kurt/2018s21.txt &
python main.py --risk_metric sortino --test_year 2018 --seed 21 --gpu cuda:3 > results/sort/2018s21.txt &
wait
