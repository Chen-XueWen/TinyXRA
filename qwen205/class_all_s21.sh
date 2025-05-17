#!/bin/bash
python main-class.py --risk_metric std --test_year 2024 --seed 21 --gpu cuda:0 &
python main-class.py --risk_metric skew --test_year 2024 --seed 21 --gpu cuda:1 &
python main-class.py --risk_metric kurt --test_year 2024 --seed 21 --gpu cuda:2 &
python main-class.py --risk_metric sortino --test_year 2024 --seed 21 --gpu cuda:3 &
wait

python main-class.py --risk_metric std --test_year 2023 --seed 21 --gpu cuda:0 &
python main-class.py --risk_metric skew --test_year 2023 --seed 21 --gpu cuda:1 &
python main-class.py --risk_metric kurt --test_year 2023 --seed 21 --gpu cuda:2 &
python main-class.py --risk_metric sortino --test_year 2023 --seed 21 --gpu cuda:3 &
wait

python main-class.py --risk_metric std --test_year 2022 --seed 21 --gpu cuda:0 &
python main-class.py --risk_metric skew --test_year 2022 --seed 21 --gpu cuda:1 &
python main-class.py --risk_metric kurt --test_year 2022 --seed 21 --gpu cuda:2 &
python main-class.py --risk_metric sortino --test_year 2022 --seed 21 --gpu cuda:3 &
wait

python main-class.py --risk_metric std --test_year 2021 --seed 21 --gpu cuda:0 &
python main-class.py --risk_metric skew --test_year 2021 --seed 21 --gpu cuda:1 &
python main-class.py --risk_metric kurt --test_year 2021 --seed 21 --gpu cuda:2 &
python main-class.py --risk_metric sortino --test_year 2021 --seed 21 --gpu cuda:3 &
wait

python main-class.py --risk_metric std --test_year 2020 --seed 21 --gpu cuda:0 &
python main-class.py --risk_metric skew --test_year 2020 --seed 21 --gpu cuda:1 &
python main-class.py --risk_metric kurt --test_year 2020 --seed 21 --gpu cuda:2 &
python main-class.py --risk_metric sortino --test_year 2020 --seed 21 --gpu cuda:3 &
wait

python main-class.py --risk_metric std --test_year 2019 --seed 21 --gpu cuda:0 &
python main-class.py --risk_metric skew --test_year 2019 --seed 21 --gpu cuda:1 &
python main-class.py --risk_metric kurt --test_year 2019 --seed 21 --gpu cuda:2 &
python main-class.py --risk_metric sortino --test_year 2019 --seed 21 --gpu cuda:3 &
wait

python main-class.py --risk_metric std --test_year 2018 --seed 21 --gpu cuda:0 &
python main-class.py --risk_metric skew --test_year 2018 --seed 21 --gpu cuda:1 &
python main-class.py --risk_metric kurt --test_year 2018 --seed 21 --gpu cuda:2 &
python main-class.py --risk_metric sortino --test_year 2018 --seed 21 --gpu cuda:3 &
wait
