#!/bin/bash
python main.py --risk_metric "std" --test_year 2024 --gpu cuda:7 --seed 98 &
python main.py --risk_metric "std" --test_year 2023 --gpu cuda:7 --seed 98 &
wait

python main.py --risk_metric "std" --test_year 2022 --gpu cuda:7 --seed 98 &
python main.py --risk_metric "std" --test_year 2021 --gpu cuda:7 --seed 98 &
wait

python main.py --risk_metric "std" --test_year 2020 --gpu cuda:7 --seed 98 &
python main.py --risk_metric "std" --test_year 2019 --gpu cuda:7 --seed 98 &
wait

python main.py --risk_metric "std" --test_year 2018 --gpu cuda:7 --seed 98 &
python main.py --risk_metric "std" --test_year 2024 --gpu cuda:7 --seed 62 &
wait

python main.py --risk_metric "std" --test_year 2023 --gpu cuda:7 --seed 62 &
python main.py --risk_metric "std" --test_year 2022 --gpu cuda:7 --seed 62 &
wait

python main.py --risk_metric "std" --test_year 2021 --gpu cuda:7 --seed 62 &
python main.py --risk_metric "std" --test_year 2020 --gpu cuda:7 --seed 62 &
wait

python main.py --risk_metric "std" --test_year 2019 --gpu cuda:7 --seed 62 &
python main.py --risk_metric "std" --test_year 2018 --gpu cuda:7 --seed 62 &
wait

python main.py --risk_metric "std" --test_year 2024 --gpu cuda:7 --seed 83 &
python main.py --risk_metric "std" --test_year 2023 --gpu cuda:7 --seed 83 &
wait

python main.py --risk_metric "std" --test_year 2022 --gpu cuda:7 --seed 83 &
python main.py --risk_metric "std" --test_year 2021 --gpu cuda:7 --seed 83 &
wait

python main.py --risk_metric "std" --test_year 2020 --gpu cuda:7 --seed 83 &
python main.py --risk_metric "std" --test_year 2019 --gpu cuda:7 --seed 83 &
wait

python main.py --risk_metric "std" --test_year 2018 --gpu cuda:7 --seed 83 &
python main.py --risk_metric "std" --test_year 2024 --gpu cuda:7 --seed 21 &
wait

python main.py --risk_metric "std" --test_year 2023 --gpu cuda:7 --seed 21 &
python main.py --risk_metric "std" --test_year 2022 --gpu cuda:7 --seed 21 &
wait

python main.py --risk_metric "std" --test_year 2021 --gpu cuda:7 --seed 21 &
python main.py --risk_metric "std" --test_year 2020 --gpu cuda:7 --seed 21 &
wait

python main.py --risk_metric "std" --test_year 2019 --gpu cuda:7 --seed 21 &
python main.py --risk_metric "std" --test_year 2018 --gpu cuda:7 --seed 21 &
wait
