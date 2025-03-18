#!/bin/bash
python main.py --project XRC_SkewTuning --risk_metric "skew" --encoder_lr 1e-5 --classifier_lr 4e-5 --gpu cuda:7
python main.py --project XRC_SkewTuning --risk_metric "skew" --encoder_lr 1e-5 --classifier_lr 2e-5 --gpu cuda:7
python main.py --project XRC_SkewTuning --risk_metric "skew" --encoder_lr 1e-6 --classifier_lr 6e-5 --gpu cuda:7
python main.py --project XRC_SkewTuning --risk_metric "skew" --encoder_lr 1e-6 --classifier_lr 4e-5 --gpu cuda:7
python main.py --project XRC_SkewTuning --risk_metric "skew" --encoder_lr 1e-6 --classifier_lr 2e-5 --gpu cuda:7
python main.py --project XRC_KurtTuning --risk_metric "kurt" --encoder_lr 1e-5 --classifier_lr 4e-5 --gpu cuda:7
python main.py --project XRC_KurtTuning --risk_metric "kurt" --encoder_lr 1e-5 --classifier_lr 2e-5 --gpu cuda:7
python main.py --project XRC_KurtTuning --risk_metric "kurt" --encoder_lr 1e-6 --classifier_lr 6e-5 --gpu cuda:7
python main.py --project XRC_KurtTuning --risk_metric "kurt" --encoder_lr 1e-6 --classifier_lr 4e-5 --gpu cuda:7
python main.py --project XRC_KurtTuning --risk_metric "kurt" --encoder_lr 1e-6 --classifier_lr 2e-5 --gpu cuda:7
