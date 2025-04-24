#!/bin/bash
python plot_explain_words.py --risk_metric std
python plot_explain_words.py --risk_metric skew
python plot_explain_words.py --risk_metric kurt
python plot_explain_words.py --risk_metric sortino 

python plot_explain_sents.py --risk_metric std
python plot_explain_sents.py --risk_metric skew
python plot_explain_sents.py --risk_metric kurt
python plot_explain_sents.py --risk_metric sortino 
