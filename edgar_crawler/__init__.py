import os
DATASET_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../datasets')
LOGGING_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../edgar_logs')

if not os.path.exists(DATASET_DIR):
    os.mkdir(DATASET_DIR)
    os.mkdir(DATASET_DIR + "/EXTRACTED_FILINGS")
    os.mkdir(DATASET_DIR + "/FILINGS_METADATA")
    os.mkdir(DATASET_DIR + "/INDICES")
    os.mkdir(DATASET_DIR + "/RAW_FILINGS")
 

if not os.path.exists(LOGGING_DIR):
	os.mkdir(LOGGING_DIR)