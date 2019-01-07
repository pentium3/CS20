set MODEL_NAME=censuseg
set PACKAGE_PATH=C:\\Users\\hp\\Downloads\\cloudml-samples-master\\cloudml-template\\examples\\census-classification\\trainer
set TRAIN_FILES=gs://pentiumlab/cloudml-template/examples/census-classification/data/train-data-*.csv
set EVAL_FILES=gs://pentiumlab/cloudml-template/examples/census-classification/data/eval-data-*.csv
set MODEL_DIR=gs://pentiumlab/cloudml-template/examples/census-classification/
set JOB_NAME=train_%MODEL_NAME%_gpu
gcloud ml-engine jobs submit training %JOB_NAME% --job-dir=%MODEL_DIR% --runtime-version=1.4 --module-name=trainer.task --package-path=%PACKAGE_PATH% --config=config.yaml -- --train-files=%TRAIN_FILES% --eval-files=%EVAL_FILES% --train-steps=10000
