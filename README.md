# kaggle-fruit360-classification

##네트워크 구성
 - MyModel.py 폴더 안에Timm Labrirary를 이용하여 네트워크 구성
 - timm.list_models(pretrainined==True)를 이용하여 model list 확인 가능
 - timm.create_model('model_file_name', pretrained=True)
  
##학습 방법
 - pythom main.py 
 - Config file을 이용하여 parameter 값 변경 가능
 - Optimizer, loss function, scheduler 등 getattr 함수 사용 (Config를 이용하여 변강 가능)

##전처리방법
 - albumentations 함수 용
 - Normalization, to Tensor 필수 조건
 - GaussianNoise, RandomCrop, RandomHorizontalFlip 등 다양한 argumentations 값 사용 가능

##다양한 학습 방법
 - K-Fold 방식 사용 
 - n-splits 변수 변경가능 (Config)
 - use_kfold 변수 변경가능 (Config)
 
##학습 방법
 - make_dataset.py 를 이용하여 cvs 파일 생성
 - model_architecture, data_path 및 model_output_dir 변경(config.yml)
 - train epoch, batch_size, n_splits 등 원하는 값에 따라 변경(config.yml)
 - main.py  실행


