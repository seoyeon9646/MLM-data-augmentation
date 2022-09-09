# Masked Language Modeling

비정제 한국어 문장(ex. Korean Hate Speech Dataset, Naver Sentiment Movie Corpus)에 데이터 증강을 적용하기 위해서 [KcELECTRA-base](https://github.com/Beomi/KcELECTRA)를 Masked Language Modeling 방식으로 학습하는 코드입니다. 모델을 트레이닝한 후에 새로운 문장의 일부에 마스킹을 적용하고 인퍼런스를 적용하면 아래와 같이 빈칸에 알맞은 새로운 토큰을 후보로 생성해냅니다. 대용량의 데이터를 빠르게 학습하기 위해서 deepspeed ZeRO2를 활용합니다.
프로젝트에 대해서 좀 더 자세히 알고 싶으시다면 제 [velog](https://velog.io/@seoyeon96/PLM%EC%9D%84-%EC%9D%B4%EC%9A%A9%ED%95%9C-%ED%95%9C%EA%B5%AD%EC%96%B4-%ED%98%90%EC%98%A4-%ED%91%9C%ED%98%84-%ED%83%90%EC%A7%80-6.-%EB%8D%B0%EC%9D%B4%ED%84%B0-%EC%A6%9D%EA%B0%95)로 놀러오세요!🤗

![스크린샷 2022-09-09 오후 5 38 47](https://user-images.githubusercontent.com/50821216/189335539-524842e2-c793-48f7-ac88-c7c5145f9660.png)


## 실행 방법

현재 환경의 모든 GPU를 사용하여 학습하는 방법:
```sh
deepspeed --num_gpus={총 gpu의 개수} ds_main.py <ds_main.py의 args> --deepspeed_config ds_config.json
```

특정 GPU만을 사용하는 방법(CUDA_VISIBLE_DEVICES로 제어 불가):
```sh
deepspeed --include localhost:<GPU_NUM1> <GPU_NUM2> ds_main.py <ds_main.py의 args> --deepspeed_config ds_config.json
```


## 데이터셋
비정제 데이터만을 대상으로 하기 위해서 아래와 같이 3가지 데이터를 사용하여 학습을 진행했습니다.
- [Korean Hate Speech Dataset](https://github.com/kocohub/korean-hate-speech)
- [Korean Unsmile Dataset](https://github.com/smilegate-ai/korean_unsmile_dataset)
- [Naver Sentiment Movie Corpus](https://github.com/e9t/nsmc)


## 하이퍼파라미터
```
{"scheduler_name": "linear",
"max_stop_number": 5,
"train_batch_size": 128,
"eval_batch_size": 128,
"max_seq_len": 100,
"learning_rate": 0.0002,
"num_train_epochs": 30,
"weight_decay": 0.01,
"eps": 1e-06}
```

## Reference
- [KcELECTRA](https://github.com/Beomi/KcELECTRA)
- [Korean Hate Speech Dataset](https://github.com/kocohub/korean-hate-speech)
- [Korean Unsmile Dataset](https://github.com/smilegate-ai/korean_unsmile_dataset)
- [Naver Sentiment Movie Corpus](https://github.com/e9t/nsmc)
- [Huggingface Transformers](https://github.com/huggingface/transformers)
- [한국어 상호참조해결을 위한 BERT 기반 데이터 증강 기법](https://koreascience.kr/article/CFKO202030060835857.pub?&lang=ko&orgId=sighlt)



## 📞
seoyeon9695@gmail.com
