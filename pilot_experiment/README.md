# Pilot Experiment 🚀
- 실험 목적 : 초기 1회 추론에서 얻은 내부/외부 신호만으로, 이후 추가 계산이 필요한지와 어떤 compute mechanism이 적절한지 예측 가능한지 확인하기


### 실험 설계

| 항목             | 설정                                                         |
| -------------- | ---------------------------------------------------------- |
| **Dataset**    | GSM8K 500개                                                 |
| **Model**      | Phi-3.5-mini  |
| **Setting**    | Single-turn pilot                                          |
| **Feature 시점** | One-shot inference 직후                                      |
| **Multi-turn** | 지금은 학습/판단에 사용하지 않음. 나중 확장 가능하게 로그 구조만 고려                   |

### Compute Mechanism

| Mechanism             | 구현                                      |
| --------------------- | --------------------------------------- |
| **Repeated Sampling** | Best-of-N                               |
| **Self-Correction**   | Single-step Self-Revision / Self-Refine |

* 위 근거는 TTS/System-2 survey에서 반복적으로 등장하는 대표 메커니즘. 
* Repeated Sampling은 다양한 후보를 만드는 방식
* Self-Correction은 기존 답을 반성,수정하는 방식

### Offline Oracle 실행
- S0: One-shot
- S1: Best-of-N
- S2: Self-Revision

> Decision feature는 S0의 one-shot 정보만 사용한다.

> S1/S2는 label 생성을 위한 offline oracle로만 사용한다.

### 수집할 변수

| 범주                  | 변수                             |
| ------------------- | ------------------------------ |
| **Utility 관련 변수**   | Accuracy / Correct 여부          |
|                     | Model Calls                    |
|                     | Generated Tokens               |
|                     | Prompt Tokens                  |
|                     | Total Tokens (Prompt + Output) |
| **Decision Signal** | Hidden State                   |
|                     | Entropy                        |
|                     | Logit Margin                   |
|                     | Output Length                  |
|                     | (가능하면) Representation Norm     |
|                     | (가능하면) Effective Rank          |

### Label 정의

#### Stage 1 Label: 추가 계산 필요 여부

| One-shot | Best-of-N | Self-Revision | Stage 1               |
| -------- | --------- | ------------- | --------------------- |
| 정답       | -         | -             | No additional compute |
| 오답       | 정답        | 오답            | Need compute          |
| 오답       | 오답        | 정답            | Need compute          |
| 오답       | 정답        | 정답            | Need compute          |
| 오답       | 오답        | 오답            | Unresolved            |

> 초기 분석에서는 Unresolved는 별도 그룹으로 두고, Stage1 학습에서는 제외

#### Stage 2 Label: 어떤 mechanism이 적절한가?(미정)

Stage1이 Need compute인 샘플만 대상으로 함.

| Best-of-N | Self-Revision | Stage2            |
| --------- | ------------- | ----------------- |
| 정답        | 오답            | Repeated Sampling |
| 오답        | 정답            | Self-Correction   |
| 정답        | 정답            | Tie / Utility로 결정 |
| 오답        | 오답            | Unresolved        |

> 처음에는 Tie를 제외하고, 이후 utility 기준이 정해지면 tokens/calls 기준으로 더 싼 전략을 선택하도록 함


### Pilot에서 가장 먼저 볼 분석

| Case                   | 의미                        |
| ---------------------- | ------------------------- |
| **One-shot correct**   | 추가 계산 불필요                 |
| **Best-of-N only**     | Repeated Sampling이 해결한 문제 |
| **Self-Revision only** | Self-Correction이 해결한 문제   |
| **Both success**       | 둘 다 해결 가능                 |
| **Both fail**          | 현재 전략으로 해결 불가             |

#### Pilot 성공 기준
엄격한 기준은 아니지만, 아래 중 하나라도 보이면 계속 진행할 가치가 있음.

- Stage1 예측 AUC가 chance보다 높음.
- Hidden + external이 external only보다 좋음.
- Best-of-N only와 Self-Revision only가 모두 존재함.
- 두 mechanism의 성공 케이스가 완전히 겹치지 않음.


### Repository Structure
```
pilot_experiment/
│
├── README.md
├── configs/
│   ├── pilot_gsm8k_phi.yaml
│   ├── pilot_gsm8k_qwen.yaml
│   └── strategy_prompts.yaml
│
├── data/
│   ├── raw/
│   │   └── gsm8k/
│   ├── sampled/
│   │   └── gsm8k_500.jsonl
│   └── processed/
│       └── pilot_dataset.jsonl
│
├── prompts/
│   ├── one_shot.txt
│   ├── best_of_n.txt
│   └── self_revision.txt
│
├── scripts/
│   ├── 00_sample_dataset.py
│   ├── 01_run_one_shot.py
│   ├── 02_run_best_of_n.py
│   ├── 03_run_self_revision.py
│   ├── 04_extract_signals.py
│   ├── 05_build_labels.py
│   ├── 06_analyze_mechanism_overlap.py
│   └── 07_train_decision_probe.py
│
├── src/
│   ├── data_utils.py
│   ├── model_runner.py
│   ├── answer_parser.py
│   ├── correctness.py
│   ├── signal_extractor.py
│   ├── label_builder.py
│   └── metrics.py
│
├── outputs/
│   ├── one_shot/
│   │   ├── generations.jsonl
│   │   ├── hidden_states/
│   │   └── signals.csv
│   │
│   ├── best_of_n/
│   │   └── generations.jsonl
│   │
│   ├── self_revision/
│   │   └── generations.jsonl
│   │
│   ├── labels/
│   │   ├── stage1_labels.csv
│   │   ├── stage2_labels.csv
│   │   └── utility_variables.csv
│   │
│   └── analysis/
│       ├── mechanism_overlap.csv
│       ├── stage1_probe_results.csv
│       └── stage2_probe_results.csv
│
└── notebooks/
    ├── 01_check_label_distribution.ipynb
    ├── 02_signal_distribution.ipynb
    └── 03_probe_analysis.ipynb
```

- 핵심은 outputs/one_shot/
- hidden state와 decision signal은 반드시 one-shot에서만 저장하고, best_of_n, self_revision은 label 생성용 oracle 결과로 분리
- 가장 중요한 산출물은 아래 4개
```
outputs/one_shot/signals.csv
outputs/labels/utility_variables.csv
outputs/labels/stage1_labels.csv
outputs/labels/stage2_labels.csv
```

- 로그 row는 아래와 같이 설정하고, pilot에서는 turn_id =0만 사용
```
sample_id, turn_id, strategy, prompt, output, correct,
model_calls, prompt_tokens, output_tokens, total_tokens,
latency_ms, hidden_state_path
```
