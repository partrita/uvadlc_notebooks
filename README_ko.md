UvA 딥 러닝 튜토리얼
===========================

*참고: 노트북을 더 멋진 형식으로 보려면 RTD 웹사이트를 방문하세요: https://uvadlc-notebooks.readthedocs.io/en/latest/*

*코스 웹사이트*: https://uvadlc.github.io/<br>
*코스 에디션*: 2024년 가을 (10월 28일 - 12월 20일) - 최신 상태로 유지 중</br>
*녹화본*: [YouTube 재생 목록](<https://www.youtube.com/playlist?list=PLdlPlO1QhMiAkedeu0aJixfkknLRxk1nA>)</br>
*저자*: Phillip Lippe

올해 코스 에디션을 위해, 강의에서 배운 "이론"을 해당 구현을 통해 이해하는 데 도움이 되도록 설계된 일련의 Jupyter 노트북을 만들었습니다.
최적화 기법, 트랜스포머, 그래프 신경망 등 다양한 주제를 다룰 것입니다 (전체 목록은 아래 참조).
노트북은 자료를 이해하고 **PyTorch Lightning**을 포함한 **PyTorch** 프레임워크의 세부 정보를 가르치는 데 도움이 됩니다.
또한, 대체 프레임워크로 **JAX+Flax**로 노트북을 일대일로 번역하여 제공합니다.

노트북은 모든 그룹 튜토리얼 세션의 첫 시간에 제공됩니다.
튜토리얼 세션 동안 노트북의 내용과 구현을 설명합니다.
채워진 노트북을 그냥 보거나, 직접 시도해 보거나, 실습 세션 동안 코딩을 따라 할지 스스로 결정할 수 있습니다.
노트북은 점수가 매겨지거나 유사한 방식으로 평가되는 필수 과제의 일부가 아닙니다.
그러나 노트북에 익숙해지고 직접 실험하거나 확장해 보는 것이 좋습니다.
또한, 제공된 내용은 채점 과제 및 시험과 관련이 있습니다.

튜토리얼은 PyTorch Lightning의 공식 튜토리얼로 통합되었습니다.
따라서 [해당 문서](https://pytorch-lightning.readthedocs.io/en/latest/)에서도 볼 수 있습니다.

노트북 실행 방법
------------------------

이 웹사이트에서는 HTML 형식으로 내보낸 노트북을 찾아 원하는 장치에서 읽을 수 있습니다. 그러나 직접 실행해 보는 것이 좋습니다. 노트북을 실행하는 세 가지 주요 권장 방법은 다음과 같습니다.

- **CPU에서 로컬로 실행**: 모든 노트북은 이 웹사이트를 빌드하는 github 저장소에도 저장되어 있습니다. 여기에서 찾을 수 있습니다: https://github.com/phlippe/uvadlc_notebooks/tree/master/docs/tutorial_notebooks. 노트북은 GPU 없이 일반적인 노트북에서 실행할 수 있도록 설계되었습니다. 노트북을 실행할 때 자동으로 다운로드되거나 이 [Google Drive](https://drive.google.com/drive/folders/1SevzqrkhHPAifKEHo-gi7J-dVxifvs4c?usp=sharing)에서 수동으로 다운로드할 수 있는 사전 훈련된 모델을 제공합니다. 사전 훈련된 모델 및 데이터 세트에 필요한 디스크 공간은 1GB 미만입니다. 올바른 파이썬 패키지가 모두 설치되었는지 확인하기 위해 [동일한 저장소](https://github.com/phlippe/uvadlc_notebooks/blob/master/)에 conda 환경을 제공합니다 (시스템에 따라 CPU 또는 GPU 버전 선택).

- **Google Colab**: 컴퓨터가 아닌 다른 플랫폼에서 노트북을 실행하거나 GPU 지원을 실험하고 싶다면 [Google Colab](https://colab.research.google.com/notebooks/intro.ipynb#recent=true)을 사용하는 것이 좋습니다. 이 문서 웹사이트의 각 노트북에는 Google Colab에서 열 수 있는 링크가 있는 배지가 있습니다. 노트북을 실행하기 전에 GPU 지원을 활성화하는 것을 잊지 마십시오 (`런타임 -> 런타임 유형 변경`). 각 노트북은 독립적으로 실행할 수 있으며 Google Drive 등을 연결할 필요가 없습니다. 그러나 세션을 닫을 때 로컬 컴퓨터에 저장하지 않거나 사전에 노트북을 Google Drive에 복사하지 않으면 변경 사항이 손실될 수 있습니다.

- **Snellius 클러스터**: 노트북을 기반으로 자체 (더 큰) 신경망을 훈련하려면 Snellius 클러스터를 사용할 수 있습니다. 그러나 이는 실제로 새 모델을 훈련하고 모델의 토론 및 분석을 진행하기 위해 다른 두 가지 옵션을 사용하는 경우에만 권장됩니다. Snellius는 학생 계정으로 gpu_shared 파티션에서 직접 Jupyter 노트북을 실행하는 것을 허용하지 않을 수 있습니다. 대신 `jupyter nbconvert --to script ...ipynb`를 사용하여 먼저 노트북을 스크립트로 변환한 다음 Snellius에서 스크립트를 실행하기 위한 작업을 시작할 수 있습니다. Snellius에서 실행할 때 몇 가지 조언:
   - 노트북의 tqdm 문을 비활성화합니다. 그렇지 않으면 slurm 출력 파일이 오버플로되어 몇 MB가 될 수 있습니다. PyTorch Lightning에서는 트레이너에서 `progress_bar_refresh_rate=0`을 설정하여 이 작업을 수행할 수 있습니다.
   - matplotlib 플로팅 문을 주석 처리하거나 :code:`plt.show()`를 `plt.savefig(...)`로 변경합니다.

튜토리얼-강의 정렬
--------------------------

강의 전반에 걸쳐 모든 영역의 내용을 다루기 위해 코스에서 7개의 튜토리얼을 논의할 것입니다. 주제에 따라 튜토리얼을 강의와 정렬할 수 있습니다. 튜토리얼 목록은 다음과 같습니다.

- 가이드 1: Snellius 클러스터 작업
- 튜토리얼 2: PyTorch 소개
- 튜토리얼 3: 활성화 함수
- 튜토리얼 4: 최적화 및 초기화
- 튜토리얼 5: 인셉션, ResNet 및 DenseNet
- 튜토리얼 6: 트랜스포머 및 멀티 헤드 어텐션
- 튜토리얼 7: 그래프 신경망
- 튜토리얼 8: 딥 에너지 모델
- 튜토리얼 9: 오토인코더
- 튜토리얼 10: 적대적 공격
- 튜토리얼 11: 이미지 모델링의 정규화 흐름
- 튜토리얼 12: 자기 회귀 이미지 모델링
- 튜토리얼 15: 비전 트랜스포머
- 튜토리얼 16: 메타 학습 - 학습 방법 학습
- 튜토리얼 17: SimCLR을 사용한 자기 지도 대조 학습


피드백, 질문 또는 기여
------------------------------------

딥 러닝 과정에서 이러한 튜토리얼을 제공하는 것은 이번이 처음입니다. 다른 프로젝트와 마찬가지로 사소한 버그와 문제가 발생할 수 있습니다. 오타, 구현 버그 또는 노트북 개선/추가 제안 등 학생들의 모든 피드백에 감사드립니다. 피드백을 제출하려면 다음 [링크](https://docs.google.com/forms/d/e/1FAIpQLSeIhwrFSHlDSWGAgCN-RcTKm7Sn7P6bxzIyzIGge6xId1K8DQ/viewform?usp=sf_link)를 사용하거나 메일(p dot lippe at uva dot nl)로 직접 저에게 연락하거나 TA 세션 중에 저를 찾아오십시오.

튜토리얼이 도움이 되었고 인용하고 싶다면 다음 bibtex를 사용할 수 있습니다.
```bibtex
@misc{lippe2024uvadlc,
   title        = {{UvA Deep Learning Tutorials}},
   author       = {Phillip Lippe},
   year         = 2024,
   howpublished = {\url{https://uvadlc-notebooks.readthedocs.io/en/latest/}}
}
```
