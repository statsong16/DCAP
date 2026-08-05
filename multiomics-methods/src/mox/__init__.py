"""
mox — 다중오믹스 코호트 분석의 방법론 층.

이 패키지가 담고 있는 판단은 네 가지입니다.

  1. 분할   피험자(그룹) 단위로 나눈다. 반복측정 자료에서 표본 단위 분할은
            표현형이 아니라 사람을 외우게 만든다.            -> splits.py
  2. 전처리 블록마다 값의 성격이 다르므로 변환도 다르다. 그리고 파라미터를
            학습하는 전처리는 train fold 에서만 fit 한다.    -> preprocess.py
  3. 결측   무작위 결측과 검출한계 결측은 다른 문제다.
            중앙값으로 다 메우면 후자를 위로 끌어올린다.     -> qc.py, preprocess.py
  4. 평가   판별력만 보지 않는다. 보정을 함께 본다.          -> evaluate.py

각 판단의 근거는 docs/ 에 있습니다.
"""

from mox import cluster, evaluate, integrate, io, preprocess, qc, splits, tokenize

__all__ = ["cluster", "evaluate", "integrate", "io", "preprocess", "qc",
           "splits", "tokenize"]
__version__ = "0.1.0"
