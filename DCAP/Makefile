# 실험은 각 디렉터리에서 독립적으로 돌아갑니다.
# 이 Makefile 은 편의용 단축키일 뿐입니다.

EXP ?= experiments/01-microbiome-tokenization

.PHONY: setup demo clean

setup:          ## 의존성 설치
	pip install -r requirements.txt

demo:           ## 합성 데이터로 EXP 전체 파이프라인 실행
	cd $(EXP) && bash run_all.sh

clean:          ## 생성물 삭제 (커밋 대상 아님)
	find experiments -type d -name data -prune -exec rm -rf {} + 2>/dev/null || true
	find experiments -type d -name runs -prune -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name __pycache__ -prune -exec rm -rf {} + 2>/dev/null || true
