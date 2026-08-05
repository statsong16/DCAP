.PHONY: setup test demo clean

setup:          ## 의존성 설치
	pip install -r requirements.txt

test:           ## 단위 테스트
	python -m pytest -q

demo:           ## 합성 자료로 전체 파이프라인
	bash run_all.sh

clean:          ## 생성물 삭제 (커밋 대상 아님)
	rm -rf data runs .pytest_cache
	find . -type d -name __pycache__ -prune -exec rm -rf {} + 2>/dev/null || true
