"""
preprocess.py — 블록별 값 변환과 결측 대치.

**fit 은 train fold 에서만 호출합니다.** 남길 feature 집합, pseudocount,
대치값, 평균/표준편차가 전부 fit 시점에 고정되고 transform 은 그것을 적용만
합니다. 이 구조가 아니면 "전체 자료로 스케일링한 뒤 교차검증" 같은 조용한
누수가 반드시 생깁니다.

블록 종류 (kind):

  compositional  상대풍부도. 존재율 필터 -> TSS -> pseudocount -> CLR.
                 CLR 로 폐쇄성(합=1)을 풀지 않으면 조성 자료 특유의 가짜 음의
                 상관이 그대로 모델에 들어갑니다.

  intensity      원시 강도(대사체 등). 자릿수가 여러 개 퍼져 있으므로 log10.

  log_abundance  이미 로그 스케일인 값(Olink NPX 등). 값 변환은 하지 않고
                 결측 처리와 표준화만 합니다.

결측 대치 (na_strategy):

  median         train 중앙값. 결측이 무작위일 때만 맞습니다.
  left_censored  train 의 낮은 분위수(기본 5%). 검출한계 결측일 때 맞습니다.
                 중앙값으로 메우면 낮은 값을 위로 끌어올려 편향을 만듭니다.
                 실제 LOD 값이 있으면 lod= 로 넘기세요. 그쪽이 우선입니다.
"""

from __future__ import annotations

import numpy as np

KINDS = ("compositional", "intensity", "log_abundance")
NA_STRATEGIES = ("median", "left_censored")

# 블록 이름 -> 기본 kind. 이름이 다르면 직접 지정하세요.
DEFAULT_KIND = {
    "microbiome": "compositional",
    "metabolome": "intensity",
    "proteome": "log_abundance",
}

# log_abundance 는 검출한계 결측이 기본입니다 (Olink NPX 의 below-LOD).
DEFAULT_NA_STRATEGY = {
    "compositional": "median",
    "intensity": "median",
    "log_abundance": "left_censored",
}


class BlockPrep:
    """
    한 블록의 전처리기. scikit-learn 의 fit/transform 규약을 따릅니다.

    파라미터
      kind            위의 KINDS 중 하나
      na_strategy     위의 NA_STRATEGIES 중 하나. None 이면 kind 기본값
      min_prevalence  compositional 에서 남길 최소 존재율
      max_na_frac     이 비율을 넘게 결측인 feature 는 버립니다
      censor_quantile left_censored 대치에 쓸 분위수
      lod             (p,) 실제 검출한계 값. 주면 censor_quantile 대신 씁니다
    """

    def __init__(self, kind: str, na_strategy: str | None = None,
                 min_prevalence: float = 0.10, max_na_frac: float = 0.20,
                 censor_quantile: float = 0.05, lod: np.ndarray | None = None):
        if kind not in KINDS:
            raise ValueError(f"kind 는 {KINDS} 중 하나여야 합니다: {kind!r}")
        na_strategy = na_strategy or DEFAULT_NA_STRATEGY[kind]
        if na_strategy not in NA_STRATEGIES:
            raise ValueError(f"na_strategy 는 {NA_STRATEGIES} 중 하나여야 합니다")
        self.kind = kind
        self.na_strategy = na_strategy
        self.min_prevalence = min_prevalence
        self.max_na_frac = max_na_frac
        self.censor_quantile = censor_quantile
        self.lod = None if lod is None else np.asarray(lod, dtype=float)

    # -- fit / transform ---------------------------------------------------
    def fit(self, X: np.ndarray) -> "BlockPrep":
        X = np.asarray(X, dtype=np.float64)
        self._select_features(X)
        if self.keep_.sum() == 0:
            raise ValueError(f"{self.kind}: 남는 feature 가 없습니다")

        Z = self._to_value(X)
        self.fill_ = self._fill_values(Z)
        Z = np.where(np.isnan(Z), self.fill_, Z)

        self.mean_ = Z.mean(axis=0)
        std = Z.std(axis=0)
        self.var_ok_ = std > 1e-12          # train 에서 정보가 없는 feature 제거
        self.std_ = np.where(self.var_ok_, std, 1.0)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        self._check_fitted()
        Z = self._to_value(np.asarray(X, dtype=np.float64))
        Z = np.where(np.isnan(Z), self.fill_, Z)
        return ((Z - self.mean_) / self.std_)[:, self.var_ok_]

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)

    @property
    def n_features_out(self) -> int:
        self._check_fitted()
        return int(self.var_ok_.sum())

    # -- 내부 --------------------------------------------------------------
    def _check_fitted(self):
        if not hasattr(self, "keep_"):
            raise RuntimeError("fit 을 먼저 호출하세요")

    def _select_features(self, X: np.ndarray):
        na_frac = np.isnan(X).mean(axis=0)
        keep = na_frac <= self.max_na_frac
        if self.kind == "compositional":
            prevalence = (np.nan_to_num(X) > 0).mean(axis=0)
            keep &= prevalence >= self.min_prevalence
            nz = X[:, keep]
            nz = nz[np.isfinite(nz) & (nz > 0)]
            # 0 을 로그로 보내지 않기 위한 pseudocount. 관행적으로 비영 최솟값의 절반.
            self.pseudo_ = float(nz.min() / 2.0) if nz.size else 1e-9
        self.keep_ = keep

    def _to_value(self, X: np.ndarray) -> np.ndarray:
        Xk = X[:, self.keep_]
        if self.kind == "compositional":
            # 표본마다 총합이 다르므로(시퀀싱 깊이) 먼저 TSS 로 맞춥니다.
            s = np.nansum(Xk, axis=1, keepdims=True)
            s[s <= 0] = 1.0
            L = np.log(np.nan_to_num(Xk) / s + self.pseudo_)
            return L - L.mean(axis=1, keepdims=True)       # CLR
        if self.kind == "intensity":
            return np.log10(np.where(Xk < 0, 0.0, Xk) + 1.0)
        return Xk                                          # log_abundance

    def _fill_values(self, Z: np.ndarray) -> np.ndarray:
        if self.na_strategy == "left_censored":
            if self.lod is not None:
                fill = self.lod[self.keep_]
            else:
                with np.errstate(invalid="ignore"):
                    fill = np.nanquantile(Z, self.censor_quantile, axis=0)
        else:
            with np.errstate(invalid="ignore"):
                fill = np.nanmedian(Z, axis=0)
        # 관측값이 하나도 없는 feature 는 채울 근거가 없습니다. 0(표준화 후 평균)으로 두고
        # var_ok_ 가 어차피 걸러냅니다.
        return np.where(np.isnan(fill), 0.0, fill)


def make_preps(block_names, kind_map=None, **kwargs) -> dict:
    """블록 이름 목록으로 BlockPrep 들을 한 번에 만듭니다."""
    kind_map = {**DEFAULT_KIND, **(kind_map or {})}
    out = {}
    for name in block_names:
        if name not in kind_map:
            raise ValueError(f"'{name}' 의 kind 를 알 수 없습니다. "
                             f"kind_map 으로 지정하세요 (KINDS={KINDS})")
        out[name] = BlockPrep(kind_map[name], **kwargs)
    return out
