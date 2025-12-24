import pandas as pd
from typing import Iterable, Optional

TokenDict = dict[str, int]
CrossPair = tuple[str, str]


def _escape_token_part(value: object) -> str:
    # Token separator güvenliği: '=' ve '|' gibi ayraçları kaçır
    s = str(value)
    return s.replace("|", "%7C").replace("=", "%3D")


def to_feature_dict(
    df: pd.DataFrame,
    add_feature_cross: bool = True,
    cross_pairs: Optional[Iterable[CrossPair]] = None,
    skip_missing: bool = True,
) -> list[TokenDict]:
    """
    Base representation:
      {"col=value": 1}

    Feature cross:
      {"cross:colA=valA|colB=valB": 1}

    Notlar:
    - High-cardinality için FeatureHasher / hashing trick ile uyumludur.
    - Missing değerler (NaN/None) varsayılan olarak token üretilmeden atlanır.
    - cross_pairs verilmezse varsayılan 3 cross uygulanır.
    """
    excluded = {"id", "click"}
    feature_cols = [c for c in df.columns if c not in excluded]

    if cross_pairs is None:
        cross_pairs = [
            ("site_id", "app_id"),
            ("site_domain", "app_domain"),
            ("device_type", "device_conn_type"),
        ]

    # Kolon index map’ini bir kez çıkar (performans)
    col_index = {c: i for i, c in enumerate(feature_cols)}

    # Cross pair’ları sadece df’de varsa aktive et
    active_cross_pairs: list[tuple[CrossPair, int, int]] = []
    if add_feature_cross:
        for a, b in cross_pairs:
            ia = col_index.get(a)
            ib = col_index.get(b)
            if ia is not None and ib is not None:
                active_cross_pairs.append(((a, b), ia, ib))

    dicts: list[TokenDict] = []
    # itertuples hızlı; row bir tuple
    for row in df[feature_cols].itertuples(index=False, name=None):
        feats: TokenDict = {}

        # Base tokens
        for col, idx in col_index.items():
            val = row[idx]
            if skip_missing and pd.isna(val):
                continue
            feats[f"{col}={_escape_token_part(val)}"] = 1

        # Cross tokens
        if add_feature_cross and active_cross_pairs:
            for (a, b), ia, ib in active_cross_pairs:
                va = row[ia]
                vb = row[ib]
                if skip_missing and (pd.isna(va) or pd.isna(vb)):
                    continue
                feats[
                    f"cross:{a}={_escape_token_part(va)}|{b}={_escape_token_part(vb)}"
                ] = 1

        dicts.append(feats)

    return dicts



