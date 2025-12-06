"""地理空間分析モジュール

Haversine距離計算、主要都市座標定義等
"""

import numpy as np
from typing import Dict, Tuple


# 主要都市座標定義
MAJOR_CITIES: Dict[str, Tuple[float, float]] = {
    '東京駅': (35.681236, 139.767125),
    '新宿駅': (35.689592, 139.700464),
    '渋谷駅': (35.658034, 139.701636),
    '大阪駅': (34.702485, 135.495951),
    '名古屋駅': (35.170915, 136.881537),
    '福岡駅': (33.589542, 130.420841)
}


def haversine_distance(
    lat1: float,
    lon1: float,
    lat2: float,
    lon2: float
) -> float:
    """
    2点間のHaversine距離（km）を計算

    Parameters
    ----------
    lat1, lon1 : float
        地点1の緯度・経度
    lat2, lon2 : float
        地点2の緯度・経度

    Returns
    -------
    float
        距離（km）
        入力にNaNが含まれる場合はNaNを返す

    Examples
    --------
    >>> # 東京駅 - 新宿駅（約7km）
    >>> distance = haversine_distance(35.681236, 139.767125, 35.689592, 139.700464)
    >>> 6 < distance < 8
    True

    >>> # 同一地点
    >>> haversine_distance(35.681236, 139.767125, 35.681236, 139.767125)
    0.0

    >>> # NaN入力
    >>> import numpy as np
    >>> result = haversine_distance(np.nan, 139.767125, 35.689592, 139.700464)
    >>> np.isnan(result)
    True
    """
    # NaNチェック
    if any(np.isnan([lat1, lon1, lat2, lon2])):
        return np.nan

    R = 6371  # 地球の半径 (km)

    lat1_rad = np.radians(lat1)
    lat2_rad = np.radians(lat2)
    delta_lat = np.radians(lat2 - lat1)
    delta_lon = np.radians(lon2 - lon1)

    a = (np.sin(delta_lat/2)**2 +
         np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(delta_lon/2)**2)
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))

    return R * c


def calculate_distances_to_major_cities(
    lat: float,
    lon: float
) -> Dict[str, float]:
    """
    主要都市までの距離を一括計算

    Parameters
    ----------
    lat, lon : float
        対象地点の緯度・経度

    Returns
    -------
    dict
        {'東京駅': distance, '新宿駅': distance, ...}
        NaN入力の場合は全てNaN

    Examples
    --------
    >>> # 東京駅からの距離
    >>> distances = calculate_distances_to_major_cities(35.681236, 139.767125)
    >>> distances['東京駅']
    0.0
    >>> 6 < distances['新宿駅'] < 8
    True
    """
    distances = {}

    for city_name, (city_lat, city_lon) in MAJOR_CITIES.items():
        distance = haversine_distance(lat, lon, city_lat, city_lon)
        distances[city_name] = distance

    return distances
