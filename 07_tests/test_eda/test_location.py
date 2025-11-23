"""地理空間分析のテスト"""

import pytest
import numpy as np
import sys
from pathlib import Path

# プロジェクトルートをPythonパスに追加
project_root = Path(__file__).parent.parent.parent / "04_src"
sys.path.insert(0, str(project_root))

from eda.location import haversine_distance, calculate_distances_to_major_cities, MAJOR_CITIES


def test_haversine_same_location():
    """同一地点の距離は0"""
    distance = haversine_distance(35.681236, 139.767125, 35.681236, 139.767125)
    assert distance == 0.0


def test_haversine_tokyo_shinjuku():
    """東京駅-新宿駅の距離（約7km）"""
    distance = haversine_distance(35.681236, 139.767125, 35.689592, 139.700464)
    assert 6 < distance < 8, f"Expected ~7km, got {distance:.2f}km"


def test_haversine_tokyo_osaka():
    """東京駅-大阪駅の距離（約400km）"""
    distance = haversine_distance(35.681236, 139.767125, 34.702485, 135.495951)
    assert 400 < distance < 420, f"Expected ~410km, got {distance:.2f}km"


def test_haversine_with_nan_lat1():
    """lat1がNaNの場合はNaNを返す"""
    distance = haversine_distance(np.nan, 139.767125, 35.689592, 139.700464)
    assert np.isnan(distance)


def test_haversine_with_nan_lon1():
    """lon1がNaNの場合はNaNを返す"""
    distance = haversine_distance(35.681236, np.nan, 35.689592, 139.700464)
    assert np.isnan(distance)


def test_haversine_with_nan_lat2():
    """lat2がNaNの場合はNaNを返す"""
    distance = haversine_distance(35.681236, 139.767125, np.nan, 139.700464)
    assert np.isnan(distance)


def test_haversine_with_nan_lon2():
    """lon2がNaNの場合はNaNを返す"""
    distance = haversine_distance(35.681236, 139.767125, 35.689592, np.nan)
    assert np.isnan(distance)


def test_haversine_southern_hemisphere():
    """南半球でも動作"""
    # Sydney - Melbourne（約700km）
    distance = haversine_distance(-33.8688, 151.2093, -37.8136, 144.9631)
    assert 700 < distance < 750, f"Expected ~720km, got {distance:.2f}km"


def test_haversine_western_hemisphere():
    """西半球でも動作"""
    # New York - Los Angeles（約3900km）
    distance = haversine_distance(40.7128, -74.0060, 34.0522, -118.2437)
    assert 3900 < distance < 4000, f"Expected ~3940km, got {distance:.2f}km"


def test_haversine_negative_and_positive():
    """負の座標と正の座標が混在しても動作"""
    # Sydney（南半球・東半球） - London（北半球・西半球）
    distance = haversine_distance(-33.8688, 151.2093, 51.5074, -0.1278)
    assert 16900 < distance < 17100, f"Expected ~17000km, got {distance:.2f}km"


def test_calculate_distances_to_major_cities_tokyo_station():
    """東京駅からの距離を計算"""
    distances = calculate_distances_to_major_cities(35.681236, 139.767125)

    # 東京駅からの距離は0
    assert distances['東京駅'] == 0.0
    # 新宿駅は約7km
    assert 6 < distances['新宿駅'] < 8
    # 大阪駅は約400km
    assert 400 < distances['大阪駅'] < 420


def test_calculate_distances_to_major_cities_all_cities():
    """全ての主要都市が含まれる"""
    distances = calculate_distances_to_major_cities(35.681236, 139.767125)

    # 全都市のキーが存在
    for city_name in MAJOR_CITIES.keys():
        assert city_name in distances


def test_calculate_distances_to_major_cities_with_nan():
    """NaN入力の場合は全てNaN"""
    distances = calculate_distances_to_major_cities(np.nan, 139.767125)

    # 全ての距離がNaN
    for city_name, distance in distances.items():
        assert np.isnan(distance), f"{city_name} should be NaN but got {distance}"


def test_major_cities_constant():
    """MAJOR_CITIES定数が正しく定義されている"""
    # 必要な都市が存在
    assert '東京駅' in MAJOR_CITIES
    assert '新宿駅' in MAJOR_CITIES
    assert '渋谷駅' in MAJOR_CITIES
    assert '大阪駅' in MAJOR_CITIES
    assert '名古屋駅' in MAJOR_CITIES
    assert '福岡駅' in MAJOR_CITIES

    # 各都市の座標が(lat, lon)のタプル
    for city_name, coords in MAJOR_CITIES.items():
        assert isinstance(coords, tuple)
        assert len(coords) == 2
        lat, lon = coords
        # 日本の範囲内（おおよそ）
        assert 20 < lat < 50, f"{city_name} lat out of range: {lat}"
        assert 120 < lon < 150, f"{city_name} lon out of range: {lon}"
