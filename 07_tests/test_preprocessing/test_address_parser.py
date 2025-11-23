"""AddressParserのテスト"""

import pytest
import polars as pl
from preprocessing.address_parser import AddressParser, PREFECTURE_MASTER


class TestPrefectureMaster:
    """都道府県マスタのテスト"""

    def test_prefecture_master_size(self):
        """都道府県マスタが47件あること"""
        assert len(PREFECTURE_MASTER) == 47

    def test_prefecture_master_keys(self):
        """都道府県コードが1-47の連番であること"""
        assert set(PREFECTURE_MASTER.keys()) == set(range(1, 48))

    def test_prefecture_master_sample_values(self):
        """主要都道府県の名称が正しいこと"""
        assert PREFECTURE_MASTER[1] == '北海道'
        assert PREFECTURE_MASTER[13] == '東京都'
        assert PREFECTURE_MASTER[27] == '大阪府'
        assert PREFECTURE_MASTER[47] == '沖縄県'


class TestAddressParser:
    """AddressParserのテスト"""

    @pytest.fixture
    def sample_data(self):
        """テスト用サンプルデータ"""
        return pl.DataFrame({
            'addr1_1': [13, 27, 23],
            'addr1_2': [101, 102, 224],
            'full_address': [
                '東京都千代田区丸の内1-1-1',
                '大阪府大阪市北区梅田1-1-1',
                '愛知県知多市八幡荒井105-1'
            ]
        })

    def test_init(self):
        """初期化が正しく行われること"""
        parser = AddressParser()
        assert parser.pref_master == PREFECTURE_MASTER
        assert parser.city_mapping_size == 0

    def test_build_city_mapping(self, sample_data):
        """市区町村マッピングが構築できること"""
        parser = AddressParser()
        parser.build_city_mapping(sample_data)

        # マッピングが構築されていること
        assert parser.city_mapping_size == 3

        # 正しくマッピングされていること
        assert parser.get_city_name(13, 101) == '千代田区'
        assert parser.get_city_name(27, 102) == '大阪市'
        assert parser.get_city_name(23, 224) == '知多市'

    def test_build_city_mapping_missing_columns(self):
        """必須カラムがない場合エラーになること"""
        parser = AddressParser()
        df = pl.DataFrame({'addr1_1': [13]})

        with pytest.raises(ValueError, match="Required columns"):
            parser.build_city_mapping(df)

    def test_add_address_columns(self, sample_data):
        """住所カラムが追加できること"""
        parser = AddressParser()
        parser.build_city_mapping(sample_data)

        result = parser.add_address_columns(sample_data)

        # 新しいカラムが追加されていること
        assert 'prefecture_name' in result.columns
        assert 'city_name' in result.columns

        # 都道府県名が正しいこと
        assert result['prefecture_name'].to_list() == ['東京都', '大阪府', '愛知県']

        # 市区町村名が正しいこと
        assert result['city_name'].to_list() == ['千代田区', '大阪市', '知多市']

    def test_add_address_columns_prefecture_only(self):
        """市区町村マッピングなしでも都道府県名は追加できること"""
        parser = AddressParser()
        df = pl.DataFrame({'addr1_1': [13, 27]})

        result = parser.add_address_columns(df)

        assert 'prefecture_name' in result.columns
        assert result['prefecture_name'].to_list() == ['東京都', '大阪府']
        # city_nameは追加されない（addr1_2がないため）
        assert 'city_name' not in result.columns

    def test_add_address_columns_missing_addr1_1(self):
        """addr1_1がない場合エラーになること"""
        parser = AddressParser()
        df = pl.DataFrame({'dummy': [1, 2, 3]})

        with pytest.raises(ValueError, match="Column 'addr1_1' is required"):
            parser.add_address_columns(df)

    def test_get_city_name_not_found(self):
        """存在しないコードの場合Noneを返すこと"""
        parser = AddressParser()
        assert parser.get_city_name(99, 999) is None

    def test_duplicate_city_codes(self):
        """同じ市区町村コードでも都道府県が異なれば別の市区町村として扱うこと"""
        parser = AddressParser()
        df = pl.DataFrame({
            'addr1_1': [13, 27],  # 東京都、大阪府
            'addr1_2': [101, 101],  # 同じコード
            'full_address': [
                '東京都千代田区丸の内1-1-1',
                '大阪府大阪市北区梅田1-1-1'
            ]
        })

        parser.build_city_mapping(df)

        # 異なる市区町村として扱われること
        assert parser.get_city_name(13, 101) == '千代田区'
        assert parser.get_city_name(27, 101) == '大阪市'
