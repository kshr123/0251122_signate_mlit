"""
exp010_low_price_improvement

低価格帯（特に広面積×築古）の予測精度改善を目的とした実験。

追加特徴量（4次元）:
- lp_area_value: 土地価値目安（lp_price × house_area）
- area_age_category: 面積×築年数カテゴリ（0,1,2の3段階）
- area_age_cat_te_pref: カテゴリ×都道府県TE
- area_age_cat_te_youto: カテゴリ×用途地域TE

ベース: exp009_landprice (CV MAPE: 12.53%)
"""
