#!/bin/bash
# テスト実行スクリプト
#
# 使い方:
#   ./scripts/run_tests.sh [test_module] [--coverage]
#
# 例:
#   ./scripts/run_tests.sh                    # 全テスト実行
#   ./scripts/run_tests.sh test_data          # test_data配下のみ実行
#   ./scripts/run_tests.sh test_utils         # test_utils配下のみ実行
#   ./scripts/run_tests.sh test_data --coverage  # カバレッジ測定付き
#
# 環境変数でカスタマイズ可能:
#   TESTS_DIR       - テストディレクトリ名（デフォルト: tests または 07_tests を自動検出）
#   SRC_DIR         - ソースディレクトリ名（デフォルト: src または 04_src を自動検出）

set -e  # エラー時に終了

# カラー定義
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# プロジェクトルートに移動
cd "$(dirname "$0")/.."

# ディレクトリの自動検出または環境変数から取得
if [ -z "$TESTS_DIR" ]; then
    if [ -d "tests" ]; then
        TESTS_DIR="tests"
    elif [ -d "07_tests" ]; then
        TESTS_DIR="07_tests"
    else
        echo -e "${RED}✗ テストディレクトリが見つかりません (tests/ または 07_tests/)${NC}"
        exit 1
    fi
fi

if [ -z "$SRC_DIR" ]; then
    if [ -d "src" ]; then
        SRC_DIR="src"
    elif [ -d "04_src" ]; then
        SRC_DIR="04_src"
    else
        SRC_DIR="src"  # デフォルト値
    fi
fi

# 仮想環境の有効化
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
    echo -e "${GREEN}✓ 仮想環境を有効化${NC}"
else
    echo -e "${RED}✗ 仮想環境が見つかりません${NC}"
    exit 1
fi

# テスト対象の決定
TEST_TARGET="${1:-$TESTS_DIR}"  # デフォルトは全テスト
if [ "$1" != "" ]; then
    TEST_TARGET="$TESTS_DIR/$1"
fi

# テスト結果出力ディレクトリの決定
if [ "$1" != "" ]; then
    RESULT_DIR="$TESTS_DIR/$1/test_results"
else
    RESULT_DIR="$TESTS_DIR/test_results"
    mkdir -p "$RESULT_DIR"
fi

# タイムスタンプ
TIMESTAMP=$(date '+%Y%m%d_%H%M%S')

# テスト実行
echo -e "${YELLOW}=================================${NC}"
echo -e "${YELLOW}テスト実行: $TEST_TARGET${NC}"
echo -e "${YELLOW}=================================${NC}"
echo ""

# 結果ファイルのパス
RESULT_FILE="${RESULT_DIR}/test_result_${TIMESTAMP}.txt"
COVERAGE_FILE="${RESULT_DIR}/coverage_${TIMESTAMP}.txt"

# ヘッダーの追加
cat > "$RESULT_FILE" << EOF
================================================================================
テスト実行結果
================================================================================

実行日時: $(date '+%Y-%m-%d %H:%M:%S')
対象: $TEST_TARGET
実行コマンド: pytest $TEST_TARGET -v

================================================================================
テスト結果
================================================================================

EOF

# pytestの実行（詳細モード + 結果をファイルに追記）
if pytest "$TEST_TARGET" -v --tb=short 2>&1 | tee -a "$RESULT_FILE"; then
    TEST_STATUS="✓ PASSED"
    STATUS_COLOR=$GREEN
else
    TEST_STATUS="✗ FAILED"
    STATUS_COLOR=$RED
fi

# サマリの追加
cat >> "$RESULT_FILE" << EOF

================================================================================
サマリ
================================================================================

テスト状態: $TEST_STATUS
実行日時: $(date '+%Y-%m-%d %H:%M:%S')
結果ファイル: $RESULT_FILE

EOF

# カバレッジ測定（オプション）
if [ "$2" == "--coverage" ] || [ "$2" == "-c" ]; then
    echo ""
    echo -e "${YELLOW}=================================${NC}"
    echo -e "${YELLOW}カバレッジ測定中...${NC}"
    echo -e "${YELLOW}=================================${NC}"
    echo ""

    # カバレッジヘッダー
    cat > "$COVERAGE_FILE" << EOF
================================================================================
カバレッジ測定結果
================================================================================

実行日時: $(date '+%Y-%m-%d %H:%M:%S')
対象: $TEST_TARGET
実行コマンド: pytest $TEST_TARGET --cov=$SRC_DIR --cov-report=term-missing

================================================================================
カバレッジレポート
================================================================================

EOF

    # カバレッジ測定
    pytest "$TEST_TARGET" --cov="$SRC_DIR" --cov-report=term-missing 2>&1 | tee -a "$COVERAGE_FILE"

    # カバレッジサマリ
    cat >> "$COVERAGE_FILE" << EOF

================================================================================
サマリ
================================================================================

カバレッジ測定完了
実行日時: $(date '+%Y-%m-%d %H:%M:%S')
結果ファイル: $COVERAGE_FILE

EOF

    echo ""
    echo -e "${GREEN}✓ カバレッジレポート: $COVERAGE_FILE${NC}"
fi

# 結果表示
echo ""
echo -e "${YELLOW}=================================${NC}"
echo -e "${STATUS_COLOR}$TEST_STATUS${NC}"
echo -e "${YELLOW}=================================${NC}"
echo ""
echo -e "${GREEN}✓ テスト結果: $RESULT_FILE${NC}"
echo ""

# 最新の結果へのシンボリックリンクを作成
if [ "$1" != "" ]; then
    ln -sf "$(basename "$RESULT_FILE")" "${RESULT_DIR}/latest_result.txt"
    if [ "$2" == "--coverage" ] || [ "$2" == "-c" ]; then
        ln -sf "$(basename "$COVERAGE_FILE")" "${RESULT_DIR}/latest_coverage.txt"
    fi
else
    ln -sf "test_results/$(basename "$RESULT_FILE")" "$TESTS_DIR/latest_result.txt"
    if [ "$2" == "--coverage" ] || [ "$2" == "-c" ]; then
        ln -sf "test_results/$(basename "$COVERAGE_FILE")" "$TESTS_DIR/latest_coverage.txt"
    fi
fi

echo "最新結果へのリンク: ${RESULT_DIR}/latest_result.txt"
if [ "$2" == "--coverage" ] || [ "$2" == "-c" ]; then
    echo "最新カバレッジ: ${RESULT_DIR}/latest_coverage.txt"
fi

# 終了ステータスを返す
if [ "$TEST_STATUS" == "✓ PASSED" ]; then
    exit 0
else
    exit 1
fi
