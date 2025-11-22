#!/bin/bash
# Markdown（Mermaid対応）をHTMLにエクスポート

echo "Mermaid対応のMarkdown→HTML変換を実行します..."

# 引数チェック
if [ $# -eq 0 ]; then
    echo "使用法: $0 <markdown_file.md>"
    echo "例: $0 docs/data_definition.md"
    exit 1
fi

INPUT_FILE="$1"
OUTPUT_FILE="${INPUT_FILE%.md}.html"

# ファイルの存在確認
if [ ! -f "$INPUT_FILE" ]; then
    echo "エラー: ファイルが見つかりません: $INPUT_FILE"
    exit 1
fi

# 必要なnpmパッケージの確認とインストール
echo "必要なパッケージを確認中..."
if ! command -v npx &> /dev/null; then
    echo "エラー: Node.js (npm/npx) がインストールされていません"
    echo "インストール: brew install node"
    exit 1
fi

# 一時的にグローバルパッケージを使用
echo "Markdown変換を実行中..."
npx -y markdown-it "$INPUT_FILE" \
    --no-html \
    -o "$OUTPUT_FILE"

# Mermaid対応のHTMLテンプレートで包む
echo "Mermaid対応HTMLを生成中..."
cat > "$OUTPUT_FILE.tmp" << 'EOF'
<!DOCTYPE html>
<html lang="ja">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Data Definition</title>
    <script type="module">
        import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.esm.min.mjs';
        mermaid.initialize({
            startOnLoad: true,
            theme: 'base',
            themeVariables: {
                fontSize: '16px',
                fontFamily: 'Arial, sans-serif'
            }
        });
    </script>
    <style>
        body {
            max-width: 1200px;
            margin: 40px auto;
            padding: 20px;
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Helvetica', Arial, sans-serif;
            line-height: 1.6;
            color: #24292e;
        }
        h1 { border-bottom: 2px solid #e1e4e8; padding-bottom: 10px; }
        h2 { border-bottom: 1px solid #e1e4e8; padding-bottom: 8px; margin-top: 24px; }
        h3 { margin-top: 20px; }
        table {
            border-collapse: collapse;
            width: 100%;
            margin: 20px 0;
        }
        th, td {
            border: 1px solid #dfe2e5;
            padding: 8px 12px;
            text-align: left;
        }
        th {
            background-color: #f6f8fa;
            font-weight: 600;
        }
        code {
            background-color: #f6f8fa;
            padding: 2px 6px;
            border-radius: 3px;
            font-family: 'SFMono-Regular', Consolas, 'Liberation Mono', Menlo, monospace;
        }
        pre {
            background-color: #f6f8fa;
            padding: 16px;
            border-radius: 6px;
            overflow-x: auto;
        }
        .mermaid {
            text-align: center;
            margin: 30px 0;
        }
        ul, ol {
            padding-left: 2em;
        }
        li {
            margin: 4px 0;
        }
    </style>
</head>
<body>
EOF

# markdownの変換内容を追加（mermaidブロックをmermaid divに変換）
sed 's/<pre><code class="language-mermaid">/<div class="mermaid">/g; s/<\/code><\/pre>/<\/div>/g' "$OUTPUT_FILE" >> "$OUTPUT_FILE.tmp"

cat >> "$OUTPUT_FILE.tmp" << 'EOF'
</body>
</html>
EOF

# 最終ファイルに置き換え
mv "$OUTPUT_FILE.tmp" "$OUTPUT_FILE"

echo "✅ 完了: $OUTPUT_FILE"
echo ""
echo "ブラウザで開くには："
echo "  open $OUTPUT_FILE"
