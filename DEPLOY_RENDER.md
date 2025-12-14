# Deploy to Render — 手順

1. GitHub にリポジトリをプッシュします（既にある場合はスキップ）。

```bash
git add .
git commit -m "Add Render deploy config and UI tweak"
git push origin main
```

2. Render にログインして「New Web Service」を選択。
3. GitHub のリポジトリを選び、ブランチをセット。
4. Build Command と Start Command を確認（`render.yaml` があれば自動）。手動で設定する場合:

- Build Command: `pip install -r requirements.txt`
- Start Command: `streamlit run app_streamlit_v2.py --server.port $PORT --server.headless true`

5. 環境変数やシークレットが必要なら Render のダッシュボードで追加。
6. デプロイが完了したら公開 URL にアクセス。

注意:
- `easyocr` と `torch` を含むため、ビルド時間やインスタンスのメモリ要件が高くなる可能性があります。必要に応じてインスタンスプランを調整してください。
