# 開発機の用意

新しい開発機でこのリポジトリの作業を始めるための手順（T139、2026-09-26）。AGENTS.md と TODO.md と、この文書だけで始められるように書いてある。エージェントの記憶（`~/.claude/…/memory/`）は機械ごとで、移ると無くなるので写さない。持ち主から受けた決まりは AGENTS.md の「持ち主の指示」にある。

CI（`.github/workflows/deploy.yml`）が毎回やっていることと同じ手順なので、迷ったらそちらを見る。

## 1. 入れるもの

- **git と gh**: `gh auth login`（持ち主がする。push とワークフローの起動に使う）。git の `user.name` と `user.email` は持ち主のもの。
- **Node 24**: `.nvmrc` の版。nvm などで。`npm` が付いてくる。
- **Python 3**（3.14 で確かめた）と NumPy・pytest・tokenizers・regex: `pip install numpy pytest tokenizers regex`（deploy.yml と同じ）。ディストリビューションのパッケージでもよい。Ubuntu では venv に（下）。
- **wget**: `make models` がモデルを取る。
- **Claude Code**。

### Ubuntu 26.04 LTS（OCI の Ampere A1、arm64）の場合

**この手順で a1-free を用意した（2026-09-26、T139 の段 6）**。書いたとおりに入らなかったのは 3 つで、どれも直した: transformers の版と protobuf（下の venv）、Playwright のブラウザ（下の「ブラウザ」）。

a1-free（Ubuntu 26.04.1、Neoverse-N1 × 2、11.9GB、スワップ 4GB、`/tmp` は tmpfs 5.9GB）には、持ち主が git・gh（ログイン済み）・Node v24.20.0（nvm）・Python 3.14.4・python3-numpy・python3-pytest・python3-regex・python3-jinja2・python3-sentencepiece（0.2.1）・python3-venv・wget・curl・gcc・make・Google Chrome 154 を入れてあった。T139 の段 6 で足したのは python3-protobuf、参照の venv、Playwright の Chromium だけ。エージェントの記憶のフォルダは写さない（中身は 2026-09-26 に AGENTS.md と TODO.md に移した）。

```sh
sudo apt update
sudo apt install -y git gh curl wget ca-certificates build-essential \
    python3 python3-venv python3-dev
```

- `gh` は Ubuntu の universe にある。無い・古いときは GitHub の apt のリポジトリから（https://cli.github.com/）。
- `build-essential` は必須ではない（C の計測に使ったことがある程度。NumPy と tokenizers は arm64 の wheel が出ている）。
- **Python のパッケージ**: Ubuntu の Python はシステムに `pip install` させない（PEP 668）。apt で足りるものは apt で、足りないものだけ下の 2 つの方法で。

#### 決めた形（持ち主の判断、2026-09-26）: apt で足りるものは apt、`tokenizers` と `transformers` だけ venv

毎回の試験（pytest・smoke・forward-check・build）に要るものは 26.04 の apt にある（a1-free には入っている）。sentencepiece の突き合わせ（T126）の分だけ足す:

```sh
sudo apt install python3-numpy python3-pytest python3-regex python3-jinja2 python3-sentencepiece python3-protobuf
```

`tokenizers`（pytest の本物との突き合わせ。無ければ system の python3 では `tests/test_bytebpe.py`（先頭の `importorskip("tokenizers")`）と `tests/test_llama3.py` の本物と比べる 61 件が走らない。表示は「62 skipped」。T144 までは test_llama3.py が test_bytebpe.py を import していて 185 件が走らず、うち 12 件は tokenizers と関係が無かった（RoPE の表など））と `transformers`（`tests/format_check.py` だけ）は 26.04 の apt に無いので、system の site-packages を見る小さな venv に入れる（apt の numpy・jinja2・sentencepiece・protobuf はそのまま使う。PyTorch は入らない、要らない）。**venv はリポジトリの中の `.venv`**（2026-09-26、持ち主の指示。`.gitignore` にある。pytest には `tests` を渡すので `.venv` の中は集めない）。下の 2 の clone の後、リポジトリの中で:

```sh
python3 -m venv --system-site-packages .venv
.venv/bin/pip install tokenizers==0.23.1 transformers==5.16.1
.venv/bin/python -m pytest tests -q        # 本物の tokenizers との突き合わせも走る（471 件）
.venv/bin/python tests/format_check.py .tmp/format-check hf-qwen3-0.6b
```

- **transformers は 5.16.1**（5.12.1 ではない）。PyPI の 5.12.1〜5.15.1 は `tokenizers<=0.23.0` を求め、import のときにも版を見て止まる（pip で 0.23.1 を上書きしても `ImportError`）。0.23.0 は PyPI に無く、0.23.1 を許すのは 5.16.0 から。前の開発機の 5.12.1 は AUR の `python-transformers-git`（git から作ったもの）で、この縛りが無かった。
- 5.16.1 で `format_check.py` は、記録のある 4 項目（Qwen3 0.6B・Rakuten 7B・Llama 3.2 1B・Swallow-MS）で記録と同じ結果になった。`tokenizer.model` のあるモデルでは参照が本物と違うことがある（AGENTS.md の `format_check.py` の行）。
- sentencepiece は apt の 0.2.1。T126 の突き合わせは 0.2.2 で取った。T139 のレビュー（2026-09-26）で 7 モデル（rinna japanese-gpt2 small、sarashina 0.5B instruct、CAT 0.8B、Rakuten mini、TinyLlama、Mistral v0.3、Swallow-MS）を BMP の全文字（「a」+ 文字 +「b」）と 4 つの文で比べ、0.2.1 と 0.2.2 の ID の列は全部同じだった。T126 の突き合わせの道具（16 モデル × BMP）はリポジトリに無い。
- **protobuf も要る**（apt の python3-protobuf）。transformers が sentencepiece の `tokenizer.model` を読むのに使い、無いと `format_check.py` は最初の `tokenizer.model` だけのモデル（sarashina）で止まる（tiktoken が無いという、関係の無いエラーで終わる）。前の開発機では sarashina も比べられていた（T138）。

<details><summary>調べて採らなかった形: <code>tokenizers</code> を 26.10 からピン留めで（2026-09-26、a1-free で入れずに調べた）</summary>

- 26.10（`stonking`）の universe に `python3-tokenizers` 0.23.1 と `python3-transformers` 5.12.1 がある（Debian の forky・sid も同じ版）。
- `tokenizers` は 26.04 の Python のまま入れられる: 26.10 の Python も 3.14、中身は安定 ABI の `tokenizers.abi3-aarch64-linux-gnu.so`、glibc は 2.34 まで。展開しただけの 26.10 の `.deb` を 26.04 の Python 3.14 で import できた。優先度 100 のピン留めと `--no-install-recommends` なら、26.10 から来るのはそれ 1 つ。
- `transformers` は apt では入らない: PyTorch（`python3-torch`・`python3-torchvision`）に依存し、26.04 の上では 26.10 の依存を全部許しても解けない（torchvision → libtorch → openmpi → libucc → ROCm で食い違う）。
- 片方は venv が要るなら両方 venv のほうが単純、というのが持ち主の判断。

</details>

#### venv に全部入れる（apt を使わない機械）

同じ `.venv` に、試験の分（pytest・regex）と参照の道具（`tests/requirements-reference.txt`）を全部。下の 2 の clone の後、リポジトリの中で:

```sh
python3 -m venv .venv
.venv/bin/pip install pytest regex -r tests/requirements-reference.txt
source .venv/bin/activate   # Makefile は PATH の python3 を呼ぶので、make models の前に
```

- **Node 24 は nvm で**（`.nvmrc` の版を確実に取るため。Ubuntu の `nodejs` の版は未確認）:

```sh
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/master/install.sh | bash
# 新しいシェルで
nvm install    # .nvmrc の版
```

- **ブラウザ**: Playwright 1.60 は Ubuntu 26.04 の arm64 を扱わない（`npx playwright-core install --with-deps chromium` は「Cannot install dependencies for ubuntu26.04-arm64」、`--with-deps` なしでも「does not support chromium on ubuntu26.04-arm64」で止まる）。24.04 の版を取らせれば入る:

```sh
PLAYWRIGHT_HOST_PLATFORM_OVERRIDE=ubuntu24.04-arm64 npx playwright-core install chromium
ldd ~/.cache/ms-playwright/chromium_headless_shell-*/chrome-linux/headless_shell | grep "not found"   # 何も出なければよい
```

  a1-free では足りない共有ライブラリは無かった。起動するとき（`tests/e2e.mjs`）はこの変数は要らない。
- **`systemd-run --user --scope -p MemoryMax=…`** は SSH のセッションで使えた（linger なし）。

## 2. リポジトリと生成物

```sh
git clone https://github.com/takano32/pyodide-llm.git
cd pyodide-llm
npm ci                 # AssemblyScript、Pyodide（Node 用）、playwright-core
make models kernels    # モデルの取得・変換・量子化と WASM SIMD カーネル（バイナリはコミットしない、方針 1）
npm install --no-save pyodide@latest   # smoke は最新の Pyodide で確かめる（ページと同じ）
```

`make models` は数百 MB を取り、数分かかる。`public/models/` と `*.bin` はコミットしない（`.gitignore`）。

## 3. 参照の道具（本物と突き合わせる検査だけ）

`tests/format_check.py`（書式）、sentencepiece の突き合わせ（T126）などは transformers と sentencepiece を使う。ページもデプロイも使わない。どちらの形でも 1 の `.venv` に入っている（入れ方の版は `tests/requirements-reference.txt`）。

```sh
.venv/bin/python tests/format_check.py .tmp/format-check hf-qwen3-0.6b
```

PyTorch は要らない（入れない。transformers は「PyTorch was not found」と言うだけで、語彙と書式には困らない）。

## 4. ブラウザ（手元で確かめるとき）

`npx playwright-core install chromium`（Firefox も同じ形）。**Ubuntu 26.04 の arm64 では WebKit は動かない**: 24.04 の版は ICU 74・libxml2.so.2・GTK 4 を求めるが、26.04 にあるのは ICU 78 と libxml2.so.16 で、`libicu74` は apt に無い（T139 のレビュー、「Host system is missing dependencies」）。WebKit は CI で。Firefox は上書きの変数で入った（起動は未確認）。Linux では `--with-deps` で依存のライブラリも（root が要る）。Ubuntu 26.04 の arm64 は上の「ブラウザ」のとおり 24.04 の版を取らせる。**メモリを先に見る**: Pyodide とモデルのブラウザは 400MB 以上、1B 級で数 GB（`free -m`）。a1-free では llm-jp-3 980M（ヒープ 1398MB）で機械全体の使用が 2.1GB → 5.3GB、Qwen2.5 3B（ヒープ 4050MB）で 2.1GB → 8.1GB。7〜8B（ヒープ 9.7〜11GB）は CI で。速さは手元で測らない（CI と持ち主の端末で。「開発機の値を既定にしない」）。

**Chrome DevTools MCP**（Claude Code からページを開いて、コンソールやネットワークを見る）: X サーバーの無い機械では `--headless` で登録する。a1-free では Google Chrome 154（`/usr/bin/google-chrome`）を使う:

```sh
claude mcp add chrome-devtools -s local -- bunx -y chrome-devtools-mcp@latest --headless --isolated
```

`bunx` は bun のもの（持ち主の登録。a1-free では `~/.bun` にある）。bun の無い機械では `npx -y chrome-devtools-mcp@latest …` の形（未確認）。

`--headless` が無いと「Missing X server to start the headful browser」で開かない（Xvfb の下で動かす手もあるが、要らなかった）。登録を変えたら Claude Code の `/mcp` で再接続する。a1-free で本番のページを開き、tiny-lm が準備完了になった（`crossOriginIsolated` が真、2 スレッド、コンソールにエラーと警告なし。2026-09-26）。

## 5. 機械のことを確かめる

AGENTS.md の落とし穴のいくつかは機械しだい。新しい機械では確かめて、AGENTS.md の該当の行を書き直す。

- `/tmp` が tmpfs（メモリ）か: `df -h /tmp`。tmpfs なら大きいもの（モデル、`dist` の写し）は `/tmp` に置かない。**作業ファイルはリポジトリの中の `.tmp/`**（`.gitignore` にある。2026-09-26、持ち主の指示）。`$HOME` の下には黙って作らない（AGENTS.md の「持ち主の指示」）。
- メモリの枠で計測を包めるか: `systemd-run --user --scope -p MemoryMax=1G -p MemorySwapMax=0 true`。使えなければ `ulimit -v` で。
- スワップの有無、メモリの量、コアの数と種類（`lscpu`）: AGENTS.md の「計測環境」の行に。

a1-free（2026-09-26）: `/tmp` は tmpfs 5.9GB、`systemd-run --user` の枠は使える、スワップ 4GB、11.9GB（持ち主の別の Claude Code のセッションと searxng の docker などで available は約 9.9GB）、Neoverse-N1 × 2（`asimddp` あり、`i8mm` なし）、OCI の東京（ap-tokyo-1）。

## 6. 用意できたかの確かめ方

deploy.yml と同じものが全部通れば用意できている。

```sh
.venv/bin/python -m pytest tests -q    # CI と同じ 478 件（2026-09-26、T144。system の python3 では 62 件が skip: 上の 1 の venv の節）
for t in bench summary-check models-check ladder-check kept-check coi-js-check; do node tests/$t.mjs; done
node tests/smoke.mjs
node tests/forward-check.mjs --rounds 1 --positions 128
node tests/forward-check.mjs stories260K tiny-lm --rounds 1 --positions 128 --plain
node tests/forward-check.mjs stories260K tiny-lm --rounds 1 --positions 128 --wide
npm run build
```

新しい機械で初めて通したときは、かかった時間（pytest、smoke、forward-check の tok/s、`make kernels`）を AGENTS.md に書く。前の開発機（ARM の big.LITTLE、Cortex-A78 × 4 + A55 × 4、約 6.6GB）では pytest 約 20 秒、smoke 約 8 秒（Qwen3 の検査を足した後、2026-09-26）。a1-free の時間は AGENTS.md の「開発機の移行（T139）」にある（上の 12 のコマンドは全部通った）。
