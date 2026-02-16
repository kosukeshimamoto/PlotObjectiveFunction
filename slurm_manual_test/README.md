# SLURM Manual Test Folder

このフォルダは、手動でクラスターへアップロードして実行し、結果をダウンロードするための最小テストセットです。

## 含まれるファイル

- `run_plot.jl`: 目的関数を定義して `run_plot_objective` を呼び出す Julia スクリプト
- `run_plot.sbatch`: `sbatch` で投げる SLURM ジョブ定義
- `pipeline.sh`: 対話式で upload/download を実行するローカル用シェル

## ローカルで事前確認

```bash
POF_BACKEND=local POF_RESOLUTION=12 julia --project=. slurm_manual_test/run_plot.jl
```

## クラスターにアップロードして実行

### 対話式（推奨）

```bash
bash slurm_manual_test/pipeline.sh
```

起動後に次を順番に聞かれます。

- モード（upload+download / upload only / download only）
- `user@cluster`
- ローカル/リモートのプロジェクトパス
- 出力ディレクトリ

### 手動コマンド

1. ローカルからクラスターへアップロード

```bash
rsync -av --delete /Users/kosuke/Github/PlotObjectiveFunction/ user@cluster:/path/to/PlotObjectiveFunction/
```

2. クラスター上でジョブ投入

```bash
cd /path/to/PlotObjectiveFunction
sbatch slurm_manual_test/run_plot.sbatch
```

3. ジョブ状態確認

```bash
squeue -u "$USER"
```

4. ログ確認

```bash
tail -n 100 slurm-pof-manual-test-<jobid>.out
```

5. 生成物をローカルへダウンロード

```bash
rsync -av user@cluster:/path/to/PlotObjectiveFunction/slurm_manual_test/outputs/ /Users/kosuke/Github/PlotObjectiveFunction/slurm_manual_test/downloaded_outputs/
```

## よく使う環境変数

- `POF_RESOLUTION`: 解像度（既定 50）
- `POF_PARALLEL`: `manual` または `auto`（既定 `auto`）
- `POF_THREADS_PER_TASK`: `run_plot_objective` の `threads_per_task`
- `POF_BLAS_THREADS`: `run_plot_objective` の `blas_threads`
- `POF_USE_THREADS`: `true/false`（未指定なら自動判定）
- `POF_NUM_OBS`: 回帰データのサンプル数
- `POF_NUM_FEATURES`: 回帰データの特徴量数
- `POF_OUTDIR`: 出力先ディレクトリ

例:

```bash
sbatch --export=ALL,POF_RESOLUTION=40,POF_PARALLEL=manual,POF_THREADS_PER_TASK=8,POF_BLAS_THREADS=1,POF_USE_THREADS=true slurm_manual_test/run_plot.sbatch
```
