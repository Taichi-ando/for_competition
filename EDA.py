#データ探索系
import pandas as pd
import numpy as np
from pathlib import Path

INPUT_DIR = "/Users/ando/project/obenkyo/input"
OUTPUT_DIR = "/Users/ando/project/obenkyo/output"
DF = pd.DataFrame()
TRAIN_DF = pd.DataFrame()
TEST_DF = pd.DataFrame()


##################
#pandas-profiling#
##################
from pandas_profiling import ProfileReport

report = ProfileReport(DF)
report.to_file(Path(OUTPUT_DIR, 'report.html'))


##########
#sweetviz#
##########
import sweetviz as sv

#日本語を使うおまじない
sv.config.config.set("General", "use_cjk_font", "1")
#trainとtestの比較ができる
report = sv.compare([TRAIN_DF, "train"], [TEST_DF, "test"])
report.show_html(Path(OUTPUT_DIR, "train_vs_test.html"), layout='widescreen', scale=0.7)


#######
#venn2#
#######
from matplotlib_venn import venn2
#%matplotlib inline

#trainとtestの比較ができる
columns = TEST_DF.columns # train には目的変数が含まれている
columns_num = len(columns)
n_cols = 4
n_rows = columns_num // n_cols + 1

fig, axes = plt.subplots(figsize=(n_cols*3, n_rows*3),
                         ncols=n_cols, nrows=n_rows)

for col, ax in zip(columns, axes.ravel()):
    venn2(
        subsets=(set(TRAIN_DF[col].unique()), set(TEST_DF[col].unique())),
        set_labels=('Train', 'Test'),
        ax=ax
    )
    ax.set_title(col)
    
fig.tight_layout()
