import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import os

# ================= 1. 配置区域 =================
# 如果你的 CSV 文件和这个画图脚本在同一个文件夹，就保持 './'
# 如果在别的文件夹，改成对应的路径，比如 './Scratch_Results/'
DATA_DIR = './' 

# 定义文件路径和图例名称
models = ['Baseline (Real Only)', 'WGAN-GP (1:1)', 'Mini-DDPM (1:1)']

metrics_files = {
    'Baseline (Real Only)': os.path.join(DATA_DIR, 'Metrics_Exp_A_Baseline.csv'),
    'WGAN-GP (1:1)': os.path.join(DATA_DIR, 'Metrics_Exp_B_WGAN.csv'),
    'Mini-DDPM (1:1)': os.path.join(DATA_DIR, 'Metrics_Exp_C_DDPM.csv')
}

roc_files = {
    'Baseline (Real Only)': os.path.join(DATA_DIR, 'ROC_Data_Exp_A_Baseline.csv'),
    'WGAN-GP (1:1)': os.path.join(DATA_DIR, 'ROC_Data_Exp_B_WGAN.csv'),
    'Mini-DDPM (1:1)': os.path.join(DATA_DIR, 'ROC_Data_Exp_C_DDPM.csv')
}

# 顶会配图高级色板
colors = {
    'Baseline (Real Only)': '#7f8c8d',  # 高级灰
    'WGAN-GP (1:1)': '#2980b9',         # 学术蓝
    'Mini-DDPM (1:1)': '#c0392b'        # 警示红/高光红
}

linestyles = {
    'Baseline (Real Only)': '--',       # 基线用虚线
    'WGAN-GP (1:1)': '-',               # 增强用实线
    'Mini-DDPM (1:1)': '-'
}

# 全局字体大小设置
plt.rcParams.update({'font.size': 12, 'axes.labelsize': 14, 'axes.titlesize': 16})

# ================= 2. 图 1: Train Loss 曲线 =================
plt.figure(figsize=(8, 6), dpi=300)
for model_name in models:
    if os.path.exists(metrics_files[model_name]):
        df = pd.read_csv(metrics_files[model_name])
        plt.plot(df['Epoch'], df['Train_Loss'], label=model_name, 
                 color=colors[model_name], linestyle=linestyles[model_name], linewidth=2.5)

plt.title('Training Loss Dynamics (pretrained ResNet-18)')
plt.xlabel('Epochs')
plt.ylabel('Train Loss')
plt.legend(loc='upper right')
plt.grid(True, linestyle=':', alpha=0.7)
plt.tight_layout()
plt.savefig(os.path.join(DATA_DIR, 'Plot_1_Train_Loss.png'))
print("✅ 已生成: Plot_1_Train_Loss.png")

# ================= 3. 图 2: Validation Accuracy 曲线 (带高级平滑处理) =================
plt.figure(figsize=(8, 6), dpi=300)

# 设置平滑窗口大小 (数字越大越平滑，一般推荐 3 到 5)
window_size = 5 

for model_name in models:
    if os.path.exists(metrics_files[model_name]):
        df = pd.read_csv(metrics_files[model_name])
        
        # 原始数据乘以 100 变成百分比
        raw_acc = df['Test_Accuracy'] * 100
        
        # 计算滑动平均值 (平滑处理)
        smoothed_acc = raw_acc.rolling(window=window_size, min_periods=1).mean()
        
        # 步骤 1: 先画一条半透明的细线，展示真实的震荡数据 (防学术造假质疑)
        plt.plot(df['Epoch'], raw_acc, alpha=0.2, color=colors[model_name], linestyle='-', linewidth=1)
        
        # 步骤 2: 在上面画一条实打实的粗线，展示平滑后的完美趋势
        plt.plot(df['Epoch'], smoothed_acc, label=model_name, 
                 color=colors[model_name], linestyle=linestyles[model_name], linewidth=2.5)

plt.title('Validation Accuracy Dynamics (pretrained ResNet-18)')
plt.xlabel('Epochs')
plt.ylabel('Validation Accuracy (%)')
plt.legend(loc='lower right')
plt.grid(True, linestyle=':', alpha=0.7)
plt.tight_layout()
plt.savefig(os.path.join(DATA_DIR, 'Plot_2_Val_Accuracy.png'))
print("✅ 已生成: Plot_2_Val_Accuracy.png")

# ================= 4. 图 3: ROC 曲线 =================
plt.figure(figsize=(8, 8), dpi=300)
for model_name in models:
    if os.path.exists(roc_files[model_name]):
        df = pd.read_csv(roc_files[model_name])
        fpr, tpr, _ = roc_curve(df['True_Label'], df['Pred_Prob'])
        roc_auc = auc(fpr, tpr)
        
        plt.plot(fpr, tpr, label=f'{model_name} (AUC = {roc_auc:.4f})', 
                 color=colors[model_name], linestyle=linestyles[model_name], linewidth=2.5)

# 画对角线（随机猜测线）
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle=':')

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.title('ROC Curve (pretrained ResNet-18)')
plt.xlabel('False Positive Rate (1 - Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')
plt.legend(loc='lower right')
plt.grid(True, linestyle=':', alpha=0.7)
plt.tight_layout()
plt.savefig(os.path.join(DATA_DIR, 'Plot_3_ROC_Curve.png'))
print("✅ 已生成: Plot_3_ROC_Curve.png")

print("\n🎉 所有图表绘制完毕！请在当前目录下查看生成的 PNG 文件。")