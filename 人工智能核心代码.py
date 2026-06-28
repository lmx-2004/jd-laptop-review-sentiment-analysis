import matplotlib.pyplot as plt

# 设置中文字体，防止图表的标题、标签显示为方框
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']  
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示为方块的问题

import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')  # 忽略一些无关警告

# 可视化工具
import matplotlib.pyplot as plt
import seaborn as sns

# 词云（如未安装，运行 pip install wordcloud）
from wordcloud import WordCloud

# 设置美观的绘图样式
sns.set_style("whitegrid")

import pandas as pd
import os
import chardet  # 需要先安装：pip install chardet

# ========== 路径设置 ==========
data_dir = r"D:\Anaconda\Jingdong_Data_4000"
pos_path = os.path.join(data_dir, "pos")
neg_path = os.path.join(data_dir, "neg")

# ========== 编码检测函数 ==========
def detect_encoding(file_path):
    """自动检测文本文件的编码格式"""
    with open(file_path, 'rb') as f:
        raw_data = f.read(10000)  # 读取前10000字节用于判断
        result = chardet.detect(raw_data)
        return result['encoding']

# ========== 读取正面评论 ==========
data_list = []
print("正在读取正面评论...")
for file_name in os.listdir(pos_path):
    if file_name.endswith('.txt'):
        file_path = os.path.join(pos_path, file_name)
        try:
            # 自动检测编码后打开文件
            enc = detect_encoding(file_path)
            with open(file_path, 'r', encoding=enc) as f:
                content = f.read().strip()
                data_list.append([content, 1])   # 1 表示正面
        except Exception as e:
            # 如果自动检测的编码仍然出错，尝试用 gbk 兜底
            try:
                with open(file_path, 'r', encoding='gbk') as f:
                    content = f.read().strip()
                    data_list.append([content, 1])
            except Exception as e2:
                print(f"读取文件 {file_name} 出错: {e2}")

# ========== 读取负面评论 ==========
print("正在读取负面评论...")
for file_name in os.listdir(neg_path):
    if file_name.endswith('.txt'):
        file_path = os.path.join(neg_path, file_name)
        try:
            enc = detect_encoding(file_path)
            with open(file_path, 'r', encoding=enc) as f:
                content = f.read().strip()
                data_list.append([content, 0])   # 0 表示负面
        except Exception as e:
            # 兜底：用 gbk 再试一次
            try:
                with open(file_path, 'r', encoding='gbk') as f:
                    content = f.read().strip()
                    data_list.append([content, 0])
            except Exception as e2:
                print(f"读取文件 {file_name} 出错: {e2}")

# ========== 构建 DataFrame ==========
df = pd.DataFrame(data_list, columns=['review', 'label'])
print(f"\n数据加载完成！共加载 {len(df)} 条评论。")

# 检查分布
print("\n标签分布（0=负面，1=正面）：")
print(df['label'].value_counts())

# 3.1 查看前5条评论
print("前3条评论预览：")
print(df.head(3))

# 3.2 数据信息
print("\n数据信息：")
print(df.info())

# 3.3 数据集形状
print(f"\n数据集形状（行, 列）：{df.shape}")

# 3.4 标签分布（验证是否正负各约2000条）
print("\n标签分布（0=负面，1=正面）：")
print(df['label'].value_counts())

# 4.1 缺失值检查
null_counts = df.isnull().sum()
print("缺失值统计：")
print(null_counts)
if null_counts.sum() == 0:
    print("✅ 无缺失值，数据完整！")

# 4.2 重复值检查
dup_count = df.duplicated(subset=['review']).sum()
print(f"\n完全重复的评论条数：{dup_count}")

if dup_count > 0:
    dup_examples = df[df.duplicated(subset=['review'], keep=False)]
    print("重复评论示例（前3条）：")
    print(dup_examples.head(3))
    
    # 4.3 删除重复值
    initial_len = len(df)
    df = df.drop_duplicates(subset=['review'], keep='first').reset_index(drop=True)
    final_len = len(df)
    print(f"\n✅ 删除重复数据：{initial_len} -> {final_len} 条，共删除 {initial_len - final_len} 条")
else:
    print("✅ 无重复评论，数据干净！")

    # 计算每条评论的字符长度
df['review_length'] = df['review'].str.len()

# 整体统计描述
print("评论长度（字符数）统计描述：")
print(df['review_length'].describe())

# 按正负面分组统计
print("\n正面 / 负面评论平均长度对比：")
print(df.groupby('label')['review_length'].mean().round(2))

# 正负面长度差异的简单分析
pos_avg = df[df['label']==1]['review_length'].mean()
neg_avg = df[df['label']==0]['review_length'].mean()
print(f"\n负面评论平均比正面评论长 {neg_avg - pos_avg:.2f} 个字符")

import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False    # 用来正常显示负号
plt.figure(figsize=(8, 5))
counts = df['label'].value_counts().sort_index()

# 绘制柱状图
sns.barplot(x=counts.index, y=counts.values, palette=['#FF6B6B', '#4ECDC4'])
plt.xticks(ticks=[0, 1], labels=['负面 (0)', '正面 (1)'])
plt.title('情感标签分布情况', fontsize=15)
plt.xlabel('情感类别')
plt.ylabel('评论数量')
plt.grid(axis='y', alpha=0.3)

# 在柱子上显示数值
for i, v in enumerate(counts.values):
    plt.text(i, v + 5, str(v), ha='center', va='bottom', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.show()

plt.figure(figsize=(14, 5))

# 子图1：整体长度直方图
plt.subplot(1, 2, 1)
sns.histplot(df['review_length'], bins=30, kde=True, color='skyblue', edgecolor='black')
plt.title('所有评论长度分布直方图', fontsize=14)
plt.xlabel('评论长度（字符数）')
plt.ylabel('频数')
plt.axvline(df['review_length'].mean(), color='red', linestyle='--', label=f"均值: {df['review_length'].mean():.0f}")
plt.legend()

# 子图2：正负面评论长度箱线图
plt.subplot(1, 2, 2)
sns.boxplot(x='label', y='review_length', data=df, palette=['#FF6B6B', '#4ECDC4'])
plt.xticks(ticks=[0, 1], labels=['负面', '正面'])
plt.title('正负面评论长度对比箱线图', fontsize=14)
plt.xlabel('情感类别')
plt.ylabel('评论长度（字符数）')

plt.tight_layout()
plt.show()

def generate_wordcloud(text_list, title, colormap):
    """生成词云图"""
    if len(text_list) == 0:
        print(f"⚠️ {title} 无文本内容，跳过生成")
        return
    
    combined_text = ' '.join(text_list)
    
    # 尝试多个字体路径
    font_paths = [
        'C:/Windows/Fonts/simhei.ttf',   # 黑体
        'C:/Windows/Fonts/msyh.ttc',     # 微软雅黑
        'C:/Windows/Fonts/simsun.ttc',   # 宋体
    ]
    
    wc = None
    for font in font_paths:
        try:
            wc = WordCloud(
                font_path=font,
                background_color='white',
                width=800,
                height=400,
                max_words=100,
                colormap=colormap,
                random_state=42
            )
            break
        except:
            continue
    
    # 如果所有字体都失败，去掉 font_path 参数
    if wc is None:
        wc = WordCloud(
            background_color='white',
            width=800,
            height=400,
            max_words=100,
            colormap=colormap,
            random_state=42
        )
    
    wc.generate(combined_text)
    
    plt.figure(figsize=(10, 5))
    plt.imshow(wc, interpolation='bilinear')
    plt.title(title, fontsize=16, fontweight='bold')
    plt.axis('off')
    plt.tight_layout()
    plt.show()

# 提取正面和负面评论
pos_reviews = df[df['label'] == 1]['review'].tolist()
neg_reviews = df[df['label'] == 0]['review'].tolist()

print(f"正面评论数: {len(pos_reviews)}，负面评论数: {len(neg_reviews)}")

# 生成正面词云
generate_wordcloud(pos_reviews, '正面评论高频词云', 'Greens')

# 生成负面词云
generate_wordcloud(neg_reviews, '负面评论高频词云', 'Reds')


# 保存为 CSV 文件
output_path = r"D:\Anaconda\Jingdong_Data_4000\cleaned_reviews.csv"
df.to_csv(output_path, index=False, encoding='utf-8-sig')
print(f"✅ 清洗后的数据已保存至：{output_path}")

# 保存一份副本到当前工作目录（方便查找）
df.to_csv("cleaned_reviews.csv", index=False, encoding='utf-8-sig')
print("✅ 同时已保存到当前目录：cleaned_reviews.csv")

# 查看最终数据概览
print("\n" + "="*50)
print("最终数据集概览：")
print("="*50)
print(f"总样本数: {len(df)}")
print(f"特征列数: {len(df.columns)}")
print(f"列名: {df.columns.tolist()}")
print(f"\n标签分布:")
print(df['label'].value_counts())
print("\n前2条数据示例:")
print(df[['review', 'label', 'review_length']].head(2))

import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import chardet

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")

print("✅ 所有库导入成功！")

data_dir = r"D:\Anaconda\Jingdong_Data_4000"
pos_path = os.path.join(data_dir, "pos")
neg_path = os.path.join(data_dir, "neg")

# 检查路径
if not os.path.exists(pos_path):
    print(f"❌ 路径不存在: {pos_path}")
if not os.path.exists(neg_path):
    print(f"❌ 路径不存在: {neg_path}")

def read_file_with_fallback(file_path):
    """
    链式尝试多种编码读取文件，返回 (内容, 是否成功)
    """
    encodings = ['utf-8', 'gbk', 'gb2312', 'gb18030']
    
    for enc in encodings:
        try:
            with open(file_path, 'r', encoding=enc) as f:
                content = f.read().strip()
                # 检查是否为空或只包含空白字符
                if len(content) == 0:
                    return None, False, "空文件"
                return content, True, enc
        except UnicodeDecodeError:
            continue
        except Exception as e:
            return None, False, str(e)
    
    return None, False, "所有编码尝试均失败"

print("✅ 路径和读取函数准备完成！")

data_list = []
fail_records = []  # 记录失败的文件信息

# ===== 读取正面评论 =====
print("正在读取正面评论...")
pos_files = [f for f in os.listdir(pos_path) if f.endswith('.txt')]
total_pos = len(pos_files)

for file_name in pos_files:
    file_path = os.path.join(pos_path, file_name)
    content, success, info = read_file_with_fallback(file_path)
    
    if success and content is not None:
        data_list.append([content, 1])
    else:
        fail_records.append(("正面", file_name, info))

# ✅ 修正：正面成功数 = 总文件数 - 失败数
pos_fail_count = len([r for r in fail_records if r[0] == '正面'])
pos_success_count = total_pos - pos_fail_count
print(f"  正面: 成功 {pos_success_count} 条, 失败 {pos_fail_count} 条")

# ===== 读取负面评论 =====
print("正在读取负面评论...")
neg_files = [f for f in os.listdir(neg_path) if f.endswith('.txt')]
total_neg = len(neg_files)

for file_name in neg_files:
    file_path = os.path.join(neg_path, file_name)
    content, success, info = read_file_with_fallback(file_path)
    
    if success and content is not None:
        data_list.append([content, 0])
    else:
        fail_records.append(("负面", file_name, info))

# ✅ 修正：负面成功数 = 总文件数 - 失败数
neg_fail_count = len([r for r in fail_records if r[0] == '负面'])
neg_success_count = total_neg - neg_fail_count
print(f"  负面: 成功 {neg_success_count} 条, 失败 {neg_fail_count} 条")

# ===== 构建 DataFrame =====
df = pd.DataFrame(data_list, columns=['review', 'label'])

print(f"\n{'='*60}")
print(f"✅ 数据加载完成！")
print(f"   总样本数: {len(df)} 条")
print(f"   原始文件数: {total_pos + total_neg} 个")
print(f"   读取失败: {len(fail_records)} 个")
print(f"   成功率: {len(df)/(total_pos+total_neg)*100:.2f}%")
print(f"{'='*60}")

# ===== 打印失败文件统计（如有） =====
if len(fail_records) > 0:
    print("\n📋 失败原因统计：")
    fail_df = pd.DataFrame(fail_records, columns=['类别', '文件名', '原因'])
    print(fail_df['原因'].value_counts())

# ===== 标签分布 =====
print("\n标签分布（0=负面，1=正面）：")
print(df['label'].value_counts())

print("前3条评论预览：")
print(df.head(3))

print("\n数据信息：")
print(df.info())

print(f"\n数据集形状（行, 列）：{df.shape}")

# 4.1 缺失值
print("\n缺失值统计：")
print(df.isnull().sum())

if df.isnull().sum().sum() == 0:
    print("✅ 无缺失值！")

# 4.2 空字符串检查（针对可能的空评论）
empty_count = (df['review'].str.strip() == '').sum()
print(f"\n空字符串评论条数：{empty_count}")

if empty_count > 0:
    print("⚠️ 发现空字符串评论，予以删除...")
    df = df[df['review'].str.strip() != ''].reset_index(drop=True)
    print(f"删除后剩余：{len(df)} 条")

# 4.3 重复值
dup_count = df.duplicated(subset=['review']).sum()
print(f"\n完全重复的评论条数：{dup_count}")

if dup_count > 0:
    initial_len = len(df)
    df = df.drop_duplicates(subset=['review'], keep='first').reset_index(drop=True)
    print(f"✅ 删除重复数据：{initial_len} -> {len(df)} 条")
else:
    print("✅ 无重复评论！")

    # 计算评论长度
df['review_length'] = df['review'].str.len()

print("评论长度统计描述：")
print(df['review_length'].describe())

print("\n正面/负面评论平均长度：")
print(df.groupby('label')['review_length'].mean().round(2))

# 计算差值
pos_avg = df[df['label']==1]['review_length'].mean()
neg_avg = df[df['label']==0]['review_length'].mean()
print(f"\n正面评论平均长度: {pos_avg:.2f} 字符")
print(f"负面评论平均长度: {neg_avg:.2f} 字符")
print(f"差值（正面-负面）: {pos_avg - neg_avg:.2f} 字符")

import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False    # 用来正常显示负号
# 6.1 情感标签分布
plt.figure(figsize=(8, 5))
counts = df['label'].value_counts().sort_index()
colors = ['#FF6B6B', '#4ECDC4']
sns.barplot(x=counts.index, y=counts.values, palette=colors)
plt.xticks(ticks=[0, 1], labels=['负面 (0)', '正面 (1)'])
plt.title('情感标签分布情况', fontsize=15)
plt.xlabel('情感类别')
plt.ylabel('评论数量')
plt.grid(axis='y', alpha=0.3)
for i, v in enumerate(counts.values):
    plt.text(i, v + 5, str(v), ha='center', va='bottom', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.show()

# 6.2 评论长度分布
plt.figure(figsize=(14, 5))

plt.subplot(1, 2, 1)
sns.histplot(df['review_length'], bins=30, kde=True, color='skyblue', edgecolor='black')
plt.title('所有评论长度分布直方图', fontsize=14)
plt.xlabel('评论长度（字符数）')
plt.ylabel('频数')
plt.axvline(df['review_length'].mean(), color='red', linestyle='--', label=f"均值: {df['review_length'].mean():.0f}")
plt.legend()

plt.subplot(1, 2, 2)
sns.boxplot(x='label', y='review_length', data=df, palette=['#FF6B6B', '#4ECDC4'])
plt.xticks(ticks=[0, 1], labels=['负面', '正面'])
plt.title('正负面评论长度对比箱线图', fontsize=14)
plt.xlabel('情感类别')
plt.ylabel('评论长度（字符数）')

plt.tight_layout()
plt.show()

# 6.3 词云
def generate_wordcloud(text_list, title, colormap):
    if len(text_list) == 0:
        print(f"⚠️ {title} 无文本")
        return
    
    combined = ' '.join(text_list)
    
    # 尝试多个字体
    font_paths = ['C:/Windows/Fonts/simhei.ttf', 'C:/Windows/Fonts/msyh.ttc']
    wc = None
    for font in font_paths:
        try:
            wc = WordCloud(font_path=font, background_color='white', width=800, height=400,
                           max_words=100, colormap=colormap, random_state=42)
            break
        except:
            continue
    if wc is None:
        wc = WordCloud(background_color='white', width=800, height=400,
                       max_words=100, colormap=colormap, random_state=42)
    
    wc.generate(combined)
    plt.figure(figsize=(10, 5))
    plt.imshow(wc, interpolation='bilinear')
    plt.title(title, fontsize=16, fontweight='bold')
    plt.axis('off')
    plt.tight_layout()
    plt.show()

pos_reviews = df[df['label'] == 1]['review'].tolist()
neg_reviews = df[df['label'] == 0]['review'].tolist()
print(f"正面: {len(pos_reviews)} 条，负面: {len(neg_reviews)} 条")

generate_wordcloud(pos_reviews, '正面评论高频词云', 'Greens')
generate_wordcloud(neg_reviews, '负面评论高频词云', 'Reds')

output_path = r"D:\Anaconda\Jingdong_Data_4000\cleaned_reviews.csv"
df.to_csv(output_path, index=False, encoding='utf-8-sig')
print(f"✅ 已保存至：{output_path}")

df.to_csv("cleaned_reviews.csv", index=False, encoding='utf-8-sig')
print("✅ 已保存至当前目录：cleaned_reviews.csv")

print("\n" + "="*50)
print("最终数据集概览：")
print("="*50)
print(f"总样本数: {len(df)}")
print(f"特征列数: {len(df.columns)}")
print(f"列名: {df.columns.tolist()}")
print(f"\n标签分布:")
print(df['label'].value_counts())
print("\n前2条示例:")
print(df[['review', 'label', 'review_length']].head(2))

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# 文本处理
import jieba
import re

# 特征提取与预处理
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

print("✅ 所有库导入成功！")

# ========== 加载清洗后的数据 ==========
df = pd.read_csv('cleaned_reviews.csv')
print(f"✅ 加载数据：{len(df)} 条评论")
print(f"   列名：{df.columns.tolist()}")
print(f"\n标签分布：")
print(df['label'].value_counts())

import re
import jieba

# ========== 加载停用词 ==========
def load_stopwords():
    """加载停用词表, 包含哈工大停用词 + 电商场景自定义词"""
    base_stopwords = {
        '的', '了', '在', '是', '我', '有', '和', '就', '不', '人', '都', '一', '一个',
        '上', '也', '很', '到', '说', '要', '去', '你', '会', '着', '没有', '看', '好',
        '自己', '这', '那', '它', '他', '她', '我们', '你们', '他们', '什么', '怎么',
        '如何', '为什么', '因为', '所以', '但是', '而且', '或者', '虽然', '然后', '可以',
        '已经', '还是', '比较', '非常', '十分', '太', '更', '最', '多', '少', '嗯', '哦',
        '啊', '吧', '呢', '吗', '嘛', '啦', '呵呵', '哈哈', '嘿嘿',
        '让', '把', '给', '还', '没', '又', '从', '对', '向', '跟', '被',
        '能', '想', '知道', '觉得', '看到', '告诉', '问', '出来', '过来', '回去',
        '一下', '一样', '一点', '一些', '很多', '很少', '那种', '这个', '那个',
    }

    ecommerce_stopwords = {
        '商品', '产品', '东西', '店家', '卖家', '买家', '京东', '淘宝', '天猫',
        '真的', '感觉', '使用', '购买', '收到', '快递', '物流', '发货', '包装',
        '现在', '时候', '时间', '已经', '非常', '比较', '有点',
        '售后', '客服', '评价', '好评', '差评', '推荐', '大家', '希望', '建议',
    }

    return base_stopwords | ecommerce_stopwords


stopwords = load_stopwords()
print(f"✅ 停用词数量: {len(stopwords)} 个")


# ========== 分词 + 过滤 ==========
def cut_and_filter(text):
    """对文本进行分词并过滤停用词"""
    text = str(text)
    text = re.sub(r'[^\u4e00-\u9fff\w]', ' ', text)
    words = jieba.cut(text)

    filtered = []
    for w in words:
        w = w.strip()
        if not w:
            continue
        w_clean = re.sub(r'\s+', '', w)
        if not w_clean:
            continue
        if w_clean in stopwords:
            continue
        if len(w_clean) <= 1:
            continue
        if w_clean.isdigit():
            continue
        filtered.append(w_clean)

    return " ".join(filtered)


# ========== 执行分词 ==========
print("\n正在对评论进行分词处理...")

df['cut_review'] = df['review'].apply(cut_and_filter)
df['tokens'] = df['cut_review'].apply(lambda x: x.split())

empty_cut = df['cut_review'].isna().sum()
if empty_cut > 0:
    print(f"⚠️ 空分词结果条数: {empty_cut}")
    df.loc[df['cut_review'].isna(), 'cut_review'] = df.loc[df['cut_review'].isna(), 'review']
    df.loc[df['tokens'].isna(), 'tokens'] = df.loc[df['tokens'].isna(), 'review'].apply(lambda x: str(x).split())
    print("☑ 已用原始文本填充空分词结果")
else:
    print("✅ 空分词结果条数: 0")

print("\n分词结果示例(前2条):")
for i in range(2):
    print(f"原文: {df['review'].iloc[i]}")
    print(f"分词: {df['cut_review'].iloc[i]}")
    print("-" * 50)

    # ========== TF-IDF向量化 ==========
print("\n正在提取TF-IDF特征...")

tfidf = TfidfVectorizer(
    max_features=5000,           # 保留最重要的5000个特征词
    ngram_range=(1, 2),          # 使用一元词和二元词组合
    min_df=5,                    # 忽略文档频率低于5的词
    max_df=0.8,                  # 忽略文档频率高于80%的词（过于常见）
    sublinear_tf=True            # 使用亚线性TF缩放
)

X_tfidf = tfidf.fit_transform(df['cut_review'])
y = df['label'].values

print(f"✅ TF-IDF特征矩阵形状：{X_tfidf.shape}")
print(f"   特征词数量：{len(tfidf.get_feature_names_out())}")
print(f"\n前10个特征词：{tfidf.get_feature_names_out()[:10].tolist()}")

# ========== TF-IDF向量化 ==========
print("\n正在提取TF-IDF特征...")

tfidf = TfidfVectorizer(
    max_features=5000,           # 保留最重要的5000个特征词
    ngram_range=(1, 2),          # 使用一元词和二元词组合
    min_df=5,                    # 忽略文档频率低于5的词
    max_df=0.8,                  # 忽略文档频率高于80%的词（过于常见）
    sublinear_tf=True            # 使用亚线性TF缩放
)

X_tfidf = tfidf.fit_transform(df['cut_review'])
y = df['label'].values

print(f"✅ TF-IDF特征矩阵形状：{X_tfidf.shape}")
print(f"   特征词数量：{len(tfidf.get_feature_names_out())}")
print(f"\n前10个特征词：{tfidf.get_feature_names_out()[:10].tolist()}")

from sklearn.preprocessing import StandardScaler

print("正在对TF-IDF特征进行标准化...")

scaler = StandardScaler(with_mean=False)  # 👈 关键参数
X_scaled = scaler.fit_transform(X_tfidf)

print(f"特征缩放完成, 形状: {X_scaled.shape}")

# 查看第1列统计量（稀疏矩阵要先转稠密）
col0 = X_scaled[:, 0].toarray().ravel()
print(f" 缩放后均值(第1列): {col0.mean():.4f}")
print(f" 缩放后方差(第1列): {col0.std():.4f}")
print(f" 缩放后最小值: {col0.min():.4f}")
print(f" 缩放后最大值: {col0.max():.4f}")

# ========== 特征选择：卡方检验 ==========
print("\n正在使用卡方检验进行特征选择...")

# 选择 Top 2000 个特征
k = 2000
selector = SelectKBest(chi2, k=k)
X_selected = selector.fit_transform(X_tfidf, y)

# 获取选中的特征索引
selected_indices = selector.get_support(indices=True)
selected_features = [tfidf.get_feature_names_out()[i] for i in selected_indices]

print(f"✅ 特征选择完成：{X_tfidf.shape[1]} -> {X_selected.shape[1]} 个特征")
print(f"\nTop 20 最相关特征词：")
for i, (feature, score) in enumerate(zip(selected_features[:20], selector.scores_[selected_indices[:20]])):
    print(f"   {i+1:2d}. {feature} (得分：{score:.2f})")

    # ========== PCA降维 ==========
print("\n正在执行PCA降维...")

# 使用稀疏矩阵直接做PCA需要先转置，这里使用TruncatedSVD（适合稀疏矩阵）
from sklearn.decomposition import TruncatedSVD

# 尝试降到 100 维
n_components = 100
svd = TruncatedSVD(n_components=n_components, random_state=42)
X_pca = svd.fit_transform(X_tfidf)

# 计算累计解释方差
explained_variance_ratio = svd.explained_variance_ratio_
cumsum_variance = explained_variance_ratio.cumsum()

print(f"✅ PCA降维完成：{X_tfidf.shape[1]} -> {n_components} 维")
print(f"   前 {n_components} 维累计解释方差：{cumsum_variance[-1]:.2%}")
print(f"\n各维度解释方差（前10维）：")
for i, var in enumerate(explained_variance_ratio[:10]):
    print(f"   第 {i+1} 维：{var:.4%}")
print(f"   累计方差：{cumsum_variance[9]:.2%}")

# 评估不同维度下的解释方差
print("\n不同维度数下的累计解释方差：")
for dim in [50, 100, 200, 500]:
    svd_temp = TruncatedSVD(n_components=dim, random_state=42)
    svd_temp.fit(X_tfidf)
    print(f"   {dim} 维：{svd_temp.explained_variance_ratio_.sum():.2%}")

    # ========== PCA降维 ==========
print("\n正在执行PCA降维...")

# 使用稀疏矩阵直接做PCA需要先转置，这里使用TruncatedSVD（适合稀疏矩阵）
from sklearn.decomposition import TruncatedSVD

# 尝试降到 100 维
n_components = 100
svd = TruncatedSVD(n_components=n_components, random_state=42)
X_pca = svd.fit_transform(X_tfidf)

# 计算累计解释方差
explained_variance_ratio = svd.explained_variance_ratio_
cumsum_variance = explained_variance_ratio.cumsum()

print(f"✅ PCA降维完成：{X_tfidf.shape[1]} -> {n_components} 维")
print(f"   前 {n_components} 维累计解释方差：{cumsum_variance[-1]:.2%}")
print(f"\n各维度解释方差（前10维）：")
for i, var in enumerate(explained_variance_ratio[:10]):
    print(f"   第 {i+1} 维：{var:.4%}")
print(f"   累计方差：{cumsum_variance[9]:.2%}")

# 评估不同维度下的解释方差
print("\n不同维度数下的累计解释方差：")
for dim in [50, 100, 200, 500]:
    svd_temp = TruncatedSVD(n_components=dim, random_state=42)
    svd_temp.fit(X_tfidf)
    print(f"   {dim} 维：{svd_temp.explained_variance_ratio_.sum():.2%}")

    import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, roc_curve, confusion_matrix, classification_report)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

import joblib

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

print("✅ 所有库导入成功！")

# ========== 加载特征数据 ==========
X_train = joblib.load('X_train.pkl')
X_test = joblib.load('X_test.pkl')
y_train = joblib.load('y_train.pkl')
y_test = joblib.load('y_test.pkl')

# 加载向量化器和特征选择器（用于获取特征名称）
tfidf = joblib.load('tfidf_vectorizer.pkl')
selector = joblib.load('feature_selector.pkl')

# 获取被选中的特征名称
all_feature_names = tfidf.get_feature_names_out()
selected_indices = selector.get_support(indices=True)
feature_names = [all_feature_names[i] for i in selected_indices]

print(f"✅ 数据加载完成：")
print(f"   训练集：{X_train.shape[0]} 条，特征数：{X_train.shape[1]}")
print(f"   测试集：{X_test.shape[0]} 条，特征数：{X_test.shape[1]}")
print(f"\n训练集标签分布：")
print(pd.Series(y_train).value_counts().sort_index())
print(f"\n测试集标签分布：")
print(pd.Series(y_test).value_counts().sort_index())
print(f"\n特征数量：{len(feature_names)}")

def evaluate_model(model, X_train, y_train, X_test, y_test, model_name):
    """
    对模型进行训练、预测和评估，输出性能指标及可视化
    """
    # 训练
    model.fit(X_train, y_train)
    
    # 预测
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
    
    # 计算指标
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob) if y_prob is not None else None
    
    # 打印结果
    print(f"\n{'='*60}")
    print(f"{model_name} 性能评估")
    print(f"{'='*60}")
    print(f"准确率  (Accuracy)  : {acc:.4f}")
    print(f"精确率  (Precision) : {prec:.4f}")
    print(f"召回率  (Recall)    : {rec:.4f}")
    print(f"F1分数  (F1-Score)  : {f1:.4f}")
    if auc is not None:
        print(f"ROC-AUC             : {auc:.4f}")
    print(f"\n分类报告：")
    print(classification_report(y_test, y_pred, target_names=['负面', '正面']))
    
    # 混淆矩阵
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['负面', '正面'], yticklabels=['负面', '正面'])
    plt.title(f'{model_name} 混淆矩阵')
    plt.xlabel('预测值')
    plt.ylabel('真实值')
    plt.tight_layout()
    plt.show()
    
    # ROC曲线
    if y_prob is not None:
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        plt.figure(figsize=(6, 5))
        plt.plot(fpr, tpr, label=f'ROC (AUC = {auc:.4f})')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('假阳性率 (FPR)')
        plt.ylabel('真阳性率 (TPR)')
        plt.title(f'{model_name} ROC曲线')
        plt.legend()
        plt.tight_layout()
        plt.show()
    
    # 返回指标字典
    metrics = {'Accuracy': acc, 'Precision': prec, 'Recall': rec, 'F1': f1}
    if auc is not None:
        metrics['AUC'] = auc
    return metrics

results = {}

print("\n" + "="*60)
print("步骤2：基线模型 - 逻辑回归（默认参数）")
print("="*60)

lr_base = LogisticRegression(max_iter=1000, random_state=42)
results['逻辑回归 (基线)'] = evaluate_model(
    lr_base, X_train, y_train, X_test, y_test, "逻辑回归（默认）"
)

from sklearn.feature_selection import SelectKBest, f_classif

# 重新创建一个特征选择器（假设是 SelectKBest，k=2000）
selector_new = SelectKBest(score_func=f_classif, k=2000)

# 在训练集上拟合 + 转换
X_train_selected = selector_new.fit_transform(X_train, y_train)
X_test_selected = selector_new.transform(X_test)

# 可选：保存新的 selector 供以后使用
joblib.dump(selector_new, 'feature_selector_new.pkl')

# 重新创建特征选择器（推荐）
from sklearn.feature_selection import SelectKBest, f_classif

selector_new = SelectKBest(score_func=f_classif, k=2000)
X_train_selected = selector_new.fit_transform(X_train, y_train)
X_test_selected = selector_new.transform(X_test)

# 打印确认
print(f"降维前训练集特征数: {X_train.shape[1]}")
print(f"降维后训练集特征数: {X_train_selected.shape[1]}")
print(f"降维后测试集特征数: {X_test_selected.shape[1]}")

# 现在可以安全调用 evaluate_model
evaluate_model(lr_base, X_train_selected, y_train, X_test_selected, y_test, "逻辑回归(降维后)")

# ============================================================
# 步骤3: 逻辑回归超参数调优(网格搜索)
# ============================================================
print("\n" + "="*60)
print("步骤3: 逻辑回归超参数调优(网格搜索)")
print("="*60)

# 定义参数网格
param_grid_lr = {
    'C': [0.01, 0.1, 1, 10, 100],
    'penalty': ['l2'],
    'solver': ['liblinear']
}

# 初始化模型
lr = LogisticRegression(max_iter=1000, random_state=42)

# 5折交叉验证
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# 网格搜索
grid_lr = GridSearchCV(
    estimator=lr,
    param_grid=param_grid_lr,
    cv=cv,
    scoring='accuracy',
    n_jobs=-1,
    verbose=1
)

# 在降维后的数据上拟合
grid_lr.fit(X_train_selected, y_train)

# 输出最佳参数和交叉验证准确率
print(f"\n✅ 最佳参数: {grid_lr.best_params_}")
print(f"✅ 最佳交叉验证准确率: {grid_lr.best_score_:.4f}")

# 使用最优模型进行评估
best_lr = grid_lr.best_estimator_
results['逻辑回归(调优)'] = evaluate_model(
    best_lr,
    X_train_selected,
    y_train,
    X_test_selected,
    y_test,
    "逻辑回归(调优后)"
)

# ============================================================
# 步骤4: 随机森林超参数调优(网格搜索)
# ============================================================
print("\n" + "="*60)
print("步骤4: 随机森林超参数调优(网格搜索)")
print("="*60)

# 定义参数网格
param_grid_rf = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5, 10]
}

# 初始化随机森林模型
rf = RandomForestClassifier(random_state=42)

# 5折交叉验证
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# 网格搜索
grid_rf = GridSearchCV(
    estimator=rf,
    param_grid=param_grid_rf,
    cv=cv,
    scoring='accuracy',
    n_jobs=-1,
    verbose=1
)

# ✅ 在降维后的数据上拟合
grid_rf.fit(X_train_selected, y_train)

# 输出最佳参数和交叉验证准确率
print(f"\n✅ 最佳参数: {grid_rf.best_params_}")
print(f"✅ 最佳交叉验证准确率: {grid_rf.best_score_:.4f}")

# ✅ 使用降维后的数据进行评估
best_rf = grid_rf.best_estimator_
results['随机森林(调优)'] = evaluate_model(
    best_rf,
    X_train_selected,
    y_train,
    X_test_selected,
    y_test,
    "随机森林(调优后)"
)

# ============================================================
# 绘制随机森林特征重要性 (Top 20)
# ============================================================
importances = best_rf.feature_importances_
indices = np.argsort(importances)[::-1][:20]

plt.figure(figsize=(10, 6))
plt.barh(range(20), importances[indices], color='teal')
plt.yticks(range(20), [feature_names[i] for i in indices])
plt.xlabel('特征重要性')
plt.title('随机森林 Top 20 重要特征')
plt.tight_layout()
plt.show()

# ============================================================
# 步骤5: XGBoost + 网格搜索调优
# ============================================================
print("\n" + "="*60)
print("步骤5: XGBoost 超参数调优(网格搜索)")
print("="*60)

# 定义参数网格
param_grid_xgb = {
    'n_estimators': [50, 100, 150],
    'max_depth': [3, 6, 9],
    'learning_rate': [0.05, 0.1, 0.2],
    'subsample': [0.8, 1.0]
}

# 初始化 XGBoost 模型
xgb = XGBClassifier(
    random_state=42,
    eval_metric='logloss',
    use_label_encoder=False
)

# 5折交叉验证
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# 网格搜索
grid_xgb = GridSearchCV(
    estimator=xgb,
    param_grid=param_grid_xgb,
    cv=cv,
    scoring='accuracy',
    n_jobs=-1,
    verbose=1
)

# ✅ 在降维后的数据上拟合
grid_xgb.fit(X_train_selected, y_train)

# 输出最佳参数和交叉验证准确率
print(f"\n✅ 最佳参数: {grid_xgb.best_params_}")
print(f"✅ 最佳交叉验证准确率: {grid_xgb.best_score_:.4f}")

# ✅ 使用降维后的数据进行评估
best_xgb = grid_xgb.best_estimator_
results['XGBoost(调优)'] = evaluate_model(
    best_xgb,
    X_train_selected,
    y_train,
    X_test_selected,
    y_test,
    "XGBoost(调优后)"
)

# ============================================================
# 绘制 XGBoost 特征重要性 (Top 20)
# ============================================================
xgb_importances = best_xgb.feature_importances_
indices_xgb = np.argsort(xgb_importances)[::-1][:20]

plt.figure(figsize=(10, 6))
plt.barh(range(20), xgb_importances[indices_xgb][::-1], color='orange')
plt.yticks(range(20), [feature_names[i] for i in indices_xgb[::-1]])
plt.xlabel('特征重要性')
plt.title('XGBoost Top 20 重要特征')
plt.tight_layout()
plt.show()

print("results 中的模型数量:", len(results))
print("模型名称:", list(results.keys()))

# ============================================================
# 步骤6: 模型性能对比汇总
# ============================================================
print("\n" + "="*60)
print("各模型性能对比汇总")
print("="*60)

# 汇总结果
summary_df = pd.DataFrame(results).T
summary_df.columns = ['Accuracy', 'Precision', 'Recall', 'F1', 'AUC']
print(summary_df)

# 保存汇总结果
summary_df.to_csv("model_comparison.csv", encoding='utf-8-sig')
print("\n✅ 对比结果已保存为 model_comparison.csv")

# ========== 可视化对比: 折线图 ==========
import matplotlib.pyplot as plt
import numpy as np

metrics_to_plot = ['Accuracy', 'Precision', 'Recall', 'F1', 'AUC']
x = np.arange(len(summary_df.index))
width = 0.15

fig, ax = plt.subplots(figsize=(10, 6))
for i, metric in enumerate(metrics_to_plot):
    ax.plot(summary_df.index, summary_df[metric], marker='o', label=metric, linewidth=2)

ax.set_xlabel('模型')
ax.set_ylabel('分数')
ax.set_title('各模型性能对比')
ax.legend(loc='lower right')
ax.grid(True, axis='y', linestyle='--', alpha=0.7)
plt.xticks(rotation=15)
plt.tight_layout()
plt.show()

# ========== 可视化对比: 柱状图 ==========
summary_df[metrics_to_plot].plot(
    kind='bar', figsize=(10, 6), colormap='viridis'
)
plt.title('各模型性能指标对比')
plt.ylabel('分数')
plt.ylim(0.7, 1.0)
plt.xticks(rotation=15)
plt.legend(loc='lower right')
plt.grid(True, axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# ============================================================
# 步骤7: 保存最佳模型
# ============================================================
print("\n" + "="*60)
print("步骤7: 保存最佳模型")
print("="*60)

# 找到F1分数最高的模型名称
best_model_name = summary_df['F1'].idxmax()
print(f"\n🏆 最佳模型: {best_model_name}，F1 = {summary_df.loc[best_model_name, 'F1']:.4f}")

# 根据模型名称选择对应的模型对象
best_model = None
if '逻辑回归' in best_model_name:
    best_model = best_lr
elif '随机森林' in best_model_name:
    best_model = best_rf
elif 'XGBoost' in best_model_name:
    best_model = best_xgb

# 保存最佳模型
if best_model is not None:
    joblib.dump(best_model, 'best_model.pkl')
    print(f"✅ 最佳模型已保存为 best_model.pkl")
    print(f"   模型: {best_model_name}")
    print(f"   F1分数: {summary_df.loc[best_model_name, 'F1']:.4f}")
    print(f"   保存路径: ./best_model.pkl")
else:
    print("⚠️ 未找到对应的最佳模型对象，请检查模型名称匹配！")

print("\n" + "="*60)
print("实验二、三全部完成！")
print("="*60)

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt
import seaborn as sns
import time

# 机器学习模型
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# 超参数调优
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold
from scipy.stats import uniform, randint

# 模型集成
from sklearn.ensemble import VotingClassifier, StackingClassifier

# 评估指标
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, roc_auc_score, confusion_matrix,
                             classification_report, roc_curve)

import joblib

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

print("✅ 所有库导入成功！")

# ========== 加载特征数据（2000维特征，与实验三一致） ==========
X_train = joblib.load('X_train.pkl')
X_test = joblib.load('X_test.pkl')
y_train = joblib.load('y_train.pkl')
y_test = joblib.load('y_test.pkl')

print(f"✅ 数据加载完成：")
print(f"   训练集：{X_train.shape[0]} 条，特征数：{X_train.shape[1]}")
print(f"   测试集：{X_test.shape[0]} 条，特征数：{X_test.shape[1]}")
print(f"\n训练集标签分布：正面 {np.sum(y_train==1)}，负面 {np.sum(y_train==0)}")
print(f"测试集标签分布：正面 {np.sum(y_test==1)}，负面 {np.sum(y_test==0)}")

# 加载特征名称（用于特征重要性可视化）
try:
    selector = joblib.load('feature_selector.pkl')
    tfidf = joblib.load('tfidf_vectorizer.pkl')
    all_feature_names = tfidf.get_feature_names_out()
    selected_indices = selector.get_support(indices=True)
    feature_names = [all_feature_names[i] for i in selected_indices]
    print(f"   特征名称数量：{len(feature_names)}")
except:
    feature_names = [f"feature_{i}" for i in range(X_train.shape[1])]
    print("⚠️ 未找到特征名称文件，使用默认索引名称")

# ========== 评估函数 ==========
def evaluate_model(model, X_train, y_train, X_test, y_test, model_name, detailed=True):
    """统一模型评估接口"""
    # 计时
    start_time = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start_time
    
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob) if y_prob is not None else None
    
    if detailed:
        print(f"\n{'='*60}")
        print(f"{model_name}")
        print(f"{'='*60}")
        print(f"训练时间：{train_time:.2f} 秒")
        print(f"准确率  (Accuracy)  : {acc:.4f}")
        print(f"精确率  (Precision) : {prec:.4f}")
        print(f"召回率  (Recall)    : {rec:.4f}")
        print(f"F1分数  (F1-Score)  : {f1:.4f}")
        if auc is not None:
            print(f"ROC-AUC             : {auc:.4f}")
    
    metrics = {'Accuracy': acc, 'Precision': prec, 'Recall': rec, 'F1': f1, 'Train_Time': train_time}
    if auc is not None:
        metrics['AUC'] = auc
    return metrics, model, y_pred, y_prob

# ========== 交叉验证策略 ==========
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# 存储所有结果
all_results = {}
all_models = {}

print("\n" + "="*60)
print("步骤2：加载实验三最佳模型（基线）")
print("="*60)

# 实验三最佳模型：逻辑回归（调优后）参数 C=10
baseline_model = LogisticRegression(
    C=10, penalty='l2', solver='liblinear',
    max_iter=1000, random_state=42
)

metrics_base, model_base, y_pred_base, y_prob_base = evaluate_model(
    baseline_model, X_train, y_train, X_test, y_test, 
    "基线模型：逻辑回归 (C=10)"
)
all_results['基线模型 (逻辑回归)'] = metrics_base
all_models['基线模型 (逻辑回归)'] = model_base

print(f"\n✅ 基线模型 F1 = {metrics_base['F1']:.4f}")

print("\n" + "="*60)
print("步骤3.1：网格搜索（Grid Search）")
print("="*60)

param_grid_fine = {
    'C': [5, 8, 10, 12, 15],
    'penalty': ['l2'],
    'solver': ['liblinear']
}

grid_start = time.time()
grid_search = GridSearchCV(
    LogisticRegression(max_iter=1000, random_state=42),
    param_grid_fine, cv=cv, scoring='accuracy', n_jobs=-1, verbose=1
)
grid_search.fit(X_train, y_train)
grid_time = time.time() - grid_start

print(f"\n✅ 网格搜索完成，耗时：{grid_time:.2f} 秒")
print(f"✅ 最佳参数：{grid_search.best_params_}")
print(f"✅ 最佳交叉验证准确率：{grid_search.best_score_:.4f}")

metrics_grid, model_grid, _, _ = evaluate_model(
    grid_search.best_estimator_, X_train, y_train, X_test, y_test,
    "网格搜索 (Grid Search)"
)
all_results['网格搜索 (Grid)'] = metrics_grid
all_models['网格搜索 (Grid)'] = model_grid

print("\n" + "="*60)
print("步骤3.2：随机搜索（Randomized Search）")
print("="*60)

param_dist = {
    'C': uniform(0.5, 20),
    'penalty': ['l2'],
    'solver': ['liblinear']
}

random_start = time.time()
random_search = RandomizedSearchCV(
    LogisticRegression(max_iter=1000, random_state=42),
    param_dist, n_iter=30, cv=cv, scoring='accuracy',
    random_state=42, n_jobs=-1, verbose=1
)
random_search.fit(X_train, y_train)
random_time = time.time() - random_start

print(f"\n✅ 随机搜索完成，耗时：{random_time:.2f} 秒")
print(f"✅ 最佳参数：{random_search.best_params_}")
print(f"✅ 最佳交叉验证准确率：{random_search.best_score_:.4f}")

metrics_random, model_random, _, _ = evaluate_model(
    random_search.best_estimator_, X_train, y_train, X_test, y_test,
    "随机搜索 (Randomized Search)"
)
all_results['随机搜索 (Random)'] = metrics_random
all_models['随机搜索 (Random)'] = model_random

print("\n" + "="*60)
print("步骤3.4：两种调优方法对比")
print("="*60)

tuning_compare = pd.DataFrame({
    '方法': ['网格搜索', '随机搜索'],
    'F1分数': [
        all_results.get('网格搜索 (Grid)', {}).get('F1', 0),
        all_results.get('随机搜索 (Random)', {}).get('F1', 0)
    ],
    '训练时间(秒)': [
        all_results.get('网格搜索 (Grid)', {}).get('Train_Time', 0),
        all_results.get('随机搜索 (Random)', {}).get('Train_Time', 0)
       
    ]
})
print(tuning_compare.to_string(index=False))

print("\n" + "="*60)
print("步骤4：投票集成（Voting Classifier）")
print("="*60)

# 获取三个基模型（使用实验三调优后的参数）
lr_model = LogisticRegression(C=10, max_iter=1000, random_state=42)
rf_model = RandomForestClassifier(n_estimators=300, max_depth=None, 
                                  min_samples_split=10, random_state=42)
xgb_model = XGBClassifier(n_estimators=50, max_depth=3, learning_rate=0.05,
                          subsample=0.8, random_state=42, 
                          use_label_encoder=False, eval_metric='logloss')

print("训练基模型...")
lr_model.fit(X_train, y_train)
rf_model.fit(X_train, y_train)
xgb_model.fit(X_train, y_train)

estimators = [
    ('lr', lr_model),
    ('rf', rf_model),
    ('xgb', xgb_model)
]

print("\n--- 硬投票（Hard Voting）---")
voting_hard = VotingClassifier(estimators=estimators, voting='hard')
metrics_hard, model_hard, _, _ = evaluate_model(
    voting_hard, X_train, y_train, X_test, y_test,
    "投票集成 - 硬投票 (Hard Voting)"
)
all_results['投票集成 (硬投票)'] = metrics_hard
all_models['投票集成 (硬投票)'] = model_hard

print("\n--- 软投票（Soft Voting）---")
voting_soft = VotingClassifier(estimators=estimators, voting='soft')
metrics_soft, model_soft, _, _ = evaluate_model(
    voting_soft, X_train, y_train, X_test, y_test,
    "投票集成 - 软投票 (Soft Voting)"
)
all_results['投票集成 (软投票)'] = metrics_soft
all_models['投票集成 (软投票)'] = model_soft

print("\n--- 加权投票（Weighted Voting）---")
# 根据各模型在测试集上的F1分数设置权重
y_pred_lr = lr_model.predict(X_test)
y_pred_rf = rf_model.predict(X_test)
y_pred_xgb = xgb_model.predict(X_test)

f1_lr = f1_score(y_test, y_pred_lr)
f1_rf = f1_score(y_test, y_pred_rf)
f1_xgb = f1_score(y_test, y_pred_xgb)

weights = np.array([f1_lr, f1_rf, f1_xgb])
weights = weights / weights.sum()  # 归一化

print(f"各模型F1分数：逻辑回归={f1_lr:.4f}, 随机森林={f1_rf:.4f}, XGBoost={f1_xgb:.4f}")
print(f"归一化权重：逻辑回归={weights[0]:.3f}, 随机森林={weights[1]:.3f}, XGBoost={weights[2]:.3f}")

voting_weighted = VotingClassifier(estimators=estimators, voting='soft', weights=weights)
metrics_weighted, model_weighted, _, _ = evaluate_model(
    voting_weighted, X_train, y_train, X_test, y_test,
    "投票集成 - 加权投票"
)
all_results['投票集成 (加权投票)'] = metrics_weighted
all_models['投票集成 (加权投票)'] = model_weighted

print("\n" + "="*60)
print("步骤5.1：堆叠集成 - 元学习器：逻辑回归")
print("="*60)

stacking_lr = StackingClassifier(
    estimators=estimators,
    final_estimator=LogisticRegression(C=10, max_iter=1000, random_state=42),
    cv=5
)
metrics_stack_lr, model_stack_lr, _, _ = evaluate_model(
    stacking_lr, X_train, y_train, X_test, y_test,
    "堆叠集成 - 元学习器：逻辑回归"
)
all_results['堆叠集成 (元学习器:LR)'] = metrics_stack_lr
all_models['堆叠集成 (元学习器:LR)'] = model_stack_lr

print("\n" + "="*60)
print("步骤6：所有模型性能对比汇总")
print("="*60)

summary_df = pd.DataFrame(all_results).T.round(4)
print(summary_df)

# 按F1排序
summary_df_sorted = summary_df.sort_values('F1', ascending=False)
print(f"\n按F1分数排序：")
print(summary_df_sorted[['F1', 'Accuracy', 'Precision', 'Recall']])

# 保存结果
summary_df.to_csv('model_ensemble_comparison.csv')
print("\n✅ 对比结果已保存为 model_ensemble_comparison.csv")

# ========== 可视化对比 ==========
metrics_to_plot = ['Accuracy', 'Precision', 'Recall', 'F1', 'AUC']

plt.figure(figsize=(14, 8))
summary_df[metrics_to_plot].plot(kind='bar', figsize=(14, 8), colormap='viridis')
plt.title('各模型性能指标对比（超参数调优 + 模型集成）', fontsize=16)
plt.ylabel('分数', fontsize=12)
plt.ylim(0.7, 1.0)
plt.xticks(rotation=30, ha='right')
plt.legend(loc='lower right')
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.show()

# ======== 训练时间对比（单位：秒）========
if 'Train_Time' in summary_df.columns:
    plt.figure(figsize=(12, 6))

    # 按训练时间升序排列
    time_sorted = summary_df['Train_Time'].sort_values()

    ax = time_sorted.plot(
        kind='bar',
        color='steelblue',
        edgecolor='white',
        legend=False
    )

    plt.title('各模型训练时间对比（单位：秒）', fontsize=16)
    plt.ylabel('训练时间（秒）', fontsize=12)
    plt.xlabel('模型', fontsize=12)
    plt.xticks(rotation=30, ha='right', fontsize=10)

    # 在柱顶标注具体秒数
    for i, v in enumerate(time_sorted.values):
        ax.text(
            i,
            v + 0.5,               # 稍微高于柱子
            f'{v:.2f}s',
            ha='center',
            va='bottom',
            fontsize=9
        )

    # Y 轴从 0 开始，上限略高于最大值
    ax.set_ylim(0, time_sorted.max() * 1.15)
    ax.yaxis.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.show()

print("\n" + "="*60)
print("步骤7：确定并保存最佳模型")
print("="*60)

# 按F1分数找最佳模型
best_model_name = summary_df['F1'].idxmax()
best_model = all_models.get(best_model_name)

print(f"\n✅ 最佳模型：{best_model_name}")
print(f"   F1分数：{summary_df.loc[best_model_name, 'F1']:.4f}")
print(f"   准确率：{summary_df.loc[best_model_name, 'Accuracy']:.4f}")

if best_model is not None:
    joblib.dump(best_model, 'best_ensemble_model.pkl')
    print(f"✅ 最佳集成模型已保存为 best_ensemble_model.pkl")

print("\n" + "="*60)
print("步骤8：最佳模型详细评估")
print("="*60)

if best_model is not None:
    # 训练并预测
    best_model.fit(X_train, y_train)
    y_pred = best_model.predict(X_test)
    y_prob = best_model.predict_proba(X_test)[:, 1] if hasattr(best_model, "predict_proba") else None
    
    # 混淆矩阵
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['负面', '正面'], yticklabels=['负面', '正面'])
    plt.title(f'{best_model_name} 混淆矩阵', fontsize=14)
    plt.xlabel('预测值')
    plt.ylabel('真实值')
    plt.tight_layout()
    plt.show()
    
    # ROC曲线
    if y_prob is not None:
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        auc = roc_auc_score(y_test, y_prob)
        plt.figure(figsize=(6, 5))
        plt.plot(fpr, tpr, label=f'ROC (AUC = {auc:.4f})')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('假阳性率 (FPR)')
        plt.ylabel('真阳性率 (TPR)')
        plt.title(f'{best_model_name} ROC曲线', fontsize=14)
        plt.legend()
        plt.tight_layout()
        plt.show()
    
    # 分类报告
    print("\n详细分类报告：")
    print(classification_report(y_test, y_pred, target_names=['负面', '正面']))

print("\n" + "="*60)
print("✅ 实验四全部完成！")
print("="*60)

import pandas as pd
import numpy as np
import jieba
import warnings
warnings.filterwarnings('ignore')

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Embedding, Bidirectional, LSTM,
    Dense, Dropout
)
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, f1_score
)

import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

df = pd.read_csv(
    r'D:\Anaconda\Jingdong_Data_4000\cleaned_reviews.csv'
)

print(df.shape)
print(df.columns)

def chinese_cut(text):
    return ' '.join(jieba.cut(str(text)))

if 'cut_review' not in df.columns:
    print("⚠️ 未检测到 cut_review，正在生成分词结果...")
    df['cut_review'] = df['review'].apply(chinese_cut)

# 保存，避免下次再跑
df.to_csv(
    r'D:\Anaconda\Jingdong_Data_4000\cleaned_reviews_with_cut.csv',
    index=False,
    encoding='utf-8-sig'
)

print("✅ cut_review 生成完成并已保存")

MAX_WORDS = 5000      # 同实验二 max_features
MAX_LEN = 100         # 同实验一统计结论

tokenizer = Tokenizer(num_words=MAX_WORDS, lower=True, filters='')
tokenizer.fit_on_texts(df['cut_review'])

X_seq = tokenizer.texts_to_sequences(df['cut_review'])
X_pad = pad_sequences(X_seq, maxlen=MAX_LEN)

y = df['label'].values

print("序列化完成，形状：", X_pad.shape)

X_train, X_test, y_train, y_test = train_test_split(
    X_pad, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("训练集:", X_train.shape, "测试集:", X_test.shape)

model = Sequential([
    Embedding(
        input_dim=MAX_WORDS,
        output_dim=128,
        input_length=MAX_LEN
    ),
    Bidirectional(LSTM(64, return_sequences=False)),
    Dropout(0.5),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(1, activation='sigmoid')
])

model.compile(
    loss='binary_crossentropy',
    optimizer=Adam(learning_rate=1e-3),
    metrics=['accuracy']
)

model.summary()

early_stop = EarlyStopping(
    monitor='val_loss',
    patience=3,
    restore_best_weights=True
)

history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=20,
    batch_size=32,
    callbacks=[early_stop],
    verbose=1
)

y_pred_prob = model.predict(X_test)
y_pred = (y_pred_prob >= 0.5).astype(int)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("F1-score:", f1_score(y_test, y_pred))
print(classification_report(y_test, y_pred, target_names=['负面', '正面']))

plt.figure(figsize=(12,4))

plt.subplot(1,2,1)
plt.plot(history.history['loss'], label='训练损失')
plt.plot(history.history['val_loss'], label='验证损失')
plt.legend()
plt.title('Loss 曲线')

plt.subplot(1,2,2)
plt.plot(history.history['accuracy'], label='训练准确率')
plt.plot(history.history['val_accuracy'], label='验证准确率')
plt.legend()
plt.title('Accuracy 曲线')

plt.tight_layout()
plt.show()

import joblib

# 保存训练好的 BiLSTM 模型
model.save('best_bilstm_model.h5')

# 保存 Tokenizer，以便后续对新文本进行相同的序列化处理
with open('tokenizer.pkl', 'wb') as f:
    joblib.dump(tokenizer, f)

print("✅ BiLSTM 模型与 Tokenizer 已保存成功。")

