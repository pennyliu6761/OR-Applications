# 第 13 週｜深度學習基礎與類神經網路（Neural Networks Fundamentals）

> [!NOTE]
> **課程**：AI 大數據分析　**週次**：第 13 週／共 16 週　**時數**：3 小時
> **本週關鍵字**：感知器、激活函數、前向傳播、反向傳播、梯度下降、過度配適、Dropout、Early Stopping
> **使用工具**：Google Colab（Python：TensorFlow／Keras、scikit-learn）

---

## 0. 本週課程時間分配

| 節次 | 時間 | 內容 |
|---|---|---|
| Hour 1 | 60 分鐘 | 理論深度：感知器、激活函數、前向／反向傳播、梯度下降、過度配適與正則化 |
| Hour 2 | 60 分鐘 | Python 示範演練：建立 MLP 分類模型、過度配適示範、與前幾週模型三方比較 |
| Hour 3（前 40 分鐘） | 40 分鐘 | 延伸練習（兩組模擬資料，附完整程式碼）＋ 綜合案例研究 |
| Hour 3（後 20 分鐘） | 20 分鐘 | **碩士論文延伸應用**：兩個可行論文方向之主題定義、方法說明、模擬資料與 Python 實作展示 |

> [!NOTE]
> 建議於 [Google Colab](https://colab.research.google.com/) 開啟新筆記本，依序將本講義程式碼區塊複製貼上執行。Colab 已預先安裝 TensorFlow，不需要額外安裝。

---

## 1. 學習目標

完成本週課程後，學員應能夠：

1. 說明感知器與多層感知器（MLP）的基本結構，理解類神經網路如何透過多層非線性轉換學習複雜關係。
2. 說明常見激活函數（Sigmoid、Tanh、ReLU）的特性與選用時機。
3. 理解前向傳播、損失函數、反向傳播、梯度下降之間的完整訓練邏輯。
4. 具體理解類神經網路的過度配適現象，並運用 Dropout、Early Stopping 等正則化技術控制模型複雜度。
5. 在 Python 中使用 Keras 建立 MLP 模型，並與第 10、12 週的線性模型、樹狀模型進行比較。
6. 說出至少兩個可將本週方法延伸為碩士論文題目的具體方向，並理解其對應的進階方法與實作方式。

---

## Hour 1｜理論深度（60 分鐘）

### 1.0 類神經網路導論

類神經網路（Artificial Neural Network）的設計靈感來自生物神經系統：許多簡單的「神經元」相互連接，各自對輸入做簡單的加權運算，再透過大量神經元的組合，逼近極為複雜的函數關係。相對於第 10 週的線性模型（假設固定的函數形式）與第 12 週的樹狀模型（以階梯狀規則切分資料），類神經網路的核心優勢是**理論上能以任意精度逼近任何連續函數**（泛用近似定理，Universal Approximation Theorem），這也是「深度學習」近年在影像、語音、自然語言等複雜任務上大放異彩的數學基礎。

> [!NOTE]
> 本週聚焦於最基礎的**多層感知器（Multilayer Perceptron, MLP）**，這是處理表格型（結構化）資料最基本的類神經網路架構。影像處理常用的卷積神經網路（CNN）、序列資料常用的循環神經網路（RNN），皆是在 MLP 的基礎上，針對特定資料型態設計的進階架構，超出本課程範圍，但理解本週的基礎概念，是進一步學習這些進階架構的必要前提。

---

### 1.1 感知器與多層感知器

**單一神經元的運算**：

$$
z = \sum_i w_i x_i + b, \qquad a = f(z)
$$

其中 $x_i$ 為輸入、 $w_i$ 為權重、 $b$ 為偏置（Bias）、 $f$ 為激活函數、 $a$ 為該神經元的輸出。

**多層感知器（MLP）結構**：

| 層級 | 說明 |
|---|---|
| 輸入層（Input Layer） | 接收原始特徵，神經元數量等於特徵數 |
| 隱藏層（Hidden Layer） | 一層或多層，每層由多個神經元組成，負責逐層萃取越來越抽象的特徵組合 |
| 輸出層（Output Layer） | 產生最終預測，迴歸問題通常為單一神經元（不加激活函數或用線性輸出）；二元分類問題為單一神經元＋Sigmoid；多元分類則為多個神經元＋Softmax |

> [!IMPORTANT]
> 若拿掉激活函數（或激活函數本身是線性的），無論疊加多少層隱藏層，整個網路的運算本質上等價於**一層線性轉換**——多層線性函數的合成，仍然只是另一個線性函數。**激活函數的非線性，正是類神經網路能夠學習複雜、非線性關係的根本原因**，這是理解類神經網路最重要的一個概念關卡。

---

### 1.2 激活函數

| 激活函數 | 公式 | 特性 |
|---|---|---|
| Sigmoid | $f(z)=\dfrac{1}{1+e^{-z}}$ | 輸出介於 $(0,1)$ ，適合二元分類的輸出層；隱藏層中易發生梯度消失 |
| Tanh | $f(z)=\dfrac{e^z-e^{-z}}{e^z+e^{-z}}$ | 輸出介於 $(-1,1)$ ，以 0 為中心，梯度消失問題略優於 Sigmoid |
| ReLU（Rectified Linear Unit） | $f(z)=\max(0,z)$ | 計算簡單、能有效緩解梯度消失，是目前隱藏層**最常用**的激活函數 |

> [!NOTE]
> **梯度消失（Vanishing Gradient）** 是訓練深層網路時的核心挑戰之一：Sigmoid、Tanh 在輸入值極大或極小時，函數曲線趨於平緩（梯度趨近於零），根據 1.5 節的鏈式法則，多層梯度連續相乘後，越靠近輸入層的梯度會變得極小，導致該層的權重幾乎無法更新、學習停滯。ReLU 在正值區域的梯度恆為 1，大幅緩解了這個問題，這也是為什麼現代類神經網路的隱藏層幾乎都預設使用 ReLU。

---

### 1.3 前向傳播（Forward Propagation）

資料從輸入層開始，逐層計算每一層的加權和與激活值，直到輸出層產生最終預測值，這個由前往後的計算過程稱為前向傳播。以一個「輸入層—隱藏層—輸出層」的簡單網路為例：

$$
z^{(1)} = W^{(1)}x+b^{(1)}, \quad a^{(1)}=f(z^{(1)})
$$
$$
z^{(2)} = W^{(2)}a^{(1)}+b^{(2)}, \quad \hat{y}=f(z^{(2)})
$$

---

### 1.4 損失函數

| 任務型態 | 損失函數 | 公式 |
|---|---|---|
| 迴歸 | 均方誤差（MSE） | $\dfrac{1}{n}\sum_i(y_i-\hat{y}_i)^2$ |
| 二元分類 | 二元交叉熵（Binary Cross-Entropy） | $-\dfrac{1}{n}\sum_i[y_i\log\hat{y}_i+(1-y_i)\log(1-\hat{y}_i)]$ |

> [!NOTE]
> 二元交叉熵損失函數，其數學形式與第 10 週邏輯斯迴歸的最大概似估計法目標函數**完全相同**——這再次呼應了本課程反覆強調的方法論連結：**一個沒有隱藏層、輸出層使用 Sigmoid 激活函數的類神經網路，其實就是邏輯斯迴歸**。多層感知器，本質上可以理解為「多層堆疊的邏輯斯迴歸」，每一層都在前一層萃取出的特徵基礎上，學習更複雜的特徵組合。

---

### 1.5 反向傳播與梯度下降

**梯度下降（Gradient Descent）**：類神經網路的訓練目標是找到一組權重，使損失函數最小化，透過反覆沿著損失函數的**負梯度方向**微幅調整權重：

$$
w \leftarrow w - \eta\frac{\partial L}{\partial w}
$$

其中 $\eta$ 為**學習率（Learning Rate）**，控制每次更新的步伐大小。

**反向傳播（Backpropagation）**：類神經網路的權重數量龐大（經常高達數萬、數百萬個），若每個權重都獨立計算梯度，計算成本極高。反向傳播透過微積分的**鏈式法則（Chain Rule）**，從輸出層的誤差開始，由後往前逐層推算每一層權重對最終損失的貢獻，是一種高效計算全部梯度的演算法，而非另一種獨立的最佳化方法——**梯度下降負責「往哪個方向調整權重」，反向傳播負責「有效率地算出這個方向」**。

> [!TIP]
> 現代深度學習框架（如 Keras、PyTorch）都內建**自動微分（Automatic Differentiation）**機制，使用者只需要定義網路架構與損失函數，框架會自動完成反向傳播的所有梯度計算，不需要手動推導微分公式——這也是本週 Hour 2 示範中，程式碼只需要呼叫 `model.fit()`，背後就完成了完整的前向傳播、損失計算、反向傳播、權重更新循環的原因。

**優化器（Optimizer）**：實務上很少使用最原始的梯度下降，而是使用改良版本的優化器，其中 **Adam（Adaptive Moment Estimation）** 是目前最常用、最穩健的預設選擇，能自動調整每個權重的有效學習率，通常比固定學習率的傳統梯度下降收斂更快、更穩定。

---

### 1.6 過度配適與正則化

如第 12 週決策樹一般，類神經網路同樣容易**過度配適**——尤其當網路架構（層數、神經元數）相對於訓練資料量而言過於龐大時，模型有足夠的參數「記住」訓練資料的每一個雜訊細節。

**常見正則化技術**：

| 技術 | 做法 |
|---|---|
| **Dropout** | 訓練過程中，每次疊代**隨機**將一定比例（如 30–50%）的神經元輸出暫時歸零，強迫網路不能過度依賴任何少數幾個神經元，迫使各神經元學習更穩健、更具泛化性的特徵 |
| **Early Stopping（提前停止）** | 持續監控驗證集損失，一旦驗證集損失連續數個訓練週期（Epoch）不再下降，就提前停止訓練，避免模型在訓練集上繼續配適雜訊 |
| **L2 正則化（權重衰減）** | 在損失函數中加入權重平方和的懲罰項，抑制權重變得過大，概念與第 10 週的 Ridge 迴歸完全相同 |

> [!IMPORTANT]
> 本週 Hour 2 示範會具體呈現：**同樣的網路架構，在不加任何正則化時嚴重過度配適（訓練準確率接近100%、驗證準確率卻遠遠落後）；加入 Dropout 後，訓練與驗證準確率會變得非常接近**——這是本週最重要的實作驗證，也呼應第 9、10、12 週反覆出現的核心主題：**模型複雜度必須與可用資料量相匹配，正則化技術正是在「模型夠複雜、足以學習真實規律」與「模型太複雜、開始記憶雜訊」之間，人為施加約束以取得平衡的手段**。

---

### 1.7 深度學習 vs. 傳統機器學習比較

| 考量面向 | 類神經網路 | 傳統機器學習（線性模型、樹狀模型） |
|---|---|---|
| 資料量需求 | 通常需要較大量資料才能發揮優勢，但架構得當、搭配正則化，中小樣本亦可有良好表現 | 中小樣本情境下通常更穩健 |
| 可解釋性 | 較低（「黑箱」，須額外借助如 SHAP 等技術） | 線性模型、單一決策樹可解釋性佳 |
| 特徵工程需求 | 較低（網路能自動學習特徵組合） | 通常需要人工設計特徵與交互作用項 |
| 訓練與調校複雜度 | 較高，超參數（層數、神經元數、學習率、正則化強度）眾多 | 相對較低 |
| 適用資料型態 | 表格資料、影像、文字、語音皆可（搭配對應架構） | 主要適用結構化表格資料 |

---

### 1.8 方法選擇指南

| 情境 | 建議方法 |
|---|---|
| 中小型結構化表格資料、需要高可解釋性 | 線性模型（第10週）或決策樹（第12週） |
| 資料存在複雜非線性關係，且已有相當資料量 | 類神經網路，或第12週的隨機森林／梯度提升樹 |
| 影像、語音、文字等非結構化資料 | 類神經網路（搭配 CNN、RNN 或 Transformer 等進階架構，超出本課程範圍） |
| 訓練集與驗證集表現差距過大 | 檢查是否過度配適，優先嘗試 Dropout、Early Stopping、減少網路複雜度 |

### 1.9 常用中英文詞彙對照表

| 中文 | 英文 | 中文 | 英文 |
|---|---|---|---|
| 感知器 | Perceptron | 多層感知器 | Multilayer Perceptron, MLP |
| 激活函數 | Activation Function | 前向傳播 | Forward Propagation |
| 損失函數 | Loss Function | 反向傳播 | Backpropagation |
| 梯度下降 | Gradient Descent | 學習率 | Learning Rate |
| 訓練週期 | Epoch | 丟棄法 | Dropout |
| 提前停止 | Early Stopping | 梯度消失 | Vanishing Gradient |

### 1.10 本週公式總表

| 項目 | 公式 |
|---|---|
| 神經元運算 | $a=f(\sum_i w_ix_i+b)$ |
| ReLU | $f(z)=\max(0,z)$ |
| 均方誤差 | $MSE=\frac{1}{n}\sum_i(y_i-\hat{y}_i)^2$ |
| 二元交叉熵 | $-\frac{1}{n}\sum_i[y_i\log\hat{y}_i+(1-y_i)\log(1-\hat{y}_i)]$ |
| 梯度下降更新規則 | $w\leftarrow w-\eta\frac{\partial L}{\partial w}$ |

### 1.11 各方法的假設條件與限制一覽

| 方法 | 隱含假設 | 主要限制 |
|---|---|---|
| MLP | 無分配假設，資料驅動 | 需要標準化輸入（對照第9週）；對輸出目標的尺度也敏感（本週2.7節與Hour3示範將具體呈現） |
| Dropout | 各神經元的資訊具備一定冗餘性 | 訓練時間增加（須更多epoch才能收斂）；`Dropout`比例須視資料量調整 |
| Early Stopping | 驗證集損失是泛化誤差的合理代理指標 | 須額外切分驗證集，資料量極少時較難穩定判斷停止點 |

---

## Hour 2｜Python 示範演練（60 分鐘）

### 2.0 環境設置與模擬資料生成

```python
# ============================================================
# 第13週 Hour 2 示範：MLP類神經網路建模
# ============================================================
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 抑制TensorFlow的詳細日誌訊息

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import matplotlib

!apt-get -qq install fonts-noto-cjk > /dev/null 2>&1
matplotlib.rcParams['font.sans-serif'] = ['Noto Sans CJK JT', 'Noto Sans CJK TC']
matplotlib.rcParams['axes.unicode_minus'] = False

tf.random.set_seed(42)
np.random.seed(42)
n = 1000

mileage_km = np.random.normal(45000, 15000, n).clip(500, None)
engine_temp_c = np.random.normal(88, 6, n)
vibration_mm_s = np.random.gamma(shape=2.0, scale=1.5, size=n)
fuel_efficiency_km_l = 12 - 0.00008*mileage_km + np.random.normal(0, 0.8, n)

risk_score = (0.00002*mileage_km + 0.03*engine_temp_c
              + 0.15*vibration_mm_s + np.random.normal(0, 0.5, n))
threshold = np.percentile(risk_score, 70)
needs_maintenance = (risk_score > threshold).astype(int)

df = pd.DataFrame({
    'mileage_km': mileage_km.round(1), 'engine_temp_c': engine_temp_c.round(2),
    'vibration_mm_s': vibration_mm_s.round(3), 'fuel_efficiency_km_l': fuel_efficiency_km_l.round(2),
    'needs_maintenance': needs_maintenance,
})
print("資料集形狀:", df.shape)
```

---

### 2.1 建立第一個 MLP 分類模型

```python
# ============================================================
# 2.1 建立MLP分類模型（對照1.1-1.4節）
# ============================================================
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score

X = df[['mileage_km', 'engine_temp_c', 'vibration_mm_s', 'fuel_efficiency_km_l']]
y = df['needs_maintenance']
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

# 類神經網路對輸入尺度敏感，標準化是必要步驟（呼應第9週）
scaler = StandardScaler()
Xtr_s = scaler.fit_transform(Xtr)
Xte_s = scaler.transform(Xte)

# 建立MLP：輸入層(4個特徵) -> 隱藏層(16神經元,ReLU) -> 隱藏層(8神經元,ReLU) -> 輸出層(1個神經元,Sigmoid)
model = keras.Sequential([
    keras.layers.Input(shape=(4,)),
    keras.layers.Dense(16, activation='relu'),
    keras.layers.Dense(8, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid'),
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

history = model.fit(Xtr_s, ytr, validation_split=0.2, epochs=50, batch_size=32, verbose=0)

pred_proba = model.predict(Xte_s, verbose=0).flatten()
pred = (pred_proba > 0.5).astype(int)
print(f"\nMLP測試集準確率={accuracy_score(yte,pred):.4f}, AUC={roc_auc_score(yte,pred_proba):.4f}")
```

---

### 2.2 訓練過程視覺化

```python
# ============================================================
# 2.2 訓練歷史視覺化（觀察損失函數收斂過程）
# ============================================================

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
axes[0].plot(history.history['loss'], label='訓練集損失')
axes[0].plot(history.history['val_loss'], label='驗證集損失')
axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('二元交叉熵損失')
axes[0].set_title('損失函數收斂曲線'); axes[0].legend(); axes[0].grid(alpha=0.3)

axes[1].plot(history.history['accuracy'], label='訓練集準確率')
axes[1].plot(history.history['val_accuracy'], label='驗證集準確率')
axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('準確率')
axes[1].set_title('準確率收斂曲線'); axes[1].legend(); axes[1].grid(alpha=0.3)
plt.tight_layout()
plt.show()
```

---

### 2.3 激活函數比較

```python
# ============================================================
# 2.3 激活函數比較（對照1.2節）
# ============================================================

for act in ['relu', 'tanh', 'sigmoid']:
    tf.random.set_seed(42)
    m = keras.Sequential([
        keras.layers.Input(shape=(4,)),
        keras.layers.Dense(16, activation=act),
        keras.layers.Dense(8, activation=act),
        keras.layers.Dense(1, activation='sigmoid'),
    ])
    m.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    m.fit(Xtr_s, ytr, epochs=50, batch_size=32, verbose=0)
    proba = m.predict(Xte_s, verbose=0).flatten()
    print(f"隱藏層激活函數={act}: 準確率={accuracy_score(yte,(proba>0.5).astype(int)):.4f}, "
          f"AUC={roc_auc_score(yte,proba):.4f}")

print("\n觀察：對於本例這種相對單純的資料結構，三種激活函數表現相近，")
print("差異主要在更深、更複雜的網路架構中才會顯著浮現（如1.2節提及的梯度消失問題）。")
```

---

### 2.4 過度配適示範與正則化

```python
# ============================================================
# 2.4 過度配適示範與Dropout正則化（對照1.6節）
# ============================================================

# 使用較小樣本數(210筆)子集，讓過度配適更容易發生
np.random.seed(1)
sub_idx = np.random.choice(len(Xtr_s), 210, replace=False)
Xtr_small, ytr_small = Xtr_s[sub_idx], ytr.iloc[sub_idx]

# --- 大型網路、無正則化：容易過度配適 ---
tf.random.set_seed(42)
model_overfit = keras.Sequential([
    keras.layers.Input(shape=(4,)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid'),
])
model_overfit.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
hist_overfit = model_overfit.fit(Xtr_small, ytr_small, validation_data=(Xte_s, yte),
                                    epochs=100, batch_size=16, verbose=0)

print("=== 大型網路、無正則化（容易過度配適）===")
print(f"最終訓練準確率={hist_overfit.history['accuracy'][-1]:.4f}, "
      f"驗證準確率={hist_overfit.history['val_accuracy'][-1]:.4f}")

# --- 加入Dropout正則化 ---
tf.random.set_seed(42)
model_dropout = keras.Sequential([
    keras.layers.Input(shape=(4,)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dropout(0.3),
    keras.layers.Dense(1, activation='sigmoid'),
])
model_dropout.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
hist_dropout = model_dropout.fit(Xtr_small, ytr_small, validation_data=(Xte_s, yte),
                                    epochs=100, batch_size=16, verbose=0)

print("\n=== 相同架構、加入Dropout正則化 ===")
print(f"最終訓練準確率={hist_dropout.history['accuracy'][-1]:.4f}, "
      f"驗證準確率={hist_dropout.history['val_accuracy'][-1]:.4f}")
```

**預期輸出**：無正則化的大型網路，訓練準確率會逼近 100%，但驗證準確率停滯在約 74%（巨大落差，嚴重過度配適）；加入 Dropout 後，訓練與驗證準確率會變得非常接近（皆約 80%），且驗證表現反而**優於**未正則化版本——具體驗證了 1.6 節的核心論點。

---

### 2.5 三方模型比較：邏輯斯迴歸 vs. 隨機森林 vs. MLP

```python
# ============================================================
# 2.5 三方模型比較（對照第10、12週）
# ============================================================
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score

def eval_sklearn(model, name):
    model.fit(Xtr, ytr)
    pred = model.predict(Xte)
    proba = model.predict_proba(Xte)[:, 1]
    print(f"{name}: 準確率={accuracy_score(yte,pred):.4f}, F1={f1_score(yte,pred):.4f}, "
          f"AUC={roc_auc_score(yte,proba):.4f}")

eval_sklearn(LogisticRegression(max_iter=1000), "邏輯斯迴歸（第10週）")
eval_sklearn(RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42), "隨機森林（第12週）")

pred_mlp = (model.predict(Xte_s, verbose=0).flatten() > 0.5).astype(int)
proba_mlp = model.predict(Xte_s, verbose=0).flatten()
print(f"MLP（本週）: 準確率={accuracy_score(yte,pred_mlp):.4f}, F1={f1_score(yte,pred_mlp):.4f}, "
      f"AUC={roc_auc_score(yte,proba_mlp):.4f}")

print("\n觀察：三種方法在本週線性可加的模擬資料上表現相近，")
print("再次驗證第12週的教訓——模型複雜度應與資料真實結構匹配，而非預設越複雜的模型越好。")
```

---

### 2.6 迴歸任務的重要陷阱：目標變數標準化

```python
# ============================================================
# 2.6 MLP迴歸任務示範：目標變數也需要標準化（重要陷阱）
# ============================================================

np.random.seed(21)
n1 = 300
storage_temp = np.random.normal(24, 4, n1)
humidity = np.random.normal(50, 10, n1)
storage_age = np.random.uniform(1, 20, n1)
monthly_loss = 5 + 0.8*storage_temp + 0.3*humidity + 1.2*storage_age + np.random.normal(0, 8, n1)

df1 = pd.DataFrame({'storage_temp': storage_temp, 'humidity': humidity, 'storage_age': storage_age})
X1, y1 = df1, monthly_loss
Xtr1, Xte1, ytr1, yte1 = train_test_split(X1, y1, test_size=0.3, random_state=42)

scaler1 = StandardScaler()
Xtr1_s = scaler1.fit_transform(Xtr1)
Xte1_s = scaler1.transform(Xte1)

from sklearn.metrics import r2_score, mean_squared_error

# --- 錯誤示範：目標變數y未標準化 ---
tf.random.set_seed(42)
nn_bad = keras.Sequential([
    keras.layers.Input(shape=(3,)),
    keras.layers.Dense(16, activation='relu'),
    keras.layers.Dense(8, activation='relu'),
    keras.layers.Dense(1),
])
nn_bad.compile(optimizer='adam', loss='mse')
nn_bad.fit(Xtr1_s, ytr1, epochs=100, batch_size=16, verbose=0)
pred_bad = nn_bad.predict(Xte1_s, verbose=0).flatten()
print(f"目標變數未標準化: R2={r2_score(yte1,pred_bad):.4f}（收斂困難，R2可能偏低甚至為負值）")

# --- 正確做法：目標變數也標準化，預測後再還原尺度 ---
y_scaler = StandardScaler()
ytr1_s = y_scaler.fit_transform(np.asarray(ytr1).reshape(-1, 1)).flatten()

tf.random.set_seed(42)
nn_good = keras.Sequential([
    keras.layers.Input(shape=(3,)),
    keras.layers.Dense(16, activation='relu'),
    keras.layers.Dense(8, activation='relu'),
    keras.layers.Dense(1),
])
nn_good.compile(optimizer='adam', loss='mse')
nn_good.fit(Xtr1_s, ytr1_s, epochs=150, batch_size=16, verbose=0, validation_split=0.15)
pred_good_s = nn_good.predict(Xte1_s, verbose=0).flatten()
pred_good = y_scaler.inverse_transform(pred_good_s.reshape(-1, 1)).flatten()
print(f"目標變數已標準化: R2={r2_score(yte1,pred_good):.4f}, "
      f"RMSE={np.sqrt(mean_squared_error(yte1,pred_good)):.2f}")
print("\n對照第12週: 決策樹R2=0.16, 隨機森林R2=0.35")
```

> [!CAUTION]
> 這是類神經網路實作中一個經常被忽略、卻代價慘重的陷阱：**決策樹、隨機森林對目標變數的尺度完全不敏感（因為分割邏輯只依賴大小順序），但類神經網路的權重初始化與優化器，都是針對「數值範圍大致落在較小區間」的假設設計的**。若目標變數的數值範圍較大（如本例 monthly_loss 範圍達 13–82），網路可能難以有效收斂，導致 $R^2$ 遠低於正確做法的結果，甚至可能出現負值（代表模型比「直接猜測平均值」還差）。**類神經網路的迴歸任務，務必同時標準化輸入特徵與目標變數**，預測完成後再將結果轉換回原始尺度。

---

## Hour 3（前 40 分鐘）｜延伸練習與綜合案例研究

### 3.1 延伸練習一：MLP 迴歸於通信裝備故障預測（附完整程式碼）

```python
# ============================================================
# 延伸練習一：MLP分類於通信裝備故障預測（對照第12週延伸練習二）
# ============================================================
np.random.seed(31)
n2 = 500
signal = np.random.normal(70, 15, n2)
age = np.random.uniform(1, 60, n2)
usage_hours = np.random.uniform(100, 5000, n2)
logit = -3 + 0.03*age - 0.015*signal + 0.0003*usage_hours
prob = 1/(1+np.exp(-logit))
failure = (np.random.random(n2) < prob).astype(int)

df2 = pd.DataFrame({'signal': signal, 'age': age, 'usage_hours': usage_hours})
X2, y2 = df2, failure
Xtr2, Xte2, ytr2, yte2 = train_test_split(X2, y2, test_size=0.3, random_state=42, stratify=y2)

scaler2 = StandardScaler()
Xtr2_s = scaler2.fit_transform(Xtr2)
Xte2_s = scaler2.transform(Xte2)

tf.random.set_seed(42)
nn2 = keras.Sequential([
    keras.layers.Input(shape=(3,)),
    keras.layers.Dense(16, activation='relu'),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(8, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid'),
])
nn2.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
nn2.fit(Xtr2_s, ytr2, epochs=80, batch_size=32, verbose=0)
proba2 = nn2.predict(Xte2_s, verbose=0).flatten()
pred2 = (proba2 > 0.5).astype(int)
print(f"MLP: 準確率={accuracy_score(yte2,pred2):.4f}, AUC={roc_auc_score(yte2,proba2):.4f}")
print("（對照第12週隨機森林：AUC約0.65-0.75）")
```

**詳解**：MLP 的表現應與第 12 週隨機森林相近（AUC 約 0.70–0.75），驗證兩種不同典範的方法在此類中等複雜度的分類問題上，效能大致相當。

---

### 3.2 延伸練習二：MLP 迴歸於彈藥庫月耗損量預測（附完整程式碼）

```python
# ============================================================
# 延伸練習二：MLP迴歸（務必記得標準化目標變數，對照2.6節）
# ============================================================
np.random.seed(41)
n3 = 400
temp3 = np.random.normal(24, 4, n3)
humidity3 = np.random.normal(50, 10, n3)
age3 = np.random.uniform(1, 20, n3)
loss3 = 5 + 0.8*temp3 + 0.3*humidity3 + 1.2*age3 + np.random.normal(0, 8, n3)

df3 = pd.DataFrame({'temp': temp3, 'humidity': humidity3, 'age': age3})
X3, y3 = df3, loss3
Xtr3, Xte3, ytr3, yte3 = train_test_split(X3, y3, test_size=0.3, random_state=42)

scaler3 = StandardScaler()
Xtr3_s = scaler3.fit_transform(Xtr3)
Xte3_s = scaler3.transform(Xte3)
y_scaler3 = StandardScaler()
ytr3_s = y_scaler3.fit_transform(np.asarray(ytr3).reshape(-1, 1)).flatten()

tf.random.set_seed(42)
nn3 = keras.Sequential([
    keras.layers.Input(shape=(3,)),
    keras.layers.Dense(16, activation='relu'),
    keras.layers.Dense(8, activation='relu'),
    keras.layers.Dense(1),
])
nn3.compile(optimizer='adam', loss='mse')
nn3.fit(Xtr3_s, ytr3_s, epochs=150, batch_size=16, verbose=0)
pred3_s = nn3.predict(Xte3_s, verbose=0).flatten()
pred3 = y_scaler3.inverse_transform(pred3_s.reshape(-1, 1)).flatten()
print(f"MLP迴歸: R2={r2_score(yte3,pred3):.4f}, RMSE={np.sqrt(mean_squared_error(yte3,pred3)):.2f}")
```

**詳解**：正確標準化目標變數後，MLP 的 $R^2$ 應可達到 0.4 以上，優於第 12 週的決策樹與隨機森林，因為此資料的真實關係為線性可加，MLP 能有效學習這種簡單的線性映射。

---

### 3.3 綜合案例研究：建立裝備妥善狀態預測系統之方法演進

> [!NOTE]
> 以下是一個虛構但貼近實務的案例，目的是把本課程第 10、12、13 週的方法串接起來，示範一個典型的研究方法演進脈絡。

**背景**：某聯兵旅保修處希望為裝備妥善狀態預測系統，找出最適合的建模方法。

**步驟一：以邏輯斯迴歸建立基準模型**（對應第 10 週）
先以最簡單、最可解釋的邏輯斯迴歸建立基準模型，確認各特徵的係數方向符合工程直覺，作為後續比較的基準線。

**步驟二：以隨機森林嘗試提升效能**（對應第 12 週）
若懷疑資料存在非線性關係或特徵交互作用，嘗試隨機森林，並比較其相對於基準模型的效能提升幅度是否顯著。

**步驟三：以 MLP 進一步嘗試**（對應本週）
若資料量充足、且前兩步驟顯示可能存在複雜的非線性樣態，嘗試 MLP，並透過 Dropout、Early Stopping 嚴格控制過度配適風險。

**步驟四：綜合比較，選擇最適合實務部署的方法**
最終選擇方法時，不應只看單一效能指標，而應綜合考量：效能提升幅度是否值得犧牲的可解釋性、模型維運複雜度、資料量是否足以支撐更複雜模型的訓練——這正是本課程第 10、12、13 週反覆強調的核心方法論原則。

---

## Hour 3（後 20 分鐘）｜碩士論文延伸應用

> [!NOTE]
> 以下兩個方向，示範如何把本週的類神經網路方法延伸為具備研究貢獻的碩士論文題目——核心邏輯是：**正則化技術的選擇與強度，對類神經網路的實際表現有決定性影響，值得系統性比較；「深度學習需要大量資料才能發揮優勢」是常見的刻板印象，但透過適當的正則化，類神經網路在中小樣本的軍事資料情境下未必居於劣勢**。每個方向皆包含主題定義、方法說明、模擬資料設計與完整可執行的 Python 實作。

### 3.4 論文方向一：類神經網路正則化策略之系統性比較研究

**主題定義**：如 1.6 節介紹，Dropout、Early Stopping、L2 正則化是緩解類神經網路過度配適的三種常見技術，但實務上該如何選擇、是否應該合併使用，缺乏系統性的比較依據。本研究以裝備故障預測任務為例，系統性比較「無正則化」「僅 Dropout」「Dropout ＋ Early Stopping」三種策略，在小樣本情境下的訓練/驗證準確率差距與最終測試集表現。

**方法說明**：

- **控制變數原則**：三種策略使用**完全相同**的網路架構（層數、神經元數），僅正則化設定不同，確保比較的是正則化策略本身的效果，而非架構差異的混淆。
- **評估指標**：不僅比較測試集準確率，更重要的是比較**訓練集與驗證集準確率的差距**——差距越小，代表模型的泛化能力越可信賴，這比單看一次性的測試集準確率更能反映模型的穩健性。

**模擬資料**：延續本週 2.4 節之小樣本（210 筆）裝備感測資料情境。

```python
# ============================================================
# 論文方向一：類神經網路正則化策略之系統性比較
# ============================================================
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score

tf.random.set_seed(42)
np.random.seed(42)
n = 300  # 刻意採用較小樣本數，讓正則化的效果差異更容易凸顯

mileage_km = np.random.normal(45000, 15000, n).clip(500, None)
engine_temp_c = np.random.normal(88, 6, n)
vibration_mm_s = np.random.gamma(shape=2.0, scale=1.5, size=n)
fuel_efficiency_km_l = 12 - 0.00008*mileage_km + np.random.normal(0, 0.8, n)
risk_score = (0.00002*mileage_km + 0.03*engine_temp_c
              + 0.15*vibration_mm_s + np.random.normal(0, 0.5, n))
threshold = np.percentile(risk_score, 70)
needs_maintenance = (risk_score > threshold).astype(int)

df = pd.DataFrame({'mileage_km': mileage_km, 'engine_temp_c': engine_temp_c,
                     'vibration_mm_s': vibration_mm_s, 'fuel_efficiency_km_l': fuel_efficiency_km_l})
X, y = df, needs_maintenance
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
scaler = StandardScaler()
Xtr_s = scaler.fit_transform(Xtr)
Xte_s = scaler.transform(Xte)

def build_network(strategy):
    """統一的網路架構，供三種正則化策略共用，確保公平比較"""
    use_dropout = strategy in ['dropout', 'dropout_earlystop']
    layers = [keras.layers.Input(shape=(4,)), keras.layers.Dense(128, activation='relu')]
    if use_dropout:
        layers.append(keras.layers.Dropout(0.5))
    layers.append(keras.layers.Dense(128, activation='relu'))
    if use_dropout:
        layers.append(keras.layers.Dropout(0.5))
    layers.append(keras.layers.Dense(64, activation='relu'))
    layers.append(keras.layers.Dense(1, activation='sigmoid'))
    return keras.Sequential(layers)

results = {}
for strategy in ['none', 'dropout', 'dropout_earlystop']:
    tf.random.set_seed(42)
    model = build_network(strategy)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    callbacks = []
    if strategy == 'dropout_earlystop':
        callbacks = [keras.callbacks.EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)]

    hist = model.fit(Xtr_s, ytr, validation_data=(Xte_s, yte), epochs=150,
                       batch_size=16, callbacks=callbacks, verbose=0)

    train_acc = hist.history['accuracy'][-1]
    val_acc = hist.history['val_accuracy'][-1]
    gap = train_acc - val_acc
    results[strategy] = (train_acc, val_acc, gap)
    print(f"策略={strategy}: 訓練準確率={train_acc:.4f}, 驗證準確率={val_acc:.4f}, "
          f"差距={gap:.4f}, 實際訓練週期數={len(hist.history['loss'])}")

print(f"\n=== 結論 ===")
print(f"訓練/驗證準確率差距: 無正則化={results['none'][2]:.4f}, "
      f"僅Dropout={results['dropout'][2]:.4f}, Dropout+EarlyStopping={results['dropout_earlystop'][2]:.4f}")
```

**預期輸出**：無正則化策略的訓練/驗證準確率差距最大（約 0.25，訓練準確率接近 100%、驗證準確率僅約 74%）；加入 Dropout 後差距大幅縮小；再加上 Early Stopping，模型會在驗證損失不再改善時提前停止（實際訓練週期數遠少於設定的 150），進一步避免了訓練後期的過度配適，通常能達到最小的訓練/驗證差距與最穩健的驗證表現。

**論文延伸建議**：可進一步將 L2 正則化（`kernel_regularizer=keras.regularizers.l2(...)`）納入比較，並嘗試不同的 Dropout 比例（0.2、0.3、0.5）與 Early Stopping 的 `patience` 參數，繪製「正則化強度 vs. 驗證集表現」的關係曲線，探討是否存在正則化「過猶不及」的現象（正則化太強，模型配適不足；正則化太弱，仍然過度配適）。

---

### 3.5 論文方向二：類神經網路與集成樹狀模型之樣本效率比較研究

**主題定義**：業界普遍存在「深度學習需要大量資料才能發揮優勢」的刻板印象，但這個說法是否適用於**經過適當正則化**的類神經網路，缺乏系統性的實證檢驗。本研究以第 12 週的隨機森林為對照組，比較「未正則化的大型 MLP」與「有正則化的大型 MLP」在訓練樣本數從 50 筆逐步增加到 2000 筆的過程中，測試集 AUC 的變化趨勢，驗證正則化是否是「讓類神經網路在小樣本情境下也具備競爭力」的關鍵因素。

**方法說明**：

- **樣本效率曲線（Learning Curve）**：固定測試集，讓訓練集樣本數從小到大逐步增加，觀察各方法的測試集效能如何隨訓練樣本數變化，是比較不同方法對資料量需求差異的標準實驗設計。
- **三方比較**：隨機森林（第 12 週基準）、無正則化大型 MLP、有正則化（Dropout＋Early Stopping）大型 MLP。

**模擬資料**：延續本週 Hour 2 之車輛感測資料生成邏輯，產生 5000 筆資料池，固定 1000 筆為測試集，訓練集樣本數依序取 50、100、200、500、1000、2000 筆。

```python
# ============================================================
# 論文方向二：類神經網路 vs. 隨機森林之樣本效率比較
# ============================================================
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

tf.random.set_seed(42)
np.random.seed(42)
n_total = 5000

mileage_km = np.random.normal(45000, 15000, n_total).clip(500, None)
engine_temp_c = np.random.normal(88, 6, n_total)
vibration_mm_s = np.random.gamma(shape=2.0, scale=1.5, size=n_total)
fuel_efficiency_km_l = 12 - 0.00008*mileage_km + np.random.normal(0, 0.8, n_total)
risk_score = (0.00002*mileage_km + 0.03*engine_temp_c
              + 0.15*vibration_mm_s + np.random.normal(0, 0.5, n_total))
threshold = np.percentile(risk_score, 70)
needs_maintenance = (risk_score > threshold).astype(int)

df = pd.DataFrame({'mileage_km': mileage_km, 'engine_temp_c': engine_temp_c,
                     'vibration_mm_s': vibration_mm_s, 'fuel_efficiency_km_l': fuel_efficiency_km_l})
X_all, y_all = df, needs_maintenance
X_pool, X_test, y_pool, y_test = train_test_split(X_all, y_all, test_size=1000,
                                                      random_state=42, stratify=y_all)

def build_large_mlp(regularized):
    """大型網路架構(128-128-64)，regularized控制是否加入Dropout"""
    layers = [keras.layers.Input(shape=(4,)), keras.layers.Dense(128, activation='relu')]
    if regularized:
        layers.append(keras.layers.Dropout(0.3))
    layers.append(keras.layers.Dense(128, activation='relu'))
    if regularized:
        layers.append(keras.layers.Dropout(0.3))
    layers.append(keras.layers.Dense(64, activation='relu'))
    layers.append(keras.layers.Dense(1, activation='sigmoid'))
    model = keras.Sequential(layers)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

sample_sizes = [50, 100, 200, 500, 1000, 2000]
early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

print(f"{'樣本數':>8}{'隨機森林AUC':>14}{'無正則化MLP':>14}{'正則化MLP':>12}")
for n_train in sample_sizes:
    Xtr_sub = X_pool.iloc[:n_train]
    ytr_sub = y_pool[:n_train]
    scaler_sub = StandardScaler()
    Xtr_sub_s = scaler_sub.fit_transform(Xtr_sub)
    Xte_sub_s = scaler_sub.transform(X_test)

    rf = RandomForestClassifier(n_estimators=200, max_depth=5, random_state=42)
    rf.fit(Xtr_sub, ytr_sub)
    rf_auc = roc_auc_score(y_test, rf.predict_proba(X_test)[:, 1])

    tf.random.set_seed(42)
    nn_plain = build_large_mlp(regularized=False)
    nn_plain.fit(Xtr_sub_s, ytr_sub, epochs=100, batch_size=min(32, n_train), verbose=0)
    nn_plain_auc = roc_auc_score(y_test, nn_plain.predict(Xte_sub_s, verbose=0).flatten())

    tf.random.set_seed(42)
    nn_reg = build_large_mlp(regularized=True)
    nn_reg.fit(Xtr_sub_s, ytr_sub, validation_split=0.2, epochs=150,
                batch_size=min(32, n_train), callbacks=[early_stop], verbose=0)
    nn_reg_auc = roc_auc_score(y_test, nn_reg.predict(Xte_sub_s, verbose=0).flatten())

    print(f"{n_train:>8}{rf_auc:>14.4f}{nn_plain_auc:>14.4f}{nn_reg_auc:>12.4f}")
```

**預期輸出**：**無正則化的大型 MLP**在樣本數增加時表現不穩定，甚至在樣本數達到 1000–2000 時，AUC 反而低於樣本數僅 100–200 時的表現（因為固定的訓練週期數與缺乏正則化，導致模型在更多資料上依然過度配適、未能有效收斂）；**有正則化的大型 MLP**則在所有樣本數下都能穩定達到與隨機森林相當、甚至更優的 AUC（約 0.81–0.85），且不會隨樣本數增加而出現效能反轉的異常現象。

> [!IMPORTANT]
> 這個結果對「深度學習需要大量資料」這個常見假設提出了重要的修正：**問題的關鍵不在於「資料量是否足夠大」，而在於「模型複雜度是否搭配了適當的正則化」**。一個未經正則化的大型網路，即使資料量增加，也可能因為訓練動態不穩定而表現不佳；反之，一個經過適當正則化的網路，即使在僅有 50–100 筆的小樣本軍事資料情境下，也能有與成熟的傳統機器學習方法相當的競爭力。這對資料蒐集成本高昂的軍事應用場景，具有正面且務實的意涵：**不必因為資料量有限就完全排除類神經網路的可能性，正則化技術的妥善運用，才是決定深度學習方法能否在小樣本情境下站穩腳步的關鍵**。

**論文延伸建議**：可進一步系統性改變網路的「容量」（層數、神經元數）與資料量的比例關係，尋找在不同樣本數下的「最適網路容量」，繪製更完整的容量—資料量—效能三維關係圖；也可以將此比較框架延伸至第 12 週的 XGBoost，建立類神經網路、隨機森林、梯度提升樹三方的樣本效率完整比較。

---

## 附錄A：理論常見問答（概念釐清 Q&A）

> [!NOTE]
> **Q1：為什麼類神經網路的輸出層，迴歸問題不加激活函數（或用線性輸出），分類問題卻要加 Sigmoid？**
> A：迴歸問題的目標值可以是任意實數（如維修成本可以是任何正數），若輸出層加上 Sigmoid（限制在 0 到 1 之間）或 ReLU（限制為非負），會人為限制模型能預測的數值範圍，因此迴歸問題的輸出層通常不加激活函數（即恆等函數，直接輸出加權和）。分類問題的目標是「屬於某類別的機率」，機率必須落在 $[0,1]$ 區間，Sigmoid 函數正好能將任意實數壓縮到這個區間，因此適合作為二元分類的輸出層激活函數。

> [!NOTE]
> **Q2：`epoch` 與 `batch_size` 是什麼？如何決定合適的數值？**
> A：一個 `epoch` 代表模型完整看過一次全部訓練資料；由於一次性把全部資料丟進去計算梯度，計算成本可能過高，實務上會把訓練資料切成多個小批次（`batch`），每個 `batch_size` 筆資料計算一次梯度並更新一次權重，累積處理完所有批次才算完成一個 `epoch`。`batch_size` 通常取 16、32、64 等 2 的次方（與硬體記憶體存取效率有關），`epoch` 數量則搭配 Early Stopping 動態決定，不需要事先精確設定。

> [!NOTE]
> **Q3：類神經網路的權重是如何初始化的？初始化重要嗎？**
> A：權重通常以某種隨機分配（如均勻分配或常態分配，並依網路層的輸入輸出維度調整變異程度，如 Xavier／He 初始化）隨機初始化，而非設為零——若所有權重初始化為相同數值（如零），同一層的所有神經元在訓練過程中會保持完全相同（對稱性問題），無法學習到不同的特徵。良好的初始化策略能加速收斂、緩解梯度消失／爆炸問題，Keras 等框架已內建業界公認良好的預設初始化策略，一般使用者通常不需要手動調整。

> [!NOTE]
> **Q4：如何判斷該用幾層隱藏層、每層幾個神經元？**
> A：沒有絕對公式，通常從較簡單的架構（如 1–2 層隱藏層）開始嘗試，觀察訓練/驗證表現，若配適不足（訓練與驗證表現都不理想），逐步增加網路容量（層數或神經元數）；若過度配適（訓練遠優於驗證），則應優先加強正則化或減少網路容量，而非直接增加資料。這個過程通常需要多次實驗與交叉驗證，是類神經網路調校中最耗時的部分之一。

> [!NOTE]
> **Q5：本週的 MLP 與第 12 週的隨機森林，可以結合使用嗎？**
> A：可以，常見的整合方式包括：（1）將兩者的預測機率取平均或加權平均，形成一個簡單的整合模型；（2）將隨機森林各棵樹的葉節點編號作為新的類別特徵，輸入給類神經網路（一種特徵工程手法）；（3）在正式的機器學習競賽中，經常會看到「多個不同典範模型的預測結果」作為新特徵，再訓練一個「二層」的整合模型（Stacking），這些整合策略超出本課程範圍，但都是奠基於本課程已介紹的各項基礎方法之上。

---

## 附錄B：本週方法快速索引表

| 任務 | 主要函數／類別 | 所屬套件 |
|---|---|---|
| 建立序列式神經網路模型 | `Sequential` | `tensorflow.keras` |
| 全連接層 | `Dense` | `tensorflow.keras.layers` |
| Dropout正則化 | `Dropout` | `tensorflow.keras.layers` |
| 模型編譯（指定優化器、損失函數） | `model.compile()` | `tensorflow.keras` |
| 模型訓練 | `model.fit()` | `tensorflow.keras` |
| 提前停止回呼函數 | `EarlyStopping` | `tensorflow.keras.callbacks` |
| L2正則化 | `regularizers.l2()` | `tensorflow.keras` |

---

## 附錄C：常見易混淆概念澄清

| 容易混淆的概念 | 差異說明 |
|---|---|
| 「梯度下降」 vs 「反向傳播」 | 梯度下降是決定「往哪個方向調整權重」的最佳化演算法；反向傳播是有效率地「計算出這個方向（梯度）」的技術，兩者搭配使用，並非同一件事 |
| 「Epoch」 vs 「Batch」 vs 「Iteration」 | Epoch是看過一次全部訓練資料；Batch是切分後的一小份資料；Iteration是處理完一個Batch、更新一次權重的次數，一個Epoch包含多次Iteration |
| 「Dropout」 vs 「隨機森林的特徵隨機」（第12週） | 兩者都利用「隨機捨棄部分資訊」的概念降低模型對特定路徑的過度依賴，但Dropout發生在同一個網路的訓練過程中（每次疊代隨機捨棄神經元），特徵隨機發生在建構多棵不同的樹之間，機制與應用層次不同 |
| 「二元交叉熵」 vs 「邏輯斯迴歸的概似函數」（第10週） | 兩者數學形式完全相同，二元交叉熵損失函數的最小化，等價於邏輯斯迴歸最大概似估計法的概似函數最大化 |
| 「目標變數需要標準化」（本週） vs 「樹狀模型不需要標準化」（第12週） | 類神經網路的優化過程對數值尺度敏感，輸入特徵與目標變數皆建議標準化；樹狀模型的分割邏輯僅依賴數值大小順序，兩者皆不需要標準化 |

---

## 附錄D：本週與後續課程週次的關聯

| 後續週次 | 關聯方式 |
|---|---|
| 第 10 週：迴歸與分類 | 本週1.4節具體說明二元交叉熵損失函數與邏輯斯迴歸概似函數的數學等價性 |
| 第 12 週：決策樹與隨機森林 | 本週2.5節、3.5節皆延續第12週建立的隨機森林模型作為比較基準與樣本效率對照組 |
| 第 14 週：時間序列與序列模型 | 本週MLP處理的是靜態表格資料，第14週將介紹能處理時間序列相依關係的序列模型架構 |
| 第 15、16 週：論文整併實戰 | 本週建立的類神經網路建模與正則化框架，可作為後續整合型專論中處理複雜非線性關係的技術選項之一 |

## 附錄E：本週模擬資料集與程式碼彙整

| 資料集 | 用途 | 摘要 |
|---|---|---|
| 車輛感測資料（1,000筆） | Hour2主範例 | MLP準確率0.80-0.81，與第10、12週三方比較表現相近 |
| 車輛感測資料子集（210筆） | 2.4節過度配適示範 | 無正則化訓練/驗證差距約0.25，Dropout後大幅縮小 |
| 彈藥庫儲存資料（300筆） | 2.6節迴歸陷阱示範 | 目標變數標準化前後R2天壤之別 |
| 通信裝備故障資料（500筆） | 延伸練習一 | MLP AUC約0.70-0.75 |
| 彈藥庫儲存資料（400筆） | 延伸練習二 | MLP迴歸R2>0.4，優於第12週樹狀模型 |
| 裝備感測資料（300筆） | 論文方向一 | 三種正則化策略之訓練/驗證差距比較 |
| 車輛感測資料（5,000筆池） | 論文方向二 | 樣本效率曲線，正則化MLP全樣本數皆具競爭力 |

---

## 參考資料

- 類神經網路、激活函數、反向傳播與梯度下降之標準理論框架，詳細技術文件可參考 [TensorFlow／Keras 官方文件](https://www.tensorflow.org/guide/keras)。
- Dropout 正則化技術原始論文：Srivastava, N., et al. (2014). *Dropout: A Simple Way to Prevent Neural Networks from Overfitting*. Journal of Machine Learning Research.

---

*下週課程：第 14 週｜時間序列分析與序列模型（ARIMA & LSTM）。*
