import numpy as np
import torch
import torch.nn as nn
import dgl
import pickle
from sclmd import SCLMD
from sklearn.feature_extraction.text import TfidfVectorizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.decomposition import PCA
from sklearn.metrics import f1_score

epochs=300
lr = 1e-2
dimension = 128#初始向量维度
n_hid = 60
n_inp = 128
n_out = 64#表征维度
batch_size = 100
maxlen = 6000
tau = 0.5
lam = 0.5
k=32
open_r =9800
num_max = 5000

print(dimension,n_hid,batch_size,k,lr)

with open("/mnt/data1/security_train.csv.pkl", "rb") as f:
    labels = pickle.load(f)
    files = pickle.load(f)
with open("/mnt/data1/x_train_word_ids.txt", "rb") as f2:
    x_train_word_ids = pickle.load(f2)

#得到结构视角输入的图
def get_x_str(batch_x):
    # 构建file-api的边列表
    f_a = []
    for i in range(len(batch_x)):
        for j in list(set(batch_x[i])):
            cur = [i, j]
            f_a.append(cur)
    f_a = np.array(f_a)
    f_a = torch.from_numpy(f_a.astype(int))
    data_dict = {
        ('api', 'called_by', 'file'): (f_a[:, 1], f_a[:, 0])
    }
    g = dgl.heterograph(data_dict)
    return g

#得到pos矩阵的2个函数
def get_cos_sim_of_matrix(v1, v2):  # 计算两矩阵余弦相似度
    num = np.dot(v1, np.array(v2).T)  # 向量点乘
    denom = np.linalg.norm(v1, axis=1).reshape(-1, 1) * np.linalg.norm(v2, axis=1)  # 求模长的乘积
    res = num / denom
    res[np.isneginf(res)] = 0
    return 0.5 + 0.5 * res  # 返回余弦相似度矩阵
def get_whether_pos(sim_matrix, k):  # 得到正样本矩阵，一行中1表示互为正样本，0表示互为负样本
    res = []  # 提前定义tfidf向量相似度前k大的互为正样本
    for i in sim_matrix:
        arg = np.argsort(i)
        threshold = i[arg[len(i) - k:]][0]
        cur = i >= threshold
        res.append(cur)
    res = np.array(res).astype(int)
    res = torch.Tensor(res)
    return res  # 返回正样本判断矩阵

# 下游任务分类器
def hcmd_linear(x_train, y_train, x_test):
    from sklearn.linear_model import LogisticRegression
    model = LogisticRegression(penalty='l2', C=1, max_iter=10000)
    model.fit(x_train, y_train)
    y_test = model.predict(x_test)
    return y_test

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

onehot = np.eye(298)#onehot+pca 得到api初始矩阵
pca=PCA(n_components=dimension)
pca.fit(onehot)
onehot_pca = pca.transform(onehot)
emb_api=nn.Parameter(torch.tensor(onehot_pca), requires_grad=False).to(torch.float32).to(device)
emb_files = nn.Parameter(torch.Tensor(len(x_train_word_ids), dimension), requires_grad=False)
emb_file = nn.init.xavier_uniform_(emb_files).to(device)

f_a = []
e_number = []
for i in range(len(x_train_word_ids)):
    leng = len(set(x_train_word_ids[i]))
    for j in list(set(x_train_word_ids[i])):
        count = x_train_word_ids[i].count(j)
        e_number.append(count)
        cur = [i, j]
        f_a.append(cur)
f_a = np.array(f_a)
f_a = torch.from_numpy(f_a.astype(int))
data_dict = {
        ('api', 'called_by', 'file'): (f_a[:, 1], f_a[:, 0])
}
hetero_graph =  dgl.heterograph(data_dict).to(device)

e_num = []
for i in e_number:
    if i>num_max:
        k = num_max
        e_num.append(k)
    else:
        e_num.append(i)
e_tensor = torch.tensor(e_num).to(device)

vectorizer = TfidfVectorizer(use_idf=True, smooth_idf=True, norm=None)
train_features = vectorizer.fit_transform(files)
m = train_features.toarray()
sim_matrix = get_cos_sim_of_matrix(m, m)  # 得到tfidf余弦相似度矩阵
pos = get_whether_pos(sim_matrix, k)

semi_labels = labels[:open_r].astype(int)
semi_labels = torch.tensor(semi_labels, dtype=torch.long).to(device)
semi_x = x_train_word_ids[:open_r]
semi_g = get_x_str(semi_x).to(device)

print('pos tuning!')
for i in range(open_r):
    for j in range(open_r):
        if labels[i]==labels[j]:
            pos[i][j] = 1
            pos[j][i] = 1
        else:
            pos[i][j] = 0
            pos[j][i] = 0
print('pos tune finished!')

model = SCLMD(n_inp,n_hid,n_out,batch_size,tau, lam).to(device)
optimiser = torch.optim.AdamW(model.parameters(),lr=lr)

for epoch in range(epochs):
    num = len(x_train_word_ids)//batch_size
    x_train_word_ids = x_train_word_ids[:batch_size * num]
    if epoch % 5 == 0:
        labels_train = labels[:open_r].astype(float)
        labels_test = labels[9800:].astype(float)
        embeds = model.get_str_embeds(hetero_graph,emb_api,emb_file,e_tensor).cpu().detach().numpy()
        x_train = embeds[:open_r]
        x_test = embeds[9800:]
        pred = hcmd_linear(x_train, labels_train, x_test)
        corr = pred == labels_test
        n_corr = np.sum(corr.astype(int))
        f1_micro = f1_score(labels_test, pred, average='micro')
        f1_macro = f1_score(labels_test, pred, average='macro')
        del pred,corr,n_corr
        print('epoch', epoch, 'micro:', f1_micro, 'macro:', f1_macro)
    loss_all = 0
    for step in range(num):
        batch_x = x_train_word_ids[step*batch_size:(step+1)*batch_size]
        x_seq = torch.LongTensor(pad_sequences(batch_x, maxlen=maxlen)).to(device)
        pos_cur = pos[:, step * batch_size:(step + 1) * batch_size].to(device)
        model.train()
        single_loss = model(pos_cur, x_seq, hetero_graph, emb_api, emb_file, semi_g, semi_labels,e_tensor)
        optimiser.zero_grad()
        single_loss.backward()
        loss_all+=single_loss
        optimiser.step()
        del x_seq,pos_cur,single_loss














