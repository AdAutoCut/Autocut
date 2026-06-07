# !/usr/bin/env python
# -*-coding:utf-8 -*-

"""
# Author     ：Jian Jia
# File       : online_train_photo_1011-1018.py
# Time       ：2024/11/11 14:10
"""
import pickle

import numpy as np

# merge online training photo and user click sequence photo

# item 侧需要的photo
with open('/phd/content_ID/outer_rank/quantizer/mmu_embedding/ad_outer/online_1011-1018.pkl', 'rb') as rf:
    idx2photoid, idx2emb = pickle.load(rf)


with open('/phd/content_ID/outer_rank/quantizer/mmu_embedding/ad_outer/online_user_click_seq.pkl', 'rb') as rf:
    user_idx2photoid, user_idx2emb = pickle.load(rf)


update_user_idx2photoid = {k+len(idx2photoid): v for k, v in user_idx2photoid.items()}

all_idx2photoid = {**idx2photoid, **update_user_idx2photoid}
all_idx2emb = np.concatenate([idx2emb, user_idx2emb], axis=0)


with open('/phd/content_ID/outer_rank/quantizer/mmu_embedding/ad_outer/online_all_1011-1018.pkl', 'wb') as wf:
    pickle.dump((all_idx2photoid, all_idx2emb), wf)