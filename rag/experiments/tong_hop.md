| Cấu hình | Cắt | Tìm kiếm | Số chunk | Hit@5 | MRR@10 | Hạng 1 | Giao < 100 ký tự | So với | Hiệu MRR@10 [CI 95%] | Kết luận |
|---|---|---|---|---|---|---|---|---|---|---|
| B0 | fixed | dense | 178 | 0.875 | 0.736 | 50/80 | 2 | – | – | – |
| E1-sentence | sentence | dense | 180 | 0.875 | 0.715 | 49/80 | 1 | B0 | -0.021 [-0.064; +0.022] | không đủ bằng chứng |
| E1-recursive | recursive | dense | 221 | 0.863 | 0.692 | 46/80 | 0 | B0 | -0.044 [-0.108; +0.019] | không đủ bằng chứng |
| E2-hybrid | fixed | hybrid | 178 | 0.887 | 0.811 | 59/80 | 0 | B0 | +0.075 [+0.005; +0.147] | cải thiện |
| E3-rerank | fixed | hybrid+rerank | 178 | 0.975 | 0.896 | 67/80 | 1 | E2-hybrid | +0.086 [+0.024; +0.150] | cải thiện |
