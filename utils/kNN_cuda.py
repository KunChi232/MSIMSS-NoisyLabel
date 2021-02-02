from knn_cuda import KNN

class IterKNN(nn.Module):
    def __init__(self, sharpen = 1.0):
        super().__init__()
        self.k = 50
        self.knn = KNN(self.k, transpose_mode=True)
        self.sharpen = sharpen
    def forward(self, embedding, average_pred):
        patch_name = [k for k in average_pred.keys()]
        ref = [embedding[k] for k in patch_name]
        ref_y = [np.array(average_pred[k]) for k in patch_name]
        ref = np.array(ref)
        query = ref.copy()
        ref_y = np.array(ref_y)
        pred_y = []

        ref = torch.from_numpy(ref).cuda()
        ref = ref.unsqueeze(0)
        for b in tqdm(range(math.ceil(len(ref[0]) / 2048)), desc = 'kNN  ', file=sys.stdout):
            q = torch.from_numpy(query[b * 2048 : (b+1) * 2048]).cuda()
            q = q.unsqueeze(0)
            dist, indx = self.knn(ref, q)
            del dist
            indx = indx.detach().cpu().numpy()[0]
            for i in range(len(indx)):
                n_pseudo = ref_y[indx[i]]
                temp = np.array([0., 0.])
                temp[0] = sum(v[0] for v in n_pseudo) / self.k
                temp[1] = sum(v[1] for v in n_pseudo) / self.k
                temp = temp**(1/self.sharpen)
                temp = temp / temp.sum()
                pred_y.append(temp)
                
        pesudo_label = dict(zip(patch_name, pred_y))
        return pesudo_label
        