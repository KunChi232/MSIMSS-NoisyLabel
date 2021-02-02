# +
class Epoch:

    def __init__(self, model, loss, metrics, stage_name, device='cpu', verbose=True):
        self.model = model
        self.loss = loss
        self.metrics = metrics
        self.stage_name = stage_name
        self.verbose = verbose
        self.device = device

        self._to_device()

    def _to_device(self):
        self.model.to(self.device)
        self.loss.to(self.device)
        for metric in self.metrics:
            metric.to(self.device)

    def _format_logs(self, logs):
        str_logs = ['{} - {:.4}'.format(k, v) for k, v in logs.items()]
        s = ', '.join(str_logs)
        return s

    def batch_update(self, x, y):
        raise NotImplementedError

    def on_epoch_start(self):
        pass


class TrainEpoch(Epoch):

    def __init__(self, model, loss, metrics, optimizer, device='cpu', verbose=True):
        super().__init__(
            model=model,
            loss=loss,
            metrics=metrics,
            stage_name='train',
            device=device,
            verbose=verbose,
        )
        self.optimizer = optimizer
        
    def on_epoch_start(self):
        self.model.train()

    def batch_update(self, x, knn_y, ori_y, gamma):
        self.optimizer.zero_grad()
        prediction, feat = self.model.forward(x)
        del feat
        
        loss1 = self.SoftCrossEntropy(prediction, knn_y).mean()
        loss2 = self.loss(prediction, ori_y).mean()
        loss = loss1 + loss2
        loss.backward()
        self.optimizer.step()
                
        return (loss, loss1, loss2), prediction
    
    def SoftCrossEntropy(self, inputs, target, reduction='sum'):
        log_likelihood = -F.log_softmax(inputs, dim=1)
        batch = inputs.shape[0]
        if reduction == 'average':
            loss = torch.sum(torch.mul(log_likelihood, target)) / batch
        elif reduction == 'sum':
            loss = torch.sum(torch.mul(log_likelihood, target), dim = 1)
        elif reduction == 'none':
            loss = torch.mul(log_likelihood, target)
        return loss
    
    
    def run(self, dataloader, gamma):

            self.on_epoch_start()

            logs = {}
            total_loss_meter = AverageValueMeter()
            ce_loss_meter = AverageValueMeter()
            softce_loss_meter = AverageValueMeter()
            metrics_meters = {metric.__name__: AverageValueMeter() for metric in self.metrics}

            with tqdm(dataloader, desc=self.stage_name, file=sys.stdout, disable=not (self.verbose)) as iterator:
                for x, knn_y, ori_y in iterator:
                    x, knn_y, ori_y = x.to(self.device), knn_y.to(self.device), ori_y.to(self.device)
                    loss, y_pred = self.batch_update(x, knn_y, ori_y, gamma)


                    # update loss logs
                    loss_value = loss[0].cpu().detach().numpy()
                    total_loss_meter.add(loss_value)
                    loss_logs = {'Total loss': total_loss_meter.mean}
                    logs.update(loss_logs)
                    
                    loss_value = loss[2].cpu().detach().numpy()
                    ce_loss_meter.add(loss_value)
                    loss_logs = {'CE loss': ce_loss_meter.mean}
                    logs.update(loss_logs)
                    
                    loss_value = loss[1].cpu().detach().numpy()
                    softce_loss_meter.add(loss_value)
                    loss_logs = {'SoftCE loss': softce_loss_meter.mean}
                    logs.update(loss_logs)
                    
                    # update metrics logs
                    for metric_fn in self.metrics:
                        metric_value = metric_fn(y_pred, ori_y).cpu().detach().numpy()
                        metrics_meters[metric_fn.__name__].add(metric_value)
                    metrics_logs = {k: v.mean for k, v in metrics_meters.items()}
                    logs.update(metrics_logs)

                    if self.verbose:
                        s = self._format_logs(logs)
                        iterator.set_postfix_str(s)
            return logs  

class ValidEpoch(Epoch):

    def __init__(self, model, loss, metrics, lookup, int_patch_lookup, device='cpu', verbose=True):
        super().__init__(
            model=model,
            loss=loss,
            metrics=metrics,
            stage_name='valid',
            device=device,
            verbose=verbose,
        )
        self.lookup = lookup
        self.int_patch_lookup = int_patch_lookup
        
    def on_epoch_start(self):
        self.model.eval()

    def update_patient_prediciton(self, patch_int, predict, patient_predict):
        predict = torch.softmax(predict, dim = 1)
        predict = torch.argmax(predict, dim = 1)
        predict = predict.detach().cpu().numpy()
        
        for _int, pred in zip(patch_int, predict):
            patch_name = self.int_patch_lookup[_int]
            p_id = patch_name[:12]
            if(p_id in patient_predict):
                patient_predict[p_id][pred] += 1
            else:
                if(pred == 1):
                    patient_predict[p_id] = [0, 1]
                else:
                    patient_predict[p_id] = [1, 0]
        return patient_predict
        
    def AUROC(self, y_pred, y_gt):   

        auc = roc_auc_score(y_gt, y_pred)
        
        return {'AUROC' : auc}

    def AUPR(self, y_pred, y_gt):   

        aupr = average_precision_score(y_gt, y_pred)
        
        return {'AUPR' : aupr}    
    
    def Fscore(self, pr, gt, beta=1, eps=1e-7, threshold=[0.5, 0.4, 0.3, 0.2]):
        
        scores = dict()
        for t in threshold:
            pr = _threshold(pr, threshold=t)

            tp = np.sum(gt * pr)
            fp = np.sum(pr) - tp
            fn = np.sum(gt) - tp

            score = ((1 + beta ** 2) * tp + eps) \
                    / ((1 + beta ** 2) * tp + beta ** 2 * fn + fp + eps)

            scores['fscore_'+str(t)] = score

        return scores
    
    def batch_update(self, x, y):
        with torch.no_grad():
            prediction, _ = self.model.forward(x)
        return prediction
    
    def run(self, dataloader):
        self.on_epoch_start()

        logs = {}
        patient_predict = {}
        loss_meter = AverageValueMeter()
        metrics_meters = {metric.__name__: AverageValueMeter() for metric in self.metrics}

        with tqdm(dataloader, desc=self.stage_name, file=sys.stdout, disable=not (self.verbose)) as iterator:
            for x, y, patch_int in iterator:
                patch_int = patch_int.numpy()
                x, y = x.to(self.device), y.to(self.device)
                y_pred = self.batch_update(x, y)

                # update patient prediction
                patient_predict = self.update_patient_prediciton(patch_int, y_pred, patient_predict)
            
            
            y_pred = []
            y_gt = []
        
            for key, values in patient_predict.items():
                y_gt.append(self.lookup[key])
                d = values[0] + values[1]
                y_pred.append(values[1] / d)
                
            
            auroc = self.AUROC(y_pred, y_gt)
            aupr = self.AUPR(y_pred, y_gt)
            logs.update(auroc)
            logs.update(aupr)
            
            print('\n')
            print('AUROC : ', auroc['AUROC'])
            print('AUPR : ', aupr['AUPR'])
        return logs        
        
        
class ExtractEpoch():

    def __init__(self, model, device='cpu', verbose=True):

        self.model = model
        self.device = device
        self.stage_name='extra'
        self.model.to(self.device)
        self.verbose = verbose
    def on_epoch_start(self):
        self.model.eval()
    
    def batch_update(self, x, y):
        with torch.no_grad():
            pred, feat = self.model.forward(x)
        return pred, feat
    
    def run(self, dataloader, average_pred, alpha = 0.99):

        self.on_epoch_start()

        embedding = {}
        with tqdm(dataloader, desc=self.stage_name, file=sys.stdout, disable=not (self.verbose)) as iterator:
            for x, y, patch_int in iterator:
                patch_int = patch_int.numpy()
                x, y = x.to(self.device), y.to(self.device)
                pred, feat = self.batch_update(x, y)
                pred = torch.softmax(pred, dim = 1).detach().cpu().numpy()
                for i, p_int in enumerate(patch_int):
                    embedding[p_int] = feat[i].detach().cpu().numpy()
                    average_pred[p_int] = (1 - alpha) * average_pred[p_int] + alpha * pred[i]
                del feat
        return embedding, average_pred
    
    
    
def get_train_test_epoch(model, loss, metrics, optimizer, DEVICE, int_patch_lookup, lookup):
    train_epoch = TrainEpoch(
        model, 
        loss=loss, 
        metrics=metrics, 
        optimizer=optimizer,
        device=DEVICE,
        verbose=True,
    )

    extract_epoch = ExtractEpoch(
        model, 
        device=DEVICE,
        verbose=True,
    )
    
    valid_epoch = ValidEpoch(
        model, 
        loss=loss, 
        metrics=metrics,
        lookup=lookup,
        int_patch_lookup=int_patch_lookup,
        device=DEVICE,
        verbose=True,
    )
    
    return train_epoch, extract_epoch, valid_epoch
