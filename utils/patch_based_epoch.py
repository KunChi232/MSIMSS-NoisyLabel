from sklearn.metrics import roc_auc_score, average_precision_score


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

    def run(self, dataloader):

        self.on_epoch_start()

        logs = {}
        loss_meter = AverageValueMeter()
        metrics_meters = {metric.__name__: AverageValueMeter() for metric in self.metrics}

        with tqdm(dataloader, desc=self.stage_name, file=sys.stdout, disable=not (self.verbose)) as iterator:
            for x, y in iterator:
                x, y = x.to(self.device), y.to(self.device)
                loss, y_pred = self.batch_update(x, y)

                # update loss logs
                loss_value = loss.cpu().detach().numpy()
                loss_meter.add(loss_value)
                loss_logs = {self.loss.__name__: loss_meter.mean}
                logs.update(loss_logs)

                # update metrics logs
                for metric_fn in self.metrics:
                    metric_value = metric_fn(y_pred, y).cpu().detach().numpy()
                    metrics_meters[metric_fn.__name__].add(metric_value)
                metrics_logs = {k: v.mean for k, v in metrics_meters.items()}
                logs.update(metrics_logs)

                if self.verbose:
                    s = self._format_logs(logs)
                    iterator.set_postfix_str(s)

        return logs

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

    def batch_update(self, x, y):
        self.optimizer.zero_grad()
        prediction = self.model.forward(x)
        
        loss = self.loss(prediction, y)
        loss.backward()
        self.optimizer.step()
        return loss, prediction

    
class ValidEpoch(Epoch):

    def __init__(self, model, loss, metrics, lookup, device='cpu', verbose=True):
        super().__init__(
            model=model,
            loss=loss,
            metrics=metrics,
            stage_name='valid',
            device=device,
            verbose=verbose,
        )
        self.lookup = lookup
        
    def on_epoch_start(self):
        self.model.eval()

    def update_patient_prediciton(self, _id, predict, patient_predict):
        predict = torch.softmax(predict, dim = 1)
        predict = torch.argmax(predict, dim = 1)
        predict = predict.detach().cpu().numpy()
        
        for id, pred in zip(_id, predict):
            id = id[:12]
            if(id in patient_predict):
                patient_predict[id][pred] += 1
            else:
                if(pred == 1):
                    patient_predict[id] = [0, 1]
                else:
                    patient_predict[id] = [1, 0]
        return patient_predict
        
    def AUROC(self, y_gt, y_pred):   
        auc = roc_auc_score(y_gt, y_pred)
        
        return {'AUROC' : auc}
    
    def AUPR(self, y_gt, y_pred):
        
        aupr = average_precision_score(y_gt, y_pred)
        
        return {'AUPR' : aupr}    
    def batch_update(self, x, y):
        with torch.no_grad():
            prediction = self.model.forward(x)
            loss = self.loss(prediction, y)
        return loss, prediction
    
    def run(self, dataloader):
        self.on_epoch_start()

        logs = {}
        patient_predict = {}
        loss_meter = AverageValueMeter()
        metrics_meters = {metric.__name__: AverageValueMeter() for metric in self.metrics}

        with tqdm(dataloader, desc=self.stage_name, file=sys.stdout, disable=not (self.verbose)) as iterator:
            for x, y, _id in iterator:
                _id = list(_id)
                x, y = x.to(self.device), y.to(self.device)
                loss, y_pred = self.batch_update(x, y)

                # update patient prediction
                patient_predict = self.update_patient_prediciton(_id, y_pred, patient_predict)
                
            y_pred = []
            y_gt = []
        
            for key, values in patient_predict.items():
                y_gt.append(self.lookup[key])
                d = values[0] + values[1]
                y_pred.append(values[1] / d)
            auroc = self.AUROC(y_gt, y_pred)
            aupr = self.AUPR(y_gt, y_pred)
            
            logs.update(auroc)
            logs.update(aupr)
            
            print('\n')
            print('AUROC : ', auroc['AUROC'])
            print('AUPR : ', aupr['AUPR'])
            
        return logs
    
    
def get_train_test_epoch(model, loss, metrics, optimizer, device, lookup):
    train_epoch = TrainEpoch(
        model, 
        loss=loss, 
        metrics=metrics, 
        optimizer=optimizer,
        device=DEVICE,
        verbose=True,
    )

    valid_epoch = ValidEpoch(
        model, 
        loss=loss, 
        metrics=metrics, 
        device=DEVICE,
        verbose=True,
        lookup=lookup
    )  
    
    return train_epoch, valid_epoch