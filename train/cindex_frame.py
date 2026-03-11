import sys
import os
import torch
import torch.nn as nn
import warnings
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import numpy as np
from lifelines.utils import concordance_index 
import pandas as pd 

warnings.simplefilter("ignore")
sys.setrecursionlimit(10000000)



class PatchDatasetForTrain(Dataset):
    def __init__(self, feature_list, event_list, time_list):
        self.feature = feature_list
        self.event = event_list
        self.time = time_list

    def __getitem__(self, item):
        img_feature = self.feature[item]
        event = torch.tensor(self.event[item], dtype=torch.float32)
        time = torch.tensor(self.time[item], dtype=torch.float32)
        return img_feature, event, time

    def __len__(self):
        return len(self.feature)

class PatientLevelDatasetForTest(Dataset):
    def __init__(self, data_list_filepath):
        super().__init__()
        self.patient_data_paths = []
        with open(data_list_filepath, 'r', encoding='utf_8_sig') as f:
            for line in f:
                line = line.strip()
                if not line: continue
                parts = line.split(',')
                feature_path = parts[0]
                true_event = int(parts[1])
                true_time = int(parts[2])

              
                base_name = os.path.basename(feature_path).split('.')[0]
                slice_id_parts = base_name.split('_')
               
                if len(slice_id_parts) >= 2:
                    slice_id = '_'.join(slice_id_parts[1:])
                else:
                    slice_id = base_name

                self.patient_data_paths.append({
                    'feature_path': feature_path,
                    'true_event': true_event,
                    'true_time': true_time,
                    'slice_id': slice_id 
                })

    def __getitem__(self, item):
        patient_info = self.patient_data_paths[item]

      
        features = torch.load(patient_info['feature_path'], map_location=torch.device('cpu'))
       
        if features.dim() > 2:
            features = features.view(-1, 1536) 
        elif features.dim() == 1: 
            features = features.unsqueeze(0) 
        elif features.shape[1] != 1536: 
            raise ValueError(f"Feature dimension mismatch for {patient_info['feature_path']}: expected 1536, got {features.shape[1]}")


        return {
            'features': features,
            'event': patient_info['true_event'],
            'time': patient_info['true_time'],
            'slice_id': patient_info['slice_id'] 
        }

    def __len__(self):
        return len(self.patient_data_paths)


def cox_loss(risk_scores, time, event):
   
    risk_scores = risk_scores.squeeze()
    
    time = time.squeeze()
    event = event.squeeze()

    
    _, order = torch.sort(time, descending=True)
    risk_scores_sorted = risk_scores[order]
    event_sorted = event[order]

    
    hazard_ratio = torch.exp(risk_scores_sorted)
    

    log_risk = torch.log(torch.cumsum(hazard_ratio, dim=0) + 1e-6)


    uncensored_likelihood = risk_scores_sorted[event_sorted.bool()] - log_risk[event_sorted.bool()]
    

    num_events = torch.sum(event)
    loss = -torch.sum(uncensored_likelihood) / (num_events + 1e-6) 
    return loss



def main(model, train_filepath, test_filepath, BatchSize, lr=1e-4, num_work=8, model_save_dir="."):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, eps=1e-8)

    all_train_features = []
    all_train_events = []
    all_train_times = []
    
    with open(train_filepath, 'r', encoding='utf_8_sig') as f:
        train_patient_paths = f.read().strip().split('\n')
    
    for entry in tqdm(train_patient_paths, desc="Loading training data"):
        if not entry: continue
        parts = entry.split(',')
        feature_pt_path = parts[0]
        event = int(parts[1])
        time = int(parts[2])

     
        pat_features = torch.load(feature_pt_path, map_location=torch.device('cpu'))
        if pat_features.dim() > 2:
            pat_features = pat_features.view(-1, 1536)
        elif pat_features.dim() == 1:
            pat_features = pat_features.unsqueeze(0)
        elif pat_features.shape[1] != 1536:
             raise ValueError(f"Feature dimension mismatch for {feature_pt_path}: expected 1536, got {pat_features.shape[1]}")


        for patch_feature in pat_features:
            all_train_features.append(patch_feature)
            all_train_events.append(event) 
            all_train_times.append(time)
            
    train_dataset = PatchDatasetForTrain(all_train_features, all_train_events, all_train_times)
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=BatchSize,
        shuffle=True,
        num_workers=num_work,
        pin_memory=True,
        drop_last=True
    )


    test_dataset = PatientLevelDatasetForTest(test_filepath)
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=1, 
        shuffle=False,
        num_workers=num_work,
        pin_memory=True,
        drop_last=False
    )
   
    best_c_index_val = -1.0 
   
    for epoch in range(0, 100):
      
        model.train()
        train_loss = 0.0
        for i, data in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1} Training")):
            features, event, time = data
            features = features.cuda()
            event = event.cuda()
            time = time.cuda()

            optimizer.zero_grad()
            risk_scores = model(features)
            loss = cox_loss(risk_scores, time, event)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)

        model.eval()
        all_patient_risk_scores = []
        all_patient_true_events = []
        all_patient_true_times = []
        all_patient_slice_ids = [] 
        
        with torch.no_grad():
            for i, patient_data in enumerate(tqdm(test_loader, desc=f"Epoch {epoch+1} Evaluation")):
                
                features = patient_data['features'].squeeze(0).cuda() 
                true_event = patient_data['event'].item()
                true_time = patient_data['time'].item()
                slice_id = patient_data['slice_id'][0] 

                patch_risk_scores = model(features)
                
               
                patient_level_risk = torch.mean(patch_risk_scores).item()

                all_patient_risk_scores.append(patient_level_risk)
                all_patient_true_events.append(true_event)
                all_patient_true_times.append(true_time)
                all_patient_slice_ids.append(slice_id)

        
        scores_np = np.array(all_patient_risk_scores)
        events_np = np.array(all_patient_true_events)
        times_np = np.array(all_patient_true_times)

        current_c_index = concordance_index(
            event_times=times_np,
            predicted_scores=-scores_np, 
            event_observed=events_np
        )
        
 
 
        if current_c_index > best_c_index_val:
            best_c_index_val = current_c_index
            
      
            model_save_path = os.path.join(model_save_dir, f"cox_best_cindex_{best_c_index_val:.4f}.pt")
            torch.save(model.state_dict(), model_save_path)
   
            results_df = pd.DataFrame({
                'PatientID': all_patient_slice_ids,
                'TrueEvent': all_patient_true_events,
                'TrueTime': all_patient_true_times,
                'PredictedRiskScore': all_patient_risk_scores
            })
            excel_save_path = os.path.join(model_save_dir, f"risk_scores_best_cindex_{best_c_index_val:.4f}.xlsx")
            results_df.to_excel(excel_save_path, index=False)
   
            Test(model, test_filepath='',  num_work=8,model_save_dir=MODEL_SAVE_DIR)
   
        else:
             print(f"当前C-Index {current_c_index:.4f} 未超越最佳 {best_c_index_val:.4f}. 模型和风险分数未保存.")





def Test(model, test_filepath,  num_work=8, model_save_dir="."):
    

    
    test_dataset = PatientLevelDatasetForTest(test_filepath)
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=1, 
        shuffle=False,
        num_workers=num_work,
        pin_memory=True,
        drop_last=False
    )

    model.eval()
    all_patient_risk_scores = []
    all_patient_true_events = []
    all_patient_true_times = []
    all_patient_slice_ids = [] 
    
    with torch.no_grad():
        for i, patient_data in enumerate(tqdm(test_loader)):
            
            features = patient_data['features'].squeeze(0).cuda() 
            true_event = patient_data['event'].item()
            true_time = patient_data['time'].item()
            slice_id = patient_data['slice_id'][0] 

            patch_risk_scores = model(features)
            
 
            patient_level_risk = torch.mean(patch_risk_scores).item()


            all_patient_risk_scores.append(patient_level_risk)
            all_patient_true_events.append(true_event)
            all_patient_true_times.append(true_time)
            all_patient_slice_ids.append(slice_id)

  
        scores_np = np.array(all_patient_risk_scores)
        events_np = np.array(all_patient_true_events)
        times_np = np.array(all_patient_true_times)

   
        current_c_index = concordance_index(
            event_times=times_np,
            predicted_scores=-scores_np, 
            event_observed=events_np
        )
        

        results_df = pd.DataFrame({
            'PatientID': all_patient_slice_ids,
            'TrueEvent': all_patient_true_events,
            'TrueTime': all_patient_true_times,
            'PredictedRiskScore': all_patient_risk_scores
        })
        excel_save_path = os.path.join(model_save_dir, f"TCGA_risk_scores_best_cindex_{current_c_index:.4f}.xlsx")
        results_df.to_excel(excel_save_path, index=False)



class SimpleMLP(nn.Module):
    def __init__(self, input_dim=1536, hidden_dim=128):
        super().__init__()
        
        self.net = nn.Sequential(
           
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim), 
            nn.LeakyReLU(),
          
            nn.Linear(hidden_dim, 1) 
        )

    def forward(self, x):
      
        return self.net(x)


if __name__ == '__main__':

    TRAIN_FILEPATH = ''
    TEST_FILEPATH = ''

    MODEL_SAVE_DIR = ""
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True) 

    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {DEVICE}")

    model = SimpleMLP(input_dim=1536).cuda()

    main(model,
         train_filepath=TRAIN_FILEPATH,
         test_filepath=TEST_FILEPATH,
         BatchSize=512, 
         lr=1e-4,
         num_work=0,
         model_save_dir=MODEL_SAVE_DIR 
         )

  