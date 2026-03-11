
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import torch.distributed as dist
import math
import sys
import numpy as np
import matplotlib.pyplot as plt
import datetime
import os
from camel.distributed import distributed_concat

from camel.utils import roc
import torchvision



def train_KD_multi_teacher(args, model, rank, train_loader, optimizer,lr_decay,name,pt_path,log_dir,distributed='DDP',grad_freeze=None):
    Train_Plot = []
    grad_plot = []
    loss_Plot = []
  
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    if grad_freeze is None:
        model.train()
        
    else:
        model = grad_freeze(model)

   

    for epoch in range(0, args.epoch + 1):

        train_loss = 0
        plot_loss = 0        
        kd_loss =0

        train_loader.sampler.set_epoch(epoch)  
        
        for i, data in enumerate(tqdm(train_loader, 0, leave=False, ncols=70)):

            kd_image = data[0].cuda()  #
            f1 = data[1].cuda()  #
            f2 = data[2].cuda() 
            f3 = data[3].cuda() 
            f4 = data[4].cuda() 
   
           
            temperature = 2.0
   
            p_teacher1 = F.softmax(f1 / temperature, dim=1)  
            p_teacher2 = F.softmax(f2 / temperature, dim=1)  
            p_teacher3 = F.softmax(f3 / temperature, dim=1)  
            p_teacher4 = F.softmax(f4 / temperature, dim=1)  
            
            with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=args.amp):

                optimizer.zero_grad()

               
                student_logits1,student_logits2,student_logits3,student_logits4 = model(kd_image) 

                log_probs_student1 = F.log_softmax(student_logits1 / temperature, dim=1)  
                log_probs_student2 = F.log_softmax(student_logits2 / temperature, dim=1)  
                log_probs_student3 = F.log_softmax(student_logits3 / temperature, dim=1)  
                log_probs_student4 = F.log_softmax(student_logits4 / temperature, dim=1)  

                loss_kd1 = F.kl_div(log_probs_student1, p_teacher1, reduction='batchmean') * (temperature ** 2)
                loss_kd2 = F.kl_div(log_probs_student2, p_teacher2, reduction='batchmean') * (temperature ** 2)
                loss_kd3 = F.kl_div(log_probs_student3, p_teacher3, reduction='batchmean') * (temperature ** 2)
                loss_kd4 = F.kl_div(log_probs_student4, p_teacher4, reduction='batchmean') * (temperature ** 2)

                loss = loss_kd1+loss_kd2+loss_kd3+loss_kd4

                kd_loss +=loss.item()
          
                scaler.scale(loss).backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=5000, norm_type=2)
                scaler.step(optimizer)
                scaler.update()

         
            train_loss += loss.detach().item() * args.batch_size
            plot_loss += loss.detach().item() * args.batch_size
            
            if math.isnan(loss):
                sys.exit('损失报错')
            if (i + 1) % 20000 == 0:
                lr_decay.step()  
                for params in optimizer.param_groups:
                    curr_lr = params['lr']
                    lr_decay.step()  
                print(f'learn_rate:{curr_lr}')
                
            if (i + 1) % 1000 == 0:          
                 
                if rank == 0:  
                    torch.save(optimizer.state_dict(), pt_path + f"optimizer_temp.pt")
                    torch.save(model.state_dict(), pt_path + f"temp.pt")
                    print(f"loss:{round(plot_loss / 1000,6)},kd:{round(kd_loss,5)}")
                  
                    kd_loss =0
                    loss_Plot.append(plot_loss / 1000)
                    plot_loss = 0
                    fig, ax1 = plt.subplots()
                    plt.xticks(rotation=45)
                    ax1.plot(loss_Plot, color="blue", label="Train_Loss")
                    ax1.set_xlabel("item")
                    ax1.set_ylabel("Train_Loss")
                    plt.savefig(os.path.join(f'{log_dir}/logging/Loss_small_{name}.jpg'))
                    plt.close()

                    paramet = list(model.parameters())
                    grad_sum = 0
                    for g in paramet:
                        if g.grad is not None:
                            grad_sum += torch.sum(abs(g.grad)).item()

                    grad_plot.append(grad_sum)
                    fig, ax1 = plt.subplots()
                    plt.xticks(rotation=45)
                    ax1.plot(grad_plot, color="red", label="Grad")
                    ax1.set_xlabel("item")
                    ax1.set_ylabel("Grad")
                    plt.savefig(os.path.join(f'{log_dir}/logging/grad_{name}.jpg'))
                    plt.close()
        train_loss = train_loss / i
        lr_decay.step()  
        for params in optimizer.param_groups:
            curr_lr = params['lr']

        print("\r ", end="")

        if rank == 0:
            print('{},Epoch:{},Train_Loss:{},lr:{}'.format(datetime.datetime.now(), epoch, round(train_loss, 5), curr_lr))

            Train_Plot.append(train_loss)
            fig, ax1 = plt.subplots()
            plt.xticks(rotation=45)
            ax1.plot(Train_Plot, color="blue", label="Train_Loss")
            ax1.set_xlabel("item")
            ax1.set_ylabel("Train_Loss")
            plt.savefig(os.path.join(f'{log_dir}/logging/Loss_{name}.jpg'))
            plt.close()

            result_record = f'time:{datetime.datetime.now()},Epoch:{epoch},Train_Loss:{round(train_loss, 5)}\n' \
                        f'batch_size:{args.batch_size},learn_rate:{curr_lr}'
                
            with open(f'{log_dir}/logging/result_{name}.txt', mode='a') as f:
                f.write(result_record + '\n')

       
        if rank == 0:  
            torch.save(model.state_dict(), pt_path + f"Encoder_{epoch}.pt")
    dist.destroy_process_group()


