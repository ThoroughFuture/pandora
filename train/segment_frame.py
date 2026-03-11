import sys
sys.path.append('/home/yangmoxuan/main_camel/')
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import torch.distributed as dist
import math
import numpy as np
import matplotlib.pyplot as plt
import datetime
import os
from camel.utils import roc,slice_image,Dice,multiclassification_dice_with_iou,dice_coefficient,iou_coefficient
from camel.utils.loss import BceWithLogDiceLoss,Ce_with_Dice_loss, Focal_with_Dice_loss
from camel.distributed import distributed_concat
import itertools
import datetime
import  random
# current_path = os.path.abspath(__file__)# 获取当前脚本的绝对路径
# current_dir = os.path.dirname(current_path)# 获取当前脚本所在的目录
# log_dir = os.path.dirname(current_dir)# 获取上级目录


def Train(args, model, rank, train_loader, optimizer,lr_decay,name,pt_path,log_dir,distributed='DDP',grad_freeze=None):
    Train_Plot = []
    grad_plot = []
    loss_Plot = []

    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)
    bce_loss = BceWithLogDiceLoss().cuda()

    for epoch in range(0, args.epoch):
        
        if grad_freeze is None:
            model.train()
            model.module.backbone.eval()
        else:
            model = grad_freeze(model)

        Loss_Train = 0
        plot_loss = 0
        train_loader.sampler.set_epoch(epoch)  

        label_roc=[]
        pred_roc=[]
        
        
        for i, data in enumerate(tqdm(train_loader, 0, leave=False, ncols=70)):

            optimizer.zero_grad()
            In = data[0].cuda()  
            real_label = data[2].cuda()
            train_label = data[1].cuda()

            with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=args.amp):
                
                out = model(In)
                
                for b in range(0,In.shape[0]):  
                    mask_out = torch.tensor((real_label[b] < 0.1)).int()*(-9999)
             
                    out[b] = torch.add(mask_out.cuda(),out[b])
                  
                sm = torch.sigmoid(out)
                
                loss = bce_loss(train_label, out)

                if math.isnan(loss):
                    sys.exit('损失报错')

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            Loss_Train += loss.item() * args.batch_size
            plot_loss += loss.detach().item() * args.batch_size

            evalutate_size = 64
            sm=slice_image(sm,patch_size=evalutate_size).flatten(1)
            real_label=slice_image(real_label,patch_size=evalutate_size).flatten(1)
            is_zero_row = (real_label == 0).all(dim=1)
            
            non_zero_indices = ~is_zero_row
            
            real_label = real_label[non_zero_indices, :]
            sm = sm[non_zero_indices, :]
            sm = torch.mean(sm,dim=1).detach().cpu()
            real_label2 = torch.sum(real_label,dim=1)

            for kk in range(0,sm.shape[0]):
                if 1 in real_label[kk,:] and real_label2[kk]>= evalutate_size*evalutate_size*0.75: #128*128*1*0.75
                    pred_roc.append(sm[kk])
                    label_roc.append(1)   

                        
                elif float(254/255) in real_label[kk] :#and 0 not in label_crop:
                    pred_roc.append(sm[kk])
                    label_roc.append(0)

            if (i + 1) % 20000 == 0:
                dist.barrier()
                lr_decay.step()  
                for params in optimizer.param_groups:
                    curr_lr = params['lr']
                    lr_decay.step()  
                print(f'learn_rate:{curr_lr}')
           
            if (i + 1) % 500 == 0:
                
                dist.barrier()

                label_roc_np = np.array(label_roc)
                pred_roc_np = np.array(pred_roc)
                cutoff = roc.roc(label_roc_np,pred_roc_np,epoch,f'{name}',log_dir) 
                mask = torch.tensor((pred_roc_np > cutoff).astype(int)) * torch.ones_like(torch.from_numpy(pred_roc_np))
                dsc, iou = Dice(mask, torch.from_numpy(pred_roc_np))

                if (i + 1) % 10000 == 0:
                    
                    label_roc=[]
                    pred_roc=[]
                    
                if args.world_size !=1:
                    dsc = dsc.unsqueeze(0).unsqueeze(0).cuda()
                    iou = iou.unsqueeze(0).unsqueeze(0).cuda()
        
                    dsc = distributed_concat(dsc, args.world_size)
                    iou = distributed_concat(iou, args.world_size)

                    dsc = torch.mean(dsc)
                    iou = torch.mean(iou)
                if rank == 0: 
                    torch.save(optimizer.state_dict(), pt_path + f"optimizer_temp.pt")

                    torch.save(model.state_dict(), pt_path + f"temp.pt")
                    print(f"loss:{round(plot_loss / 1000, 6)},dice:{round(dsc.item(), 4)},iou:{round(iou.item(), 4)}")

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
            dist.barrier()
                    

        Loss_Train = Loss_Train / i

        label_roc_np = np.array(label_roc)
        pred_roc_np = np.array(pred_roc)
        cutoff = roc.roc(label_roc_np,pred_roc_np,epoch,name,log_dir) 
        mask = torch.tensor((pred_roc_np > cutoff).astype(int)) * torch.ones_like(torch.from_numpy(pred_roc_np))
        dsc, iou = Dice(mask, torch.from_numpy(pred_roc_np))

        if args.world_size !=1:
            dsc = dsc.unsqueeze(0).unsqueeze(0).cuda()
            iou = iou.unsqueeze(0).unsqueeze(0).cuda()
            dsc = distributed_concat(dsc, args.world_size)
            iou = distributed_concat(iou, args.world_size)
            dsc = torch.mean(dsc)
            iou = torch.mean(iou)           


        if rank == 0:
            
            torch.save(optimizer.state_dict(), pt_path + f"optimizer_temp.pt")
            torch.save(model.state_dict(), pt_path + f"{name}_{epoch}.pt")
            lr_decay.step()
            for params in optimizer.param_groups:
                curr_lr = params['lr']

            print("\r ", end="")
            print('{},Epoch:{},Train_Loss:{},dice:{},iou:{}'.format(datetime.datetime.now(), epoch, round(Loss_Train, 5),round(dsc.item(), 3),round(iou.item(), 3)))
            result_record = f'time:{datetime.datetime.now()},Epoch:{epoch},Train_Loss:{round(Loss_Train, 5)}\n' \
                            f'dice:{round(dsc.item(), 4)},iou:{round(iou.item(), 4)}\n' \
                            f'batch_size:{args.batch_size},learn_rate:{curr_lr}'

            with open(f'{log_dir}/logging/{name}.txt', mode='a') as f:
                f.write(result_record + '\n')

            Train_Plot.append(Loss_Train)
            fig, ax1 = plt.subplots()
            plt.xticks(rotation=45)
            ax1.plot(Train_Plot, color="blue", label="Train_Loss")
            ax1.set_xlabel("item")
            ax1.set_ylabel("Train_Loss")
            plt.savefig(os.path.join(f'{log_dir}/logging/{name}.png'))
            plt.close()

    dist.destroy_process_group()
    




def Train_classification(args, model, rank, train_loader, optimizer,lr_decay,name,pt_path,log_dir,distributed='DDP',grad_freeze=None):
    Train_Plot = []
    grad_plot = []
    loss_Plot = []

    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)
    CE_loss = Ce_with_Dice_loss( ignore_index=0).cuda()
   # CE_loss = Focal_with_Dice_loss(focal_weight=3.0, dice_weight=1, ignore_index=0, alpha=[0, 0.4, 0.1, 0.5], gamma=2).cuda()

    for epoch in range(0, args.epoch):
        
        if grad_freeze is None:
            model.train()
            model.module.backbone.eval()
        else:
            model = grad_freeze(model)

        Loss_Train = 0
        plot_loss = 0
        train_loader.sampler.set_epoch(epoch)  

        scores = []


        for i, data in enumerate(tqdm(train_loader, 0, leave=False, ncols=70)):

            optimizer.zero_grad()
            In = data[0].cuda()  
            real_label = data[2].cuda()

            real_label = torch.where(real_label == 254, torch.tensor(3).cuda(), real_label)
            # real_label = torch.where(real_label == 2, torch.tensor(1).cuda(), real_label)
            # real_label = torch.where(real_label == 3, torch.tensor(2).cuda(), real_label)


            #train_label = data[1].cuda()
       
            with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=args.amp):
                out = model(In)
                

                sm = torch.softmax(out,dim=1)
                sm = sm.argmax(dim=1)  # B W H
               
                loss = CE_loss(out,real_label)

                if math.isnan(loss):
                    sys.exit('损失报错')

            scaler.scale(loss).backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5, norm_type=2)
            scaler.step(optimizer)
            scaler.update()
            
            Loss_Train += loss.item() * args.batch_size
            plot_loss += loss.detach().item() * args.batch_size


          
            
            num_classes = out.shape[1]  
   
            
            dice_score = dice_coefficient(sm, real_label, num_classes, ignore_index=0)
            iou_score = iou_coefficient(sm, real_label, num_classes, ignore_index=0)

            scores.append((dice_score, iou_score))

            if (i + 1) % 20000 == 0:
                dist.barrier()
                lr_decay.step()  
                for params in optimizer.param_groups:
                    curr_lr = params['lr']
                    lr_decay.step()  
                print(f'learn_rate:{curr_lr}')

            if (i + 1) % 500 == 0:
                dist.barrier()
                avg_dice = [sum(x[i] for x, _ in scores) / len(scores) for i in range(num_classes-1)]
                avg_iou = [sum(x[i] for _, x in scores) / len(scores) for i in range(num_classes-1)]
                # avg_dice = [sum(x[i] for x, _ in scores) / len(scores) for i in range(num_classes)]
                # avg_iou = [sum(x[i] for _, x in scores) / len(scores) for i in range(num_classes)]


                mean_dice = torch.tensor(avg_dice)
                mean_iou = torch.tensor(avg_iou)

                if args.world_size != 1:
                    mean_dice = mean_dice.unsqueeze(0).unsqueeze(0).cuda()
                    mean_iou = mean_iou.unsqueeze(0).unsqueeze(0).cuda()

                    mean_dice = distributed_concat(mean_dice, args.world_size)
                    mean_iou = distributed_concat(mean_iou, args.world_size)

                    mean_dice = torch.mean(mean_dice, dim=(0, 1))
                    mean_iou = torch.mean(mean_iou, dim=(0, 1))

                if rank == 0:
                    current_time = datetime.datetime.now()
                    
                    time_str = current_time.strftime("%Y%m%d_%H%M%S")

                       
                    torch.save(optimizer.state_dict(), pt_path + f"optimizer_temp.pt")
                    torch.save(model.state_dict(), pt_path + f"temp.pt")

                    print(f"loss:{round(plot_loss / 1000, 6)}, dice:{[round(x, 4) for x in mean_dice.tolist()]}, iou:{[round(x, 4) for x in mean_iou.tolist()]}")


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
            dist.barrier()
                    

        Loss_Train = Loss_Train / i
        avg_dice = [sum(x[i] for x, _ in scores) / len(scores) for i in range(num_classes-1)]
        avg_iou = [sum(x[i] for _, x in scores) / len(scores) for i in range(num_classes-1)]

        mean_dice = torch.tensor(avg_dice)
        mean_iou = torch.tensor(avg_iou)

        if args.world_size !=1:

            #accuracy = accuracy.unsqueeze(0).unsqueeze(0).cuda()
            mean_dice =  mean_dice.unsqueeze(0).unsqueeze(0).cuda()
            mean_iou =  mean_iou.unsqueeze(0).unsqueeze(0).cuda()

           # accuracy = distributed_concat(accuracy, args.world_size)
            mean_dice = distributed_concat(mean_dice, args.world_size)
            mean_iou = distributed_concat(mean_iou, args.world_size)

            #accuracy = torch.mean(accuracy)
            mean_dice = torch.mean(mean_dice)
            mean_iou = torch.mean(mean_iou)       


        if rank == 0:
            
            torch.save(optimizer.state_dict(), pt_path + f"optimizer_temp.pt")
            torch.save(model.state_dict(), pt_path + f"{name}_{epoch}.pt")
            lr_decay.step()
            for params in optimizer.param_groups:
                curr_lr = params['lr']

            print("\r ", end="")
            #print(f"epoch:{epoch},loss:{round(plot_loss / 1000, 6)}, dice:{[round(x, 4) for x in mean_dice.tolist()]}, iou:{[round(x, 4) for x in mean_iou.tolist()]}")
            print(f"epoch:{epoch}, loss:{round(plot_loss / 1000, 6)}, dice:{round(mean_dice.item(), 4)}, iou:{round(mean_iou.item(), 4)}")
            result_record = f'time:{datetime.datetime.now()},Epoch:{epoch},Train_Loss:{round(Loss_Train, 5)}\n' \
                            f'dice:{round(mean_dice.item(), 4)}, iou:{round(mean_iou.item(), 4)}' \
                            f'batch_size:{args.batch_size},learn_rate:{curr_lr}'

            with open(f'{log_dir}/logging/{name}.txt', mode='a') as f:
                f.write(result_record + '\n')

            Train_Plot.append(Loss_Train)
            fig, ax1 = plt.subplots()
            plt.xticks(rotation=45)
            ax1.plot(Train_Plot, color="blue", label="Train_Loss")
            ax1.set_xlabel("item")
            ax1.set_ylabel("Train_Loss")
            plt.savefig(os.path.join(f'{log_dir}/logging/{name}.png'))
            plt.close()

    dist.destroy_process_group()






def Test(args, model,rank, test_loader,name,log_dir,distributed='DDP'):
    model.eval()
    label_roc_test=[]
    pred_roc_test=[]
    test_loader.sampler.set_epoch(1)  

    for i, data in enumerate(tqdm(test_loader, 0, leave=False, ncols=70)):

        In = data[0].cuda()  
        real_label = data[1].cuda()
        
        with torch.no_grad():
            out = model(In)
            sm = torch.sigmoid(out)

        evalutate_size = 32

        # for b in range(0,In.shape[0]):  
        #     for x in range(0,In.shape[2]-evalutate_size+1,evalutate_size):
        #         for y in range(0,In.shape[3]-evalutate_size+1,evalutate_size):
        #             heatmap_crop=sm[b,:,x:x+evalutate_size,y:y+evalutate_size].mean().item()
        #             label_crop=real_label[b,:,x:x+evalutate_size,y:y+evalutate_size] # 255癌 254 非癌 0 未知
                    
        #             if 1 in label_crop and label_crop.sum()>= evalutate_size*evalutate_size*0.75: #128*128*1*0.75
        #                 pred_roc_test.append(heatmap_crop)
        #                 label_roc_test.append(1)   
                    
        #             elif float(254/255) in label_crop :#and 0 not in label_crop:
        #                 pred_roc_test.append(heatmap_crop)
        #                 label_roc_test.append(0)

        sm=slice_image(sm,patch_size=evalutate_size).flatten(1)
        real_label=slice_image(real_label,patch_size=evalutate_size).flatten(1)
        is_zero_row = (real_label == 0).all(dim=1)
        
        non_zero_indices = ~is_zero_row
        
        real_label = real_label[non_zero_indices, :]
        sm = sm[non_zero_indices, :]
        sm = torch.mean(sm,dim=1).detach().cpu()
        real_label2 = torch.sum(real_label,dim=1)

        for kk in range(0,sm.shape[0]):
            if 1 in real_label[kk,:] and real_label2[kk]>= evalutate_size*evalutate_size*0.75: #128*128*1*0.75
                pred_roc_test.append(sm[kk])
                label_roc_test.append(1)   
                    
            elif float(254/255) in real_label[kk] :#and 0 not in label_crop:
                pred_roc_test.append(sm[kk])
                label_roc_test.append(0)

    


        if (i + 1) % 640 == 0:
            dist.barrier()
            label_roc_np_test = np.array(label_roc_test)
            pred_roc_np_test = np.array(pred_roc_test)

            cutoff_test = roc.roc(label_roc_np_test,pred_roc_np_test,'test',f'{name}',log_dir) 
            mask_test = torch.tensor((pred_roc_np_test > cutoff_test).astype(int)) * torch.ones_like(torch.from_numpy(pred_roc_np_test))
            dsc_test, iou_test = Dice(mask_test, torch.from_numpy(pred_roc_np_test))
            if args.world_size !=1:
                dsc_test = dsc_test.unsqueeze(0).unsqueeze(0).cuda()
                iou_test = iou_test.unsqueeze(0).unsqueeze(0).cuda()
        
                dsc_test = distributed_concat(dsc_test, args.world_size)
                iou_test = distributed_concat(iou_test, args.world_size)

                dsc_test = torch.mean(dsc_test)
                iou_test = torch.mean(iou_test)
            if rank == 0: 
                print(f"dice:{round(dsc_test.item(), 4)},iou:{round(iou_test.item(), 4)}")

        dist.barrier()
    
       
    label_roc_np_test = np.array(label_roc_test)
    pred_roc_np_test = np.array(pred_roc_test)
    cutoff_test = roc.roc(label_roc_np_test,pred_roc_np_test,f'test',f'{name}',log_dir) 
    mask_test = torch.tensor((pred_roc_np_test > cutoff_test).astype(int)) * torch.ones_like(torch.from_numpy(pred_roc_np_test))
    dsc_test, iou_test = Dice(mask_test, torch.from_numpy(pred_roc_np_test))
    if args.world_size !=1:
        dsc_test = dsc_test.unsqueeze(0).unsqueeze(0).cuda()
        iou_test = iou_test.unsqueeze(0).unsqueeze(0).cuda()

        dsc_test = distributed_concat(dsc_test, args.world_size)
        iou_test = distributed_concat(iou_test, args.world_size)

        dsc_test = torch.mean(dsc_test)
        iou_test = torch.mean(iou_test)
    
    if rank == 0:

        result_record = f'Test:  time:{datetime.datetime.now()},dice:{round(dsc_test.item(), 4)},iou:{round(iou_test.item(), 4)}\n' 
                       
        with open(f'{log_dir}/logging/{name}.txt', mode='a') as f:
            f.write(result_record + '\n')

    dist.destroy_process_group()