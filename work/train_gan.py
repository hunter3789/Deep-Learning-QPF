import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.utils.tensorboard as tb

from load_dataset_gan import load_data
from metrics_regressor import ContingencyMetric
from models_gan import RegressorLoss, CrossEntropyLoss, load_model

def train(
    exp_dir: str = "logs",
    model_name: str = "regressor",
    num_epoch: int = 50,
    lr: float = 1e-4,
    batch_size: int = 32,
    seed: int = 2024,
    mode: str = 'scratch',
    **kwargs,
):
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        print("CUDA not available, using CPU")
        device = torch.device("cpu")

    print(device)

    # set random seed so each run is deterministic
    torch.manual_seed(seed)
    np.random.seed(seed)

    # directory with timestamp to save tensorboard logs and model checkpoints
    log_dir = Path(exp_dir) / f"{model_name}_{datetime.now().strftime('%m%d_%H%M%S')}"
    logger = tb.SummaryWriter(log_dir)

    if (mode == "scratch"):
        model_gen = load_model('regressor', **kwargs)
        model_dis = load_model('discriminator', **kwargs)
    elif (mode == "load"):
        model_gen = load_model('regressor', with_weights=True)
        model_dis = load_model('discriminator', with_weights=True)
    else:
        print("input mode error")
        exit()

    model_gen = model_gen.to(device)
    model_dis = model_dis.to(device)

    mean_std_file = '../dataset/trainset_ssrd_agg_mean_std.npz'
    mean, std = np.load(mean_std_file)['mean'], np.load(mean_std_file)['std']

    label_mean_std_file = '../dataset/trainset_label_agg_mean_std.npz'
    label_mean, label_std = np.load(label_mean_std_file)['mean'], np.load(label_mean_std_file)['std']

    topo_file = '../dataset/topo_norm.npz'
    topo = np.load(topo_file)['topo']

    train_data = load_data(mean=mean, std=std, topo=topo, label_mean=label_mean, label_std=label_std, dataset_path="../dataset/train_ssrd", shuffle=True, batch_size=batch_size, num_workers=0, transform_pipeline="default")
    val_data = load_data(mean=mean, std=std, topo=topo, label_mean=label_mean, label_std=label_std, dataset_path="../dataset/val_ssrd", shuffle=True, batch_size=batch_size, num_workers=0, transform_pipeline="default")

    # create loss function and optimizer
    loss_func_dis = CrossEntropyLoss()
    loss_func_gen = RegressorLoss()
    optimizer_gen = torch.optim.AdamW(model_gen.parameters(), lr=lr, weight_decay=0.01)
    optimizer_dis = torch.optim.AdamW(model_dis.parameters(), lr=lr, weight_decay=0.01)

    constant = 0.1

    thresh = np.array([0.1, 0.5, 1, 5, 10, 15, 20, 25, 30, 40, 50])
    thresh_transformed = (np.log10(thresh + constant) - label_mean) / label_std

    cm = []
    cm_step = []

    for th in thresh_transformed:
        cm.append(ContingencyMetric(thresh = th))
        cm_step.append(ContingencyMetric(thresh = th))

    global_step = 0
    global_val_step = 0

    # training loop
    for epoch in range(num_epoch):
        losses = []        
        epoch_train_score = []
        data_num = 0

        model_dis.train()
        model_gen.train()
        for data in train_data:
            img, label, mask = data['image'], data['label'], data['mask']
            img, label, mask = img.to(device), label.to(device), mask.to(device)
            label = label.type(torch.float32)

            # Train Discriminator
            optimizer_dis.zero_grad()
            pred = model_gen(img).squeeze()
            
            real_dis = model_dis(label[:,None,:,:], img)
            fake_dis = model_dis(pred[:,None,:,:] * mask[:,None,:,:], img)

            dims = (fake_dis.size(0), fake_dis.size(2), fake_dis.size(3))

            real_loss = loss_func_dis(real_dis, torch.ones(*dims, device=device, dtype=torch.long))
            fake_loss = loss_func_dis(fake_dis, torch.zeros(*dims, device=device, dtype=torch.long))
            loss_dis = 0.5 * (real_loss + fake_loss)
            loss_dis.backward()
            optimizer_dis.step()

            # Train Generator
            optimizer_gen.zero_grad()
            pred = model_gen(img).squeeze()
            fake_dis = model_dis(pred[:,None,:,:] * mask[:,None,:,:], img)

            adv_loss = loss_func_dis(fake_dis, torch.ones(*dims, device=device, dtype=torch.long))
            pix_loss = loss_func_gen(pred, label, mask)

            loss_gen = adv_loss + 100 * pix_loss
            loss_gen.backward()
            optimizer_gen.step()

            loss = loss_gen.detach().cpu().numpy()
            losses.append(loss)

            logger.add_scalar('train_step_loss', loss, global_step)
            print(f"Global step {global_step + 1:10d}: ")
            for i in range(len(cm)):
                cm[i].add(pred, label, mask)
                cm_step[i].add(pred, label, mask)
                step_score = cm_step[i].compute()            
                cm_step[i].reset()           

                print("thresh={thresh}".format(thresh=thresh[i]))
                print("h={h:8d}   f={f:8d}   m={m:8d}   c={c:8d} ".format(h=int(step_score["H"]), 
                    f=int(step_score["F"]), m=int(step_score["M"]), c=int(step_score["C"])))
                print("csi={csi:8.2f} pod={pod:8.2f} far={far:8.2f} fbias={fbias:8.2f}".format(csi=step_score["CSI"]*100, 
                    pod=step_score["POD"]*100, far=step_score["FAR"]*100, fbias=step_score["FBIAS"]))

                logger.add_scalar('train_step_h_{th}'.format(th=thresh[i]), step_score["H"], global_step)
                logger.add_scalar('train_step_f_{th}'.format(th=thresh[i]), step_score["F"], global_step)
                logger.add_scalar('train_step_m_{th}'.format(th=thresh[i]), step_score["M"], global_step)
                logger.add_scalar('train_step_c_{th}'.format(th=thresh[i]), step_score["C"], global_step)
                logger.add_scalar('train_step_csi_{th}'.format(th=thresh[i]), step_score["CSI"], global_step)
                logger.add_scalar('train_step_pod_{th}'.format(th=thresh[i]), step_score["POD"], global_step)
                logger.add_scalar('train_step_far_{th}'.format(th=thresh[i]), step_score["FAR"], global_step)
                logger.add_scalar('train_step_fbias_{th}'.format(th=thresh[i]), step_score["FBIAS"], global_step)

            global_step += 1

            data_num += 1
            if data_num % 10 == 0:
                print(
                    f"{data_num}/{len(train_data)} processed"
                    f", {data_num/len(train_data)*100:.2f}%"
                )

        epoch_train_loss = np.mean(losses)
        print(f"train_loss={epoch_train_loss:.4f}")

        for i in range(len(cm)):
            score = cm[i].compute()
            epoch_train_score.append(score)
            cm[i].reset()            

        # save a copy of model weights in the log directory
        torch.save(model_gen.state_dict(), log_dir / f"regressor_epoch_{epoch}.th")
        torch.save(model_dis.state_dict(), log_dir / f"discriminator_epoch_{epoch}.th")

        torch.save(optimizer_gen.state_dict(), log_dir / f"regressor_optim_epoch_{epoch}.th")
        torch.save(optimizer_dis.state_dict(), log_dir / f"discriminator_optim_epoch_{epoch}.th")

        # torch.inference_mode calls model.eval() and disables gradient computation
        model_gen.eval()
        model_dis.eval()
        with torch.inference_mode():
            losses = []
            epoch_val_score = []
            data_num = 0
            for data in val_data:
                img, label, mask = data['image'], data['label'], data['mask']
                img, label, mask = img.to(device), label.to(device), mask.to(device)
                label = label.type(torch.float32)

                pred = model_gen(img).squeeze()

                print(f"Val step {global_val_step + 1:10d}: ")
                for i in range(len(cm)):
                    cm[i].add(pred, label, mask)
                    cm_step[i].add(pred, label, mask)
                    step_score = cm_step[i].compute()            
                    cm_step[i].reset()   

                    print("thresh={thresh}".format(thresh=thresh[i]))
                    print("h={h:8d}   f={f:8d}   m={m:8d}   c={c:8d} ".format(h=int(step_score["H"]), 
                        f=int(step_score["F"]), m=int(step_score["M"]), c=int(step_score["C"])))
                    print("csi={csi:8.2f} pod={pod:8.2f} far={far:8.2f} fbias={fbias:8.2f}".format(csi=step_score["CSI"]*100, 
                        pod=step_score["POD"]*100, far=step_score["FAR"]*100, fbias=step_score["FBIAS"]))

                    logger.add_scalar('val_step_h_{th}'.format(th=thresh[i]), step_score["H"], global_val_step)
                    logger.add_scalar('val_step_f_{th}'.format(th=thresh[i]), step_score["F"], global_val_step)
                    logger.add_scalar('val_step_m_{th}'.format(th=thresh[i]), step_score["M"], global_val_step)
                    logger.add_scalar('val_step_c_{th}'.format(th=thresh[i]), step_score["C"], global_val_step)
                    logger.add_scalar('val_step_csi_{th}'.format(th=thresh[i]), step_score["CSI"], global_val_step)
                    logger.add_scalar('val_step_pod_{th}'.format(th=thresh[i]), step_score["POD"], global_val_step)
                    logger.add_scalar('val_step_far_{th}'.format(th=thresh[i]), step_score["FAR"], global_val_step)
                    logger.add_scalar('val_step_fbias_{th}'.format(th=thresh[i]), step_score["FBIAS"], global_val_step)

                global_val_step += 1

                data_num += 1
                if data_num % 10 == 0:
                    print(
                        f"{data_num}/{len(val_data)} processed"
                        f", {data_num/len(val_data)*100:.2f}%"
                    )                     

        # log average train and val accuracy to tensorboard
        logger.add_scalar('train_loss', epoch_train_loss, global_step)

        print(f"Epoch {epoch + 1:2d} / {num_epoch:2d}: ")
        for i in range(len(cm)):
            score = cm[i].compute()
            epoch_val_score.append(score)
            cm[i].reset()                        

            logger.add_scalar('train_h_{th}'.format(th=thresh[i]), epoch_train_score[i]["H"], global_step)
            logger.add_scalar('train_f_{th}'.format(th=thresh[i]), epoch_train_score[i]["F"], global_step)
            logger.add_scalar('train_m_{th}'.format(th=thresh[i]), epoch_train_score[i]["M"], global_step)
            logger.add_scalar('train_c_{th}'.format(th=thresh[i]), epoch_train_score[i]["C"], global_step)
            logger.add_scalar('train_csi_{th}'.format(th=thresh[i]), epoch_train_score[i]["CSI"], global_step)
            logger.add_scalar('train_pod_{th}'.format(th=thresh[i]), epoch_train_score[i]["POD"], global_step)
            logger.add_scalar('train_far_{th}'.format(th=thresh[i]), epoch_train_score[i]["FAR"], global_step)
            logger.add_scalar('train_fbias_{th}'.format(th=thresh[i]), epoch_train_score[i]["FBIAS"], global_step)

            logger.add_scalar('val_h_{th}'.format(th=thresh[i]), epoch_val_score[i]["H"], global_step)
            logger.add_scalar('val_f_{th}'.format(th=thresh[i]), epoch_val_score[i]["F"], global_step)
            logger.add_scalar('val_m_{th}'.format(th=thresh[i]), epoch_val_score[i]["M"], global_step)
            logger.add_scalar('val_c_{th}'.format(th=thresh[i]), epoch_val_score[i]["C"], global_step)
            logger.add_scalar('val_csi_{th}'.format(th=thresh[i]), epoch_val_score[i]["CSI"], global_step)
            logger.add_scalar('val_pod_{th}'.format(th=thresh[i]), epoch_val_score[i]["POD"], global_step)
            logger.add_scalar('val_far_{th}'.format(th=thresh[i]), epoch_val_score[i]["FAR"], global_step)
            logger.add_scalar('val_fbias_{th}'.format(th=thresh[i]), epoch_val_score[i]["FBIAS"], global_step)

            print("thresh={thresh}".format(thresh=thresh[i]))
            print("train_h={epoch_train_h:8d}   train_f={epoch_train_f:8d}   train_m={epoch_train_m:8d}   train_c={epoch_train_c:8d} ".format(epoch_train_h=int(epoch_train_score[i]["H"]), 
                epoch_train_f=int(epoch_train_score[i]["F"]), epoch_train_m=int(epoch_train_score[i]["M"]), epoch_train_c=int(epoch_train_score[i]["C"])))
            print("train_csi={epoch_train_csi:8.2f} train_pod={epoch_train_pod:8.2f} train_far={epoch_train_far:8.2f} train_fbias={epoch_train_fbias:8.2f}".format(epoch_train_csi=epoch_train_score[i]["CSI"]*100, 
                epoch_train_pod=epoch_train_score[i]["POD"]*100, epoch_train_far=epoch_train_score[i]["FAR"]*100, epoch_train_fbias=epoch_train_score[i]["FBIAS"]))

            print("  val_h={epoch_val_h:8d}     val_f={epoch_val_f:8d}     val_m={epoch_val_m:8d}     val_c={epoch_val_c:8d}".format(epoch_val_h=int(epoch_val_score[i]["H"]), 
                epoch_val_f=int(epoch_val_score[i]["F"]), epoch_val_m=int(epoch_val_score[i]["M"]), epoch_val_c=int(epoch_val_score[i]["C"])))
            print("  val_csi={epoch_val_csi:8.2f}   val_pod={epoch_val_pod:8.2f}   val_far={epoch_val_far:8.2f}   val_fbias={epoch_val_fbias:8.2f}".format(epoch_val_csi=epoch_val_score[i]["CSI"]*100, 
                epoch_val_pod=epoch_val_score[i]["POD"]*100, epoch_val_far=epoch_val_score[i]["FAR"]*100, epoch_val_fbias=epoch_val_score[i]["FBIAS"]))
        
    # save a copy of model weights in the log directory
    torch.save(model_gen.state_dict(), log_dir / "regressor.th")
    torch.save(model_dis.state_dict(), log_dir / "discriminator.th")

    torch.save(optimizer_gen.state_dict(), log_dir / "regressor_optim.th")
    torch.save(optimizer_dis.state_dict(), log_dir / "discriminator_optim.th")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--exp_dir", type=str, default="gan_logs")
    parser.add_argument("--model_name", type=str, default="gan")
    parser.add_argument("--num_epoch", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--mode", type=str, default="scratch")    

    # pass all arguments to train
    train(**vars(parser.parse_args()))
