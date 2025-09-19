import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import os
from model.transocr import Transformer
from utils import get_data_package, converter, tensor2str, get_alphabet
import zhconv

import numpy as np

parser = argparse.ArgumentParser(description='')
parser.add_argument('--exp_name', type=str, default='test', help='')
parser.add_argument('--batch_size', type=int, default=32, help='')
parser.add_argument('--lr', type=float, default=1.0, help='')
parser.add_argument('--epoch', type=int, default=1000, help='')
parser.add_argument('--radical', action='store_true', default=False)
parser.add_argument('--test', action='store_true', default=False)
parser.add_argument('--resume', type=str, default='', help='')
parser.add_argument('--train_dataset', type=str, default='', help='')
parser.add_argument('--test_dataset', type=str, default='', help='')
parser.add_argument('--imageH', type=int, default=32, help='')
parser.add_argument('--imageW', type=int, default=256, help='')
parser.add_argument('--coeff', type=float, default=1.0, help='')
parser.add_argument('--alpha_path', type=str, default='./data/benchmark.txt', help='')
parser.add_argument('--alpha_path_radical', type=str, default='./data/radicals.txt', help='')
parser.add_argument('--decompose_path', type=str, default='./data/decompose.txt', help='')
parser.add_argument('--language', type=str, default='zh', help='')
parser.add_argument('--patience', type=int, default=5, help='Number of epochs to wait for improvement before stopping training')

# Do not use the following arguments if you have your own datasets
# Only use them if your are using the origial lmdb dataset given in the repo
# https://github.com/FudanVI/benchmarking-chinese-text-recognition/tree/main
parser.add_argument('--extract_single_char', action='store_true', default=False, help='Extract single characters from label if there are more than one character')
parser.add_argument('--img_from_lbl', action='store_true', default=False, help='Generate image with specific font from label')
parser.add_argument('--revealed_part', type=str, choices=['upper', 'lower', 'full', 'random'], default='full', help='Part of the character to be revealed')
parser.add_argument('--part_labeled_img', action='store_true', default=False, help='Use the image and a label indicating the revealed part of the character as ipput to the model')


args = parser.parse_args()
print(args)

alphabet = get_alphabet(args, 'char')
print('alphabet:',alphabet)

model = Transformer(args).cuda()
model = nn.DataParallel(model)
train_loader, test_loader = get_data_package(args)
optimizer = optim.Adadelta(model.parameters(), lr=args.lr, rho=0.9, weight_decay=1e-4)
criterion = torch.nn.CrossEntropyLoss().cuda()
best_acc = -1
patience = args.patience  # how many epochs to wait for improvement
epochs_no_improve = 0  # number of epochs with no improvement
early_stop = False  # flag to indicate early stopping

if args.resume.strip() != '':
    model.load_state_dict(torch.load(args.resume))
    print('loading pretrained model！！！')

def train(epoch, iteration, image, length, text_input, part_label, text_gt, length_radical, radical_input, radical_gt):
    model.train()
    optimizer.zero_grad()

    # Ensure inputs are on the same device as the model
    device = next(model.parameters()).device
    image = image.to(device)
    length = length.to(device)
    text_input = text_input.to(device)
    if part_label is not None: part_label = part_label.to(device)
    text_gt = text_gt.to(device)
    if args.radical:
        length_radical = length_radical.to(device)
        radical_input = radical_input.to(device)
        radical_gt = radical_gt.to(device)

    result = model(image, length, text_input, part_label,length_radical, radical_input)

    text_pred = result['pred']
    loss_char = criterion(text_pred, text_gt)
    if args.radical:
        radical_pred = result['radical_pred']
        loss_radical = criterion(radical_pred, radical_gt)
        loss = loss_char + args.coeff * loss_radical
        print(
            'epoch : {} | iter : {}/{} | loss : {} | char : {} | radical : {} '.format(epoch, iteration, len(train_loader), loss, loss_char, loss_radical))
    else:
        loss = loss_char
        print('epoch : {} | iter : {}/{} | loss : {}'.format(epoch, iteration, len(train_loader), loss))
    loss.backward()
    optimizer.step()

test_time = 0
@torch.no_grad()
def test(epoch):

    torch.cuda.empty_cache()
    global test_time
    test_time += 1
    torch.save(model.state_dict(), './history/{}/model.pth'.format(args.exp_name))
    result_file = open('./history/{}/result_file_test_{}.txt'.format(args.exp_name, test_time), 'w+', encoding='utf-8')
    result_file.write('ID | Part | Prediction | GroundTruth | Correct | Prob | MeanEntropySent\n')


    print("Start Eval!")
    if epoch == -1: 
        print('{} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {}'.format(
        "ID", "Part", "Prediction", "GroundTruth", "Correct", "Prob", "Acc", "MeanEntropyChar", "MeanEntropySent", "LastTokenEntropy", "LastChar", "Top-5"))

    device = next(model.parameters()).device 
    model.eval()
    dataloader = iter(test_loader)
    test_loader_len = len(test_loader)

    correct = 0
    total = 0

    all_total_entropies = []
    all_ave_entropies = []

    for iteration in range(test_loader_len):
        # data = dataloader.next()
        data = next(dataloader) 
        if args.part_labeled_img:
            image, part_label, label, _ = data
        else:
            image, label, _ = data
            part_label = None
        image = torch.nn.functional.interpolate(image, size=(args.imageH, args.imageW)).to(device)  
        length, text_input, text_gt, length_radical, radical_input, radical_gt, string_label = converter(label, args)

        length = length.to(device)
        text_input = text_input.to(device)
        text_gt = text_gt.to(device)
        if args.radical:
            length_radical = length_radical.to(device)
            radical_input = radical_input.to(device)
            radical_gt = radical_gt.to(device)
        if part_label is not None: part_label = part_label.to(device)

        max_length = max(length)
        batch = image.shape[0]
        pred = torch.zeros(batch,1).long().cuda()
        image_features = None
        prob = torch.zeros(batch, max_length).float().to(device)

        entropies = []
        for i in range(max_length):
            length_tmp = torch.zeros(batch).long().to(device) + i + 1
            result = model(image, length_tmp, pred, part_label, conv_feature=image_features, test=True)
            prediction = result['pred']  # shape: [B, T, C]
            probs = torch.softmax(prediction, dim=2)  # softmax over the alphabet dimension

            # Get current prediction
            now_pred = torch.max(probs, 2)[1]
            prob[:, i] = torch.max(probs, 2)[0][:, -1]
            pred = torch.cat((pred, now_pred[:, -1].view(-1, 1)), 1)
            image_features = result['conv']

            # Compute entropy at this step
            log_probs = torch.log(probs + 1e-8)
            step_entropy = -torch.sum(probs * log_probs, dim=2)  # shape: [B, T]
            entropies.append(step_entropy[:, -1])  # only last token per step


        entropies = torch.stack(entropies, dim=1)  # [B, max_length]
        eos_token_id = len(alphabet) - 1
        ave_entropy_per_sample = torch.zeros(batch, device=entropies.device)
        total_entropy_per_sample = torch.zeros(batch, device=entropies.device)
        last_entropy_before_eos = torch.zeros(batch, device=entropies.device)
        eos_lengths = []
        top5_chars_before_eos = []
        top5_probs_before_eos = []  
        for i in range(batch):
            eos_pos = None
            for j in range(1, max_length):  # skip BOS
                if pred[i][j] == eos_token_id:
                    eos_pos = j
                    break
            if eos_pos is None:
                eos_pos = max_length - 1
            valid_len = eos_pos
            eos_lengths.append(valid_len)

            if valid_len > 0:
                ave_entropy_per_sample[i] = entropies[i, :valid_len].mean()
                total_entropy_per_sample[i] = entropies[i, :valid_len].sum()
                last_entropy_before_eos[i] = entropies[i, valid_len - 2]
                top5_probs, top5_indices = torch.topk(probs[i, valid_len - 2, :], k=5, dim=-1)
                
                chars = [
                    alphabet[c.item()] if c.item() < len(alphabet) else '<EOS>'
                    for c in top5_indices
                ]
                scores = [round(p.item(), 4) for p in top5_probs]

                top5_chars_before_eos.append(chars)
                top5_probs_before_eos.append(scores)
            else:
                ave_entropy_per_sample[i] = 0.0
                total_entropy_per_sample[i] = 0.0
                last_entropy_before_eos[i] = 0.0
        # Store for dataset-level summary
        all_ave_entropies.extend(ave_entropy_per_sample.detach().cpu().tolist())
        all_total_entropies.extend(total_entropy_per_sample.detach().cpu().tolist())



        text_gt_list = []
        start = 0
        for i in length:
            text_gt_list.append(text_gt[start: start + i])
            start += i

        text_pred_list = []
        text_prob_list = []
        for i in range(batch):
            now_pred = []
            for j in range(max_length):
                if pred[i][j] != len(alphabet) - 1:
                    now_pred.append(pred[i][j])
                else:
                    break
            text_pred_list.append(torch.Tensor(now_pred)[1:].long().cuda())

            overall_prob = 1.0
            for j in range(len(now_pred)):
                overall_prob *= prob[i][j]
            text_prob_list.append(overall_prob)


        start = 0
        for i in range(batch):
            state = False
            if args.language == 'zh':
                pred = zhconv.convert(tensor2str(text_pred_list[i], args),'zh-cn')
                gt = zhconv.convert(tensor2str(text_gt_list[i], args), 'zh-cn')
            else:
                pred = tensor2str(text_pred_list[i], args)
                gt = tensor2str(text_gt_list[i], args)

            if pred == gt:
                correct += 1
                state = True
            start += i
            total += 1

            part = 'n' # for none
            if part_label is not None:
                part = {0: 'f', 1: 'u', 2: 'l'}[part_label[i].item()]
            
            if epoch == -1:
                if 0 <= eos_lengths[i] - 2 <= len(pred) - 1:
                    last_pred_char = pred[eos_lengths[i] - 2]
                else:
                    last_pred_char = '?'
                
                print('{} | {} | {} | {} | {} | {} | {} | {:.4f} | {:.4f} | {:.4f} | {} | {}'.format(
                                            total, part, pred, gt, state, 
                                            text_prob_list[i], correct / total, 
                                            ave_entropy_per_sample[i].item(), 
                                            total_entropy_per_sample[i].item(), 
                                            last_entropy_before_eos[i].item(), 
                                            last_pred_char,
                                            ', '.join([f'{c}:{p}' for c, p in zip(top5_chars_before_eos[i], top5_probs_before_eos[i])])
                                            ))
            else:
                print('{} | {} | {} | {} | {} | {} | {} | entropy: {:.4f} | entropy last token: {:.4f}'.format(total, part, pred, gt, state, 
                                    text_prob_list[i], correct / total, total_entropy_per_sample[i].item(), last_entropy_before_eos[i].item()))
            result_file.write(
                '{} | {} | {} | {} | {} | {} | {:.4f}\n'.format(total, part, pred, gt, state, text_prob_list[i], total_entropy_per_sample[i].item()))

    mean_ave_entropy = sum(all_ave_entropies) / len(all_ave_entropies)
    mean_total_entropy = sum(all_total_entropies) / len(all_total_entropies)
    print(f"Mean entropy of char/letter over test set: {mean_ave_entropy:.4f}")
    print(f"Mean entropy of word/sent over test set: {mean_total_entropy:.4f}")
    print("ACC : {}".format(correct/total))
    global best_acc
    global epochs_no_improve
    global early_stop

    if correct/total > best_acc:
        best_acc = correct / total
        torch.save(model.state_dict(), './history/{}/best_model.pth'.format(args.exp_name))
        epochs_no_improve = 0
    else:
        epochs_no_improve += 1
        print(f"No improvement in {epochs_no_improve} epoch(s).")


    f = open('./history/{}/record.txt'.format(args.exp_name),'a+',encoding='utf-8')
    f.write("Epoch : {} | ACC : {} | MeanEntropyChar: {} | MeanEntropySent: {}\n".format(epoch, correct/total, mean_ave_entropy, mean_total_entropy))
    f.close()

    if epochs_no_improve >= patience:
        print("Early stopping triggered!")
        early_stop = True

if __name__ == '__main__':
    print('-------------')
    if not os.path.isdir('./history/{}'.format(args.exp_name)):
        os.mkdir('./history/{}'.format(args.exp_name))
    if args.test:
        test(-1)
        exit(0)

    for epoch in range(args.epoch):
        torch.save(model.state_dict(), './history/{}/model.pth'.format(args.exp_name))
        dataloader = iter(train_loader)
        train_loader_len = len(train_loader)
        print('length of training datasets:', train_loader_len)
        for iteration in range(train_loader_len):
            # data = dataloader.next()
            data = next(dataloader)
            if args.part_labeled_img:
                image, part_label, label, _ = data
            else:
                image, label, _ = data
                part_label = None
            image = torch.nn.functional.interpolate(image, size=(args.imageH, args.imageW))

            length, text_input, xtext_gt, length_radical, radical_input, radical_gt, string_label = converter(label, args)
            train(epoch, iteration, image, length, text_input, part_label, xtext_gt, length_radical, radical_input, radical_gt)

        test(epoch)

        # early stopping check
        if early_stop:
            print("Stopping training early at epoch:", epoch)
            break