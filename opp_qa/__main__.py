# from opp_qa.preprocessing import *

# data_name = 's1234_500_400_balanced.pkl'
# context_data = 's1234_500_400_context.pkl'
# save_name = 'opp_sim8'

# print('Source data: ', data_name)
# print('context data: ', context_data)
# print('Processed data name: ', save_name)

# preprocess_data(data_name,
#             save_name,
#             data_folder = 'sqa_data/',
#             create_ebd = False, # already generated
#             context_name = context_data,
#             source_data = 'opp',
#             win_len = 500
#                 )


from pathlib import Path
from einops import rearrange
import torch
from opp_qa.data import OppQADataModule
from opp_qa.deepsqa_baselines import Word2VecModel, SensoryModel, BaselineSQA
from opp_qa.embedding import load as load_embedding
from opp_qa.mac import MACNetwork
from torchmetrics import Accuracy
import matplotlib.pyplot as plt
from argparse import ArgumentParser
import os 
from torchinfo import summary

answer_ids= [
                    "no",
                    "yes",
                    "0",
                    "1",
                    "2",
                    "open the front door",
                    "clean the table",
                    "open the third drawer",
                    "close the front door",
                    "toggle the switch",
                    "close the third drawer",
                    "open the second drawer",
                    "close the first drawer",
                    "close the second drawer",
                    "open the first drawer",
                    "close the back door",
                    "open the back door",
                    "close the fridge",
                    "open the fridge",
                    "close the dishwasher",
                    "drink from the cup",
                    "open the dishwasher",
                    "3",
                    "6",
                    "4",
                    "7",
                ]


parser = ArgumentParser()
parser.add_argument("--task", type=str, default="binary")
parser.add_argument("--short", type=bool, default=True)
parser.add_argument("--kind", type=str, default="cnn_lstm_mul", choices=["cnn_lstm_mul", "cnn_lstm_cat", "deepsqa", "lstm", "cnn", "mac","mac-train", "mac-scratch"])
parser.add_argument("--gpu", type=int, default=0)


args = parser.parse_args()
short = args.short
task = args.task
kind = args.kind
match task:
    case "binary":
        num_classes = 2
    case "count":
        num_classes = 4
    case "multi":
        len(answer_ids)
    case _:
        raise ValueError("Task not recognized")
# num_classes = 2 if task == "binary" else len(answer_ids)
os.environ['CUDA_VISIBLE_DEVICES'] = f"{args.gpu}"

print(f"Starting training with:\n------\nTask:{task}\nShort:{'True' if short else 'False'}\nKind:{kind}\n------\n")


datamodule = OppQADataModule(batch_size=64, task=task, short=short)
epochs = 50

datamodule.prepare_data()
datamodule.setup("fit")


emb = load_embedding()

num_words = 400001
embedding_dim = 300

num_hidden_lstm = 128
output_dim = 128
dropout_rate = 0.5
num_feat_map = 64

batch = next(iter(datamodule.train_dataloader()))

text_len = batch["text_idx"].shape[1]
sen_dim = 77
sen_channel = 1

# w2vec = Word2VecModel(
#     emb,
#     num_words=num_words,
#     embedding_dim=embedding_dim,
#     num_hidden_lstm=num_hidden_lstm,
#     output_dim=output_dim,
#     dropout_rate=dropout_rate,
# )

# sensory_model = SensoryModel(
#     sen_dim,
#     sen_channel,
#     num_feat_map,
#     num_hidden_lstm=num_hidden_lstm,
#     output_dim=output_dim,
#     dropout_rate=dropout_rate,
# )

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if "mac" in kind:
    dim = 512

    embed_train = "scratch" in kind or "train" in kind
    vocab_embed = "scratch" not in kind

    model = MACNetwork(emb,dim=dim,embed_hidden=300,vocabulary_embd=vocab_embed, embd_train=embed_train, max_step=12, self_attention=False, memory_gate=False,classes=num_classes,dropout=0.15)
else:
    model = BaselineSQA(emb,num_words,embedding_dim,num_hidden_lstm,output_dim,dropout_rate,sen_dim,sen_channel,num_feat_map,num_classes,model_type=kind)
model = model.to(device)

summary(model, [(1, 1, 77, batch["trajectory"].shape[-2]), (1,batch["text_idx"].shape[-1])], dtypes=[torch.float, torch.long])

optim = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
criterion = torch.nn.CrossEntropyLoss()

# out = w2vec(batch["text_idx"])
# traj = batch["trajectory"] # (batch, seq_len, sen_dim)
# traj = rearrange(traj, "b seq var -> b 1 var seq")
# out2 = sensory_model(traj)

train_acc_mt = Accuracy(num_classes=num_classes, task="multiclass").to(device)
val_acc_mt = Accuracy(num_classes=num_classes, task="multiclass").to(device)
test_acc_mt = Accuracy(num_classes=num_classes, task="multiclass").to(device)

val_epoch_accs = []
train_epoch_accs = []
test_epoch_accs = []

train_epoch_losses = []
val_epoch_losses = []
test_epoch_losses = []

for epoch in range(epochs):
    print(f"------- Train {epoch} -------")
    losses_train = []
    model.train()
    for idx, batch in enumerate(datamodule.train_dataloader()):
        text_x = batch["text_idx"].to(device)
        traj = batch["trajectory"].to(device)
        traj = rearrange(traj, "b seq var -> b 1 var seq")
        label = batch["answer"].to(device)

        out = model(traj, text_x)

        loss = criterion(out, label)
        optim.zero_grad()
        loss.backward()
        optim.step()

        losses_train.append(loss.item())
        acc = train_acc_mt(out.argmax(1), label)


        if idx % 50 == 0:
            print(f"Step {idx}: Train/Loss: {losses_train[-1]}, Train/Acc: {acc.item()}")

    epoch_acc = train_acc_mt.compute().item()
    train_epoch_accs.append(epoch_acc)
    train_epoch_losses.append(sum(losses_train)/len(losses_train))
    print(f"Epoch: {epoch} Loss: {sum(losses_train)/len(losses_train)}, Acc: {epoch_acc}")

    print(f"------- VAL {epoch} -------")
    model.eval()
    with torch.no_grad():
        losses_val = []
        for idx, batch in enumerate(datamodule.val_dataloader()):
            text_x = batch["text_idx"].to(device)
            traj = batch["trajectory"].to(device)
            traj = rearrange(traj, "b seq var -> b 1 var seq")
            label = batch["answer"].to(device)

            out = model(traj, text_x)

            loss = criterion(out, label)
            losses_val.append(loss.item())

            acc = val_acc_mt(out.argmax(1), label)

            if idx % 50 == 0:
                print(f"Step {idx}: Val/Loss: {losses_val[-1]}, Val/Acc: {acc.item()}")

        val_epoch_acc = val_acc_mt.compute().item()
        val_epoch_accs.append(val_epoch_acc)
        val_epoch_losses.append(sum(losses_val)/len(losses_val))
        print(f"Epoch: {epoch} Val/Loss: {sum(losses_val)/len(losses_val)}, Val/Acc: {val_epoch_acc}")

    if epoch % 10 == 0 or epoch == epochs-1:
        torch.save(model.state_dict(), f"models/{task}_{kind}_{epoch}_val_acc_{val_epoch_acc}.pt")

    # print("------- Test -------")
    # model.eval()
    # with torch.no_grad():
    #     losses_test = []
    #     accs_test = []
    #     for idx, batch in enumerate(datamodule.test_dataloader()):
    #         text_x = batch["text_idx"].to(device)
    #         traj = batch["trajectory"].to(device)
    #         traj = rearrange(traj, "b seq var -> b 1 var seq")
    #         label = batch["answer"].to(device)

    #         out = model(traj, text_x)

    #         loss = criterion(out, label)
    #         losses_test.append(loss.item())

    #         test_acc_mt(out.argmax(1), label)

        
    #     test_acc = test_acc_mt.compute().item()
    #     test_epoch_accs.append(test_acc)
    #     test_loss = sum(losses_test)/len(losses_test)
    #     test_epoch_losses.append(test_loss)
    #     print(f"Epoch: {epoch} Test/Loss: {test_loss}, Test/Acc: {test_acc}")

model.eval()
with torch.no_grad():
    losses_test = []
    accs_test = []
    for idx, batch in enumerate(datamodule.test_dataloader()):
        text_x = batch["text_idx"].to(device)
        traj = batch["trajectory"].to(device)
        traj = rearrange(traj, "b seq var -> b 1 var seq")
        label = batch["answer"].to(device)

        out = model(traj, text_x)

        loss = criterion(out, label)
        losses_test.append(loss.item())

        test_acc_mt(out.argmax(1), label)
        

    
    test_acc = test_acc_mt.compute().item()
    test_loss = sum(losses_test)/len(losses_test)
    print(f"Test/Loss: {test_loss}, Test/Acc: {test_acc}")





# two figures for loss and acc (subfigure for train and val)
plt.subplots(1,2, figsize=(10,5))
plt.subplot(1,2,1)
plt.plot(train_epoch_losses, label="Train")
plt.plot(val_epoch_losses, label="Val")
# plt.plot(test_epoch_losses, label="Test")
plt.hlines(test_loss, 0, epochs, label="Test", color="red", linestyle="--")

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.title("Loss")

plt.subplot(1,2,2)
plt.plot(train_epoch_accs, label="Train")
plt.plot(val_epoch_accs, label="Val")
# plt.plot(test_epoch_accs, label="Test")
plt.hlines(test_acc, 0, epochs, label="Test", color="red", linestyle="--")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.title("Accuracy")


# main title for the figure
plt.suptitle(f"Training {kind} model for {task} task with {epochs} epochs")

res_path = Path("plots") / "mac" / ('short' if short else 'long')
res_path.mkdir(exist_ok=True)

res_path = res_path / f"result_{task}_{kind}_{epochs}_1e-4.png"
plt.savefig(res_path)
