import argparse
import os
import logging
import shutil
import time

from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from torch.nn import functional as F
import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from transformers import AdamW, T5Tokenizer, Adafactor
from model import modeling_t5
import transformers
from data_utils import EmotionDataset, build_co_term_idx, build_term_dict
from eval_utils import evaluate, extract_GoEmotion
from contrastive_loss import SupConLoss
from label_co_existence import get_co_existence, get_co_existence_tar

logger = logging.getLogger(__name__)


def init_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default='Triplets', type=str, help='select from GoEmotions, SemEvalEc and Triplets, Triplets_Restaurant')
    parser.add_argument("--do_train", action='store_true', default=True)
    parser.add_argument("--do_eval", action='store_true', default=True)

    parser.add_argument("--CLP", action='store_true', default=True) 
    parser.add_argument("--send_CLP", action="store_true", default=False)
    parser.add_argument("--unify_dict", action="store_true", default=True)
    parser.add_argument("--combine_all_co_occurs", action="store_true", default=True)
    parser.add_argument("--only_tar", action="store_true", default=False)
    parser.add_argument("--only_opi", action="store_true", default=False)
    parser.add_argument("--only_tar_opi", action="store_true", default=False)
    parser.add_argument("--all", action="store_true", default=True)
    parser.add_argument("--learning_rate", default=3e-5, type=float)
    parser.add_argument("--alpha", default=0.4, type=float)
    parser.add_argument("--temperature", default=0.07, type=float)
    parser.add_argument("--tar_limit", default=70, type=int)
    parser.add_argument("--opi_limit", default=70, type=int)
    parser.add_argument("--model_name_or_path", default='t5-base', type=str, help="Path of pre-trained model")
    parser.add_argument("--max_seq_length", default=512, type=int)
    parser.add_argument("--n_gpu", default=[0])
    parser.add_argument("--num_beams", default=3, type=int)
    parser.add_argument("--train_batch_size", default=16, type=int)
    parser.add_argument("--eval_batch_size", default=16, type=int)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
    parser.add_argument("--num_train_epochs", default=100, type=int)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument("--weight_decay", default=0.0, type=float)
    parser.add_argument("--adam_epsilon", default=1e-8, type=float)
    parser.add_argument("--warmup_steps", default=0.0, type=float)

    args = parser.parse_args()

    if not os.path.exists('./output'):
        os.mkdir('./output')

    args.device = args.n_gpu[0]

    return args


def get_dataset(tokenizer, data_type, args):
    if args.CLP:
        max_len = args.max_seq_length - label_size
    else:
        max_len = args.max_seq_length
    return EmotionDataset(tokenizer=tokenizer, data_dir=args.dataset, data_type=data_type, max_len=max_len, args=args)


def get_label_size():
    return label_size


def get_num_tar():
    return args.num_tar


def get_num_opi():
    return args.num_opi


def send_CLP():
    return args.send_CLP


class T5EmotionGeneration(pl.LightningModule):
    def __init__(self, hparams):
        super(T5EmotionGeneration, self).__init__()
        self.hyparams = hparams
        if args.CLP:
            self.model = modeling_t5.T5ForConditionalGeneration.from_pretrained(hparams.model_name_or_path)
        else:
            self.model = transformers.T5ForConditionalGeneration.from_pretrained(hparams.model_name_or_path)

        self.tokenizer = T5Tokenizer.from_pretrained("pre_trained_model/t5_base")
        self.alpha = torch.tensor(args.alpha)
        self.temperature = 0.07 if args.dataset == 'GoEmotions' else args.temperature
        self.loss_scl = SupConLoss(temperature=self.temperature)

    def forward(self, input_ids, attention_mask=None, labels=None, decoder_attention_mask=None, decoder_input_ids=None,
                past_key_values=None):
        if args.CLP:
            label_mask = torch.ones(input_ids.shape[0], label_size, dtype=torch.int).to(attention_mask.device)
            attention_mask = torch.cat((label_mask, attention_mask), dim=1)
        model_output = self.model(                          # self.CLP 为 True 的情况下, 各输入参数的具体解释见 .forward() 方法下的注释
            input_ids,                                      # 一个 batch 的输入句子(单词的编码), (16, 100)
            attention_mask=attention_mask,                  # 一个 batch 输入句子的 mask,      (16, 128)
            labels=labels,                                  # 一个 batch 的目标句子(单词的编码), (16, 100)
            decoder_attention_mask=decoder_attention_mask,  # 一个 batch 目标句子的 mask,      (16, 100)
            decoder_input_ids=decoder_input_ids,            # None
        )
        return model_output

    def _step(self, batch, validate=False):

        labels = batch["target_ids"]
        labels[labels[:, :] == self.tokenizer.pad_token_id] = -100

        outputs = self(
            input_ids=batch["source_ids"],                  # 一个 batch 的输入句子的每个单词的 id, (16, 100)
            attention_mask=batch["source_mask"],            # 一个 batch 的输入句子的掩码, (16, 100)
            labels=labels,                                  # 一个 batch 的目标句子的每个单词的 id, (16, 100)
            decoder_attention_mask=batch['target_mask'],    # 一个 batch 的目标句子的掩码
        )

        loss_gen = outputs[0]

        if validate or not args.CLP:
            return loss_gen
        else:
            normed_prompts = F.normalize(self.model.encoder.label, dim=1)
            loss_tar = self.loss_scl(normed_prompts, labels=batch["tar_idx"], weight=weight_tar)
            loss_opi = self.loss_scl(normed_prompts, labels=batch["opi_idx"], weight=weight_opi)
            loss_tar_opi = self.loss_scl(normed_prompts, labels=batch["tar_opi_idx"], weight=weight_tar_opi)

            loss = 0
            if args.only_tar:
                loss = self.alpha * loss_gen + (1 - self.alpha) * loss_tar
            elif args.only_opi:
                loss = self.alpha * loss_gen + (1 - self.alpha) * loss_opi
            elif args.only_tar_opi:
                loss = self.alpha * loss_gen + (1 - self.alpha) * (0.5*loss_tar + 0.5*loss_opi)
            elif args.all:
                loss = self.alpha * loss_gen + (1 - self.alpha) * (1/3*loss_tar + 1/3*loss_opi + 1/3*loss_tar_opi)
            return loss

    def training_step(self, batch, batch_idx):
        loss = self._step(batch)
        return {"loss": loss}

    def training_epoch_end(self, outputs):
        avg_train_loss = torch.stack([x["loss"] for x in outputs]).mean()
        self.log("avg_train_loss", avg_train_loss)

    def validation_step(self, batch, batch_idx):
        loss = self._step(batch, validate=True)
        return {"val_loss": loss}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        self.log("val_loss", avg_loss)
        print("avg_val_loss:", avg_loss)

    def configure_optimizers(self):
        model = self.model
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.hyparams.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]

        optimizer = Adafactor(
            optimizer_grouped_parameters,
            lr=self.hyparams.learning_rate,
            eps=(1e-30, 1e-3),
            clip_threshold=1.0,
            decay_rate=-0.8,
            beta1=None,
            weight_decay=0.0,
            relative_step=False,
            scale_parameter=False,
            warmup_init=False,
        )

        self.opt = optimizer

        return [optimizer]

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_idx, optimizer_closure,
                       on_tpu=False, using_native_amp=False, using_lbfgs=False):
        if self.trainer.global_step < self.hyparams.warmup_steps:
            lr_scale = float(self.trainer.global_step + 1) / self.hyparams.warmup_steps

        elif self.trainer.global_step < self.total_step:
            lr_scale = min(1.0, float(self.trainer.global_step + 1) / self.total_step)
        for pg in optimizer.param_groups:
            pg["lr"] = lr_scale * self.hyparams.learning_rate

        optimizer.step(closure=optimizer_closure)

    def train_dataloader(self):
        train_dataset = get_dataset(self.tokenizer, data_type='train', args=self.hyparams)
        dataloader = DataLoader(train_dataset, batch_size=self.hyparams.train_batch_size,
                                num_workers=4, shuffle=False)
        print(f"length of train_dataset: {len(train_dataset)}; length of dataloader: {len(dataloader)}")
        t_total = (
                (len(dataloader.dataset) // (self.hyparams.train_batch_size * max(1, len(self.hyparams.n_gpu))))
                // self.hyparams.gradient_accumulation_steps
                * float(self.hyparams.num_train_epochs)
        )

        self.total_step = t_total
        return dataloader

    def val_dataloader(self):
        val_dataset = get_dataset(self.tokenizer, data_type='dev', args=self.hyparams)
        data_loader = DataLoader(val_dataset, batch_size=self.hyparams.eval_batch_size, num_workers=4, shuffle=False)
        print(f"length of dev_dataset: {len(val_dataset)}; length of dataloader: {len(data_loader)}")
        return data_loader


args = init_args()
tar_co_occurs_idx, opi_co_occurs_idx, tar_opi_co_occur_idx, tar_terms, opi_terms, tar_and_opi_terms = None, None, None, None, None, None

if args.dataset == "Res-Im":
    tar_co_occurs_idx, opi_co_occurs_idx, tar_opi_co_occur_idx, tar_terms, opi_terms, tar_and_opi_terms = build_co_term_idx(
        f"data/Res-Im/train.tsv", args)
    # 1. tar_co_occurs_idx, 为一个 list, 长度为 1407, 其每个元素为一个样本中(同时)出现的 target 在字典中的索引(只有出现达到一定次数才可以进入到字典中). 如:
    #       [[0, 2, 12], [4, 1], [], [], ...]
    # 2. opi_co_occurs_idx, 为一个 list, 长度为 1407, 其每个元素为一个样本中(同时)出现的 opinion 在字典中的索引(只有出现达到一定次数才可以进入到字典中). 如:
    #       [[1, 32, 15], [20, 2], [26], [], ...]
    # 3. tar_opi_co_occurs_idx, 为一个 list, 长度为 1407, 其每个元素为一个样本中出现的 target-opinion 对在字典中的索引(target 和 opinion 必须都出现达到一定次数才可以进入到字典中). 如:
    #       [[(0, 82), (12, 65), (2, 51)], [(1, 52), (4, 70)], [], [], ...]
    # 4. tar_keys: target 字典的键, 取出现次数前 50 的 target. 如:
    #       ['laptop', 'chromebook', 'keyboard', 'computer', ...]
    # 5. opi_keys: opinion 字典的键, 取出现次数前 50 的 opinion. 如:
    #       ['great', 'good', 'nice', 'love', 'fast', ...]
    # 6. tar_opi_keys: tar_keys 和 opi_keys 的合并
    #
    label_list = tar_and_opi_terms
    label_size = len(label_list)

elif args.dataset == 'Lap-Im':
    tar_co_occurs_idx, opi_co_occurs_idx, tar_opi_co_occur_idx, tar_terms, opi_terms, tar_and_opi_terms = build_co_term_idx(
        f"data/Lap-Im/train.tsv", args)
    # 1. tar_co_occurs_idx, 为一个 list, 长度为 1407, 其每个元素为一个样本中(同时)出现的 target 在字典中的索引(只有出现达到一定次数才可以进入到字典中). 如:
    #       [[0, 2, 12], [4, 1], [], [], ...]
    # 2. opi_co_occurs_idx, 为一个 list, 长度为 1407, 其每个元素为一个样本中(同时)出现的 opinion 在字典中的索引(只有出现达到一定次数才可以进入到字典中). 如:
    #       [[1, 32, 15], [20, 2], [26], [], ...]
    # 3. tar_opi_co_occurs_idx, 为一个 list, 长度为 1407, 其每个元素为一个样本中出现的 target-opinion 对在字典中的索引(target 和 opinion 必须都出现达到一定次数才可以进入到字典中). 如:
    #       [[(0, 82), (12, 65), (2, 51)], [(1, 52), (4, 70)], [], [], ...]
    # 4. tar_keys: target 字典的键, 取出现次数前 50 的 target. 如:
    #       ['laptop', 'chromebook', 'keyboard', 'computer', ...]
    # 5. opi_keys: opinion 字典的键, 取出现次数前 50 的 opinion. 如:
    #       ['great', 'good', 'nice', 'love', 'fast', ...]
    # 6. tar_opi_keys: tar_keys 和 opi_keys 的合并
    #
    label_list = tar_and_opi_terms
    label_size = len(label_list)
else:
    raise NameError("No such dataset")


if __name__ == '__main__':
    if args.dataset == "Lap-Im" or args.dataset == "Res-Im":
        weight_tar = torch.from_numpy(get_co_existence_tar(args.dataset, label_size, tar_co_occurs_idx))
        weight_opi = torch.from_numpy(get_co_existence_tar(args.dataset, label_size, opi_co_occurs_idx))
        weight_tar_opi = torch.from_numpy(get_co_existence(args.dataset, label_size, tar_opi_co_occur_idx))
    else:
        weight = torch.from_numpy(get_co_existence(args.dataset, label_size))

    args.max_seq_length = 512 if args.dataset == "Lap-Im" or args.dataset == "Res-Im" else 128

    if "test_results" not in os.listdir("./"):
        os.mkdir("test_results")
    path_time_now = time.strftime("%Y%m%d-%H%M", time.localtime())
    with open(f"test_results/{path_time_now}.txt", "w", encoding="utf-8") as f:
        pass

    output_dir = f"./output/{args.dataset}/CLP-{str(args.CLP)}"
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    args.output_dir = output_dir
    print(f"args.learning_rate: {args.learning_rate}\nargs.alpha: {args.alpha}\nargs.temperature: {args.temperature}\nargs.CLP: {args.CLP}\nargs.output_dir: {args.output_dir}")

    if args.do_train:
        print(f"\nTrain on {args.dataset}\n")

        # 如果是训练，则删除之前已训练的模型
        if os.listdir(args.output_dir):
            shutil.rmtree(args.output_dir)

        seed_everything(args.seed)
        model = T5EmotionGeneration(hparams=args)
        tokenizer = model.tokenizer


        dataset = get_dataset(tokenizer=tokenizer, data_type='dev', args=args)
        # dataset 为一个 EmotionDataset() 类对象，其主要有三个属性:
        #   1. dataset.inputs:
        #           [BatchEncoding(), ...]，长度为 5426
        #      BatchEncoding() 类对象为一个 dict，有两个键:
        #           1) input_ids:
        #               tensor([[27, 7, 48, ...]]),  (1, 100)
        #           2) attention_mask:
        #               tensor([[ 1, 1,  1, ...]]),  (1, 100)
        #   2. dataset.targets 同 self.inputs
        #   3. dataset.label_idx:
        #           [tensor([0, ..., 0, 1, 1, ..., 0]), ...]

        data_sample = dataset[14]
        # dataset[14] 调用 EmotionDataset() 的 __getitem__() 方法，返回一个字典，包括一个样本的:
        #   source_ids:     输入句子的编码，如 tensor([1815, 40, 2461, ...])
        #   source_mask:    输入句子的掩码，如 tensor([   1,  1,    1, ...])
        #   target_ids:     目标句子的编码，如 tensor([  37, 68,  183, ...])
        #   target_mask:    目标句子的掩码，如 tensor([   1,  1,    1, ...])
        #   labels_idx:     每个样本的类别，如 tensor([   0,  1,    1, ...])
        print("\n Start Training")

        callback = pl.callbacks.ModelCheckpoint(
            dirpath=args.output_dir,
            monitor='val_loss',
            mode='min',
            save_top_k=3
        )

        log = TensorBoardLogger('logs', name=args.dataset)

        train_params = dict(
            default_root_dir=args.output_dir,
            accumulate_grad_batches=args.gradient_accumulation_steps,
            gpus=args.n_gpu,
            # strategy="dp",
            gradient_clip_val=1.0,
            # amp_level='O1',
            max_epochs=args.num_train_epochs,
            callbacks=[callback, EarlyStopping(monitor="val_loss", patience=8, mode="min")],
            logger=log,
            num_sanity_val_steps=0,
        )

        trainer = pl.Trainer(**train_params, auto_scale_batch_size='binsearch')
        trainer.fit(model)

    if args.do_eval:
        all_checkpoints = []

        if args.dataset == "GoEmotions" or args.dataset == "SemEvalEc":
            best_micro_f1, best_macro_f1, best_jaccord = 0, 0, 0
        if args.dataset == "Triplets" or args.dataset == "Triplets_Restaurant":
            best_precision, best_recall, best_micro_f1 = 0, 0, 0

        for f in os.listdir(args.output_dir):
            file_name = os.path.join(args.output_dir, f)
            if 'ckpt' in file_name:
                all_checkpoints.append(file_name)
        print(f'Test model on following checkpoints: {all_checkpoints}')



        for ckpt in all_checkpoints:
            model = T5EmotionGeneration.load_from_checkpoint(f'{ckpt}', hparams=args)
            tokenizer = model.tokenizer                                                     # {T5Tokenizer: 32100}
            test_dataset = get_dataset(tokenizer=tokenizer, data_type='test', args=args)

            device = torch.device(f'cuda:{args.device}')
            model.model.to(device)
            model.model.eval()

            test_dataloader = DataLoader(test_dataset, batch_size=32)   # len(test_dataset): 5427, len(test_dataloader): 340
            print(f"length of test_dataset: {len(test_dataset)}; length of test_dataloader: {len(test_dataloader)}")
            outputs, targets = [], []
            outputs_1, targets_1 = [], []
            print(f'results on {ckpt}')

            for i, batch in enumerate(tqdm(test_dataloader)):
            # =* COMMENT 2 *=:
            # 在 COMMENT 1 中介绍过, test_dataset 为一个 EmotionDataset() 类对象, 有两种表达方式:
            #   1. 使用其三个属性
            #   2. 可被一一索引. 索引时其每一个元素为一个 dict, test_dataset[i]:
            #       {
            #           "source_ids":   tensor([27, 22, 51, ..., 1, 0, ..., 0])                                 (100,)
            #           "source_mask":  tensor([ 1,  1,  1, ..., 1, 0, ..., 0])                                 (100,)
            #           "target_ids":   tensor([37, 13868, 24784, 19, 7103, 16, 48, 7142, 5, 1, 0, ..., 0])     (100,)
            #           "target_mask":  tensor([ 1,     1,     1,  1,    1,  1,  1,    1, 1, 1, 0, ..., 0])     (100,)
            #           "labels_idx":   tensor([ 0, 0, ..., 0, 1, 0, ..., 0])                                   (28,)
            #       }
            # 现 test_dataset 经 DataLoader() 处理后, 得到 test_dataloader, 共有 340 (5427 / 16) 个元素, 每个元素是一个 batch
            #   每个 batch 仍然是一个 dict, dict 的键和 test_dataset 被单独索引一样, 只不过元素成了相应值的一个 batch:
            #       {
            #           "source_ids":   tensor([[   27,    22,    51,  ...,     0,     0,     0],
            #                                       ...,
            #                                   [   27,   470,  1114,  ...,     0,     0,     0]])              (16, 100)
            #           "source_mask":  tensor([[    1,     1,     1,  ...,     0,     0,     0],
            #                                       ...,
            #                                   [    1,     1,     1,  ...,     0,     0,     0]])              (16, 100)
            #           "target_ids":   tensor([[   37, 13868, 24784,  ...,     0,     0,     0],
            #                                       ...,
            #                                   [   37, 13868,  1028,  ...,     0,     0,     0]])              (16, 100)
            #           "target_mask":  tensor([[   1,      1,     1,  ...,     0,     0,     0],
            #                                       ...,
            #                                   [   1,      1,     1,  ...,     0,     0,     0]])              (16, 100)
            #           "labels_idx":   tensor([[   0.,     0.,    0., ...,     1.,    0.,    0.],
            #                                       ...,
            #                                   [   0.,     0.,    0., ...,     0.,    0.,    0.]])             (16, 28)
            #       }

                if args.CLP:
                    label_mask = torch.ones(
                        batch['source_ids'].shape[0],
                        label_size,
                        dtype=torch.int
                    ).to(f'cuda:{args.device}')
                    attention_mask = torch.cat((label_mask, batch['source_mask'].to(device)), dim=1)
                else:
                    attention_mask = batch['source_mask'].to(device)

                outs = model.model.generate(                        # .generate() 方法是 T5ForConditionalGeneration() 类的方法,
                                                                    #   负责(通过利用交叉注意力)将输入句子的编码, (在解码器端)自回归地生成输出句子的编码.
                                                                    #       注意,这里所说输入输出的编码不是指单词的 embedding, 而是单词在词表中的编码,为一个整数值.如:
                                                                    #           输入 batch['source_ids'] 为:
                                                                    #               tensor([[27,  22,    51, ...,    0, 0, 0],
                                                                    #                       [94,  31,     7, ...,    0, 0, 0],
                                                                    #                       ...,
                                                                    #                       [27, 470,  1114, ...,    0, 0, 0]])     (16, 100)
                                                                    #           输出 outs 为:
                                                                    #               tensor([[ 0,  37, 13868, ..., 7142, 5, 1],
                                                                    #                       [ 0,  37, 13868, ...,    0, 0, 0],
                                                                    #                       ...,
                                                                    #                       [ 0,  37, 13868, ...,    0, 0, 0]])     (16, ?): 输出序列的长度是可变化的.
                                                                    #           输出可以通过 tokenizer.decode() 转化为真实的生成句子.
                    input_ids=batch['source_ids'].to(device),       # 注意, 这里在做预测的时候, input_ids 的维度为 (16, 100).
                                                                    #   即在做预测的时候, 模型应该也是将 label prompt 和输入句子一起编码
                    attention_mask=attention_mask,                  # attention_mask 的维度为 (16, 128), 即 28 + 100
                    max_length=256,
                    num_beams=args.num_beams
                )                                                   # .generate() 方法的输出是生成的一个 batch 的句子的编码. 其中, 编码指的是单词在词表中的编码; 每一个句子长度是变化的. (16, ?)

                pred_sentences, target_sentences = [], []
                for out in outs:                                                            # out:
                                                                                            #   一个生成句子的编码, 如:
                                                                                            #       tensor([0, 37, 13868, ..., 7142, 5, 1])
                    pred_sentence = tokenizer.decode(out, skip_special_tokens=True)         # pred_sentence:
                                                                                            #   由 out 得到的一个生成的句子, 如:
                                                                                            #       "The emotion remorse is expressed in this sentence. [SSEP] The emotion sadness is expressed in this sentence."
                    pred_sentences.append(pred_sentence)                                    # pred_sentences:
                                                                                            #   一个 batch 的生成句子:
                                                                                            #       ["The emotion ...", "The emotion ...", ..., "The emotion ...."]     (16个句子)
                outputs.append(pred_sentences)                                              # outputs:  所有的生成句子, 用 batch 分隔:
                                                                                            #       [["The emotion ...", "The emotion ...", ..., "The emotion ..."], ["", "", ..., ""], ..., ["", "", ..., ""]]     (5427/16=340个batch)
                                                                                            #         |<-------------------------------------------------------->|    |<----------->|         |<----------->|
                                                                                            #                       1 个 batch 的句子 (16个)                            一个 batch 的句子         一个 batch 的句子
                for ids in batch["target_ids"]:
                    target_sentence = tokenizer.decode(ids, skip_special_tokens=True)
                    target_sentences.append(target_sentence)
                targets.append(target_sentences)                                            # targets 的结构同 outputs, 为所有 batch 的 target 句子:
                                                                                            #       [["The emotion ...", "The emotion ...", ..., "The emotion ..."], ["", "", ..., ""], ..., ["", "", ..., ""]]     (5427/16=340个batch)
                                                                                            #         |<-------------------------------------------------------->|    |<----------->|         |<----------->|
                                                                                            #                       1 个 batch 的句子 (16个)                            一个 batch 的句子         一个 batch 的句子


            results = evaluate(outputs, targets, args.dataset)      # outputs 即为 pred, targets 即为 gold


            if args.dataset == "GoEmotions" or args.dataset == "SemEvalEc":
                if results['classification']['micro avg']['f1-score'] > best_micro_f1:
                    best_micro_f1, best_macro_f1, best_jaccord = results['classification']['micro avg']['f1-score'], \
                        results['classification']['macro avg']['f1-score'], \
                        results['jaccard_score']
                    best_out, labels = outputs, targets
                print(f"best_micro_f1: {best_micro_f1}, best_macro_f1: {best_macro_f1}, best_jaccord: {best_jaccord}")
                print("/*******************************/")
            elif args.dataset == "Triplets" or args.dataset == "Triplets_Restaurant":
                # 用于 Triplets 数据集
                if results[2] > best_micro_f1:
                    best_precision, best_recall, best_micro_f1 = results[0], results[1], results[2]

                print(f'precision: {results[0]}, recall: {results[1]}, f1_score: {results[2]}')
                print("/*******************************/")

        with open(f"test_results/{path_time_now}.txt", "a", encoding="utf-8") as f:
            f.write(f"{args.seed}\t{args.learning_rate}\t{args.alpha}\t{args.temperature}\t{args.CLP}\tBest: P: {best_precision}, R: {best_recall}, F1: {best_micro_f1}\n")
        print(f"args.seed: {args.seed}\nargs.CLP: {args.CLP}\nargs.learning_rate: {args.learning_rate}\nargs.alpha: {args.alpha}\nargs.temperature: {args.temperature}")
        print(f'Best performance: precision: {best_precision}, recall: {best_recall}, best_f1_score: {best_micro_f1}')
        print(f"==================================================================================================================================")
