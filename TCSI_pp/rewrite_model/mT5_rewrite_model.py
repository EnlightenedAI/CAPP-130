import torch
import json
import argparse
import os
import copy
from Data_loading import dataload
from torch.utils.data import Dataset, DataLoader
from eval import rouge_scorces
import pytorch_lightning as pl
from transformers import (
    AdamW,
    MT5ForConditionalGeneration,
    MT5Tokenizer,
    MT5Config,
)

pl.seed_everything(42)
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name = 'google/mt5-small'
tokenizer = MT5Tokenizer.from_pretrained(model_name)  # 'google/mt5-small
t5_model = MT5ForConditionalGeneration.from_pretrained(model_name)  # 'google/mt5-small'
t5_config = MT5Config.from_pretrained(model_name)
t5_model=t5_model.to(device)
t5_model.config.eos_token_id = tokenizer.eos_token_id
t5_model.config.pad_token_id = tokenizer.pad_token_id

batch_size=32
max_len_inp=250
max_len_out=150
max_epochs=100
train_is_ture=True
testdata_is_ture=True
lr=3e-4
data_path=f'../../../capp_130/Subdataset/Rewrite_Sentences/'
savepath=f"./result/mt5_rewrite_epoch{max_epochs}.ckpt" #save model
save_rewrite_path=f'./result/mt5_rewrite_epoch{max_epochs}.json' #save rewrite result
true_false_adjective_tuples_train,true_false_adjective_tuples_validation,true_false_adjective_tuples_test=dataload(data_path)

class FalseGenerationDataset(Dataset):
    def __init__(self, tokenizer, tf_list, max_len_inp=250, max_len_out=150):
        self.true_false_adjective_tuples = tf_list
        self.max_len_input = max_len_inp
        self.max_len_output = max_len_out
        self.tokenizer = tokenizer #token-model
        self.inputs = []
        self.templates=[]
        self.targets = []
        self.skippedcount = 0
        self._build()

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        source_ids = self.inputs[index]["input_ids"].squeeze()
        target_ids = self.targets[index]["input_ids"].squeeze()
        src_mask = self.inputs[index]["attention_mask"].squeeze()  # might need to squeeze
        target_mask = self.targets[index]["attention_mask"].squeeze()  # might need to squeeze
        labels = copy.deepcopy(target_ids)
        labels[labels == 0] = -100
        return {"source_ids": source_ids, "source_mask": src_mask,  "target_ids": target_ids, "target_mask": target_mask,
                "labels": labels}

    def _build(self):
        for inputs, outputs in self.true_false_adjective_tuples:
            input_sent = "summarization: " + inputs[:400]
            ouput_sent = outputs
            tokenized_inputs = self.tokenizer.batch_encode_plus(
                [input_sent], max_length=self.max_len_input, pad_to_max_length=True,return_tensors="pt"
            )
            tokenized_targets = self.tokenizer.batch_encode_plus(
                [ouput_sent], max_length=self.max_len_output, pad_to_max_length=True,return_tensors="pt"
            )
            self.inputs.append(tokenized_inputs)
            self.targets.append(tokenized_targets)

train_dataset = FalseGenerationDataset(tokenizer,true_false_adjective_tuples_train, max_len_inp, max_len_out)
validation_dataset = FalseGenerationDataset(tokenizer,true_false_adjective_tuples_validation, max_len_inp, max_len_out)

class FineTuner(pl.LightningModule):
    def __init__(self, hparams):
        super(FineTuner, self).__init__()
        self.save_hyperparameters()
        self.hparams = hparams
        self.model = t5_model
        self.tokenizer = tokenizer

    def forward(self, input_ids, attention_mask=None,template_ids=None,
            template_mask=None, decoder_input_ids=None, decoder_attention_mask=None,
                    lm_labels=None):
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=lm_labels,
        )
        return outputs

    def training_step(self, batch, batch_idx):
        outputs = self.forward(
            input_ids=batch["source_ids"],
            attention_mask=batch["source_mask"],
            decoder_input_ids=batch["target_ids"],
            decoder_attention_mask=batch['target_mask'],
            lm_labels=batch['labels']
        )

        loss = outputs[0]

        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        outputs = self.forward(
            input_ids=batch["source_ids"],
            attention_mask=batch["source_mask"],
            decoder_input_ids=batch["target_ids"],
            decoder_attention_mask=batch['target_mask'],
            lm_labels=batch['labels']
        )
        loss = outputs[0]
        self.log("val_loss", loss)
        return loss

    def train_dataloader(self):
        return DataLoader(train_dataset, batch_size=self.hparams.batch_size, num_workers=0)

    def val_dataloader(self):
        return DataLoader(validation_dataset, batch_size=self.hparams.batch_size, num_workers=0)

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=lr, eps=1e-8)
        return optimizer

args_dict = dict(
    batch_size=batch_size,
)
args = argparse.Namespace(**args_dict)
print('args',args)
model = FineTuner(args).to(device)
if train_is_ture:
    trainer = pl.Trainer(max_epochs=max_epochs, amp_level='01', gpus=1, progress_bar_refresh_rate=30)
    trainer.fit(model)
    trainer.save_checkpoint(savepath)

if testdata_is_ture:
    new_model = FineTuner(args).to(device)
    new_model = new_model.load_from_checkpoint(checkpoint_path=savepath)
    new_model.to(device)
    new_model.eval()
    new_model.model.to(device)
    with open(save_rewrite_path, 'w', encoding='utf-8') as fp:
        for text in true_false_adjective_tuples_test:
            test_tokenized = tokenizer.encode_plus('summarization'+text[0][:350], return_tensors="pt")
            test_input_ids = test_tokenized["input_ids"].to(device)
            test_attention_mask = test_tokenized["attention_mask"].to(device)
            beam_outputs = new_model.model.generate(
                input_ids=test_input_ids,
                attention_mask=test_attention_mask,
                max_length=250,
                early_stopping=True,
                num_beams=10,
                num_return_sequences=1,
                no_repeat_ngram_size=2,
            )
            for beam_output in beam_outputs:
                sent = tokenizer.decode(beam_output, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            sent=''.join(sent.split())
            fp.write(json.dumps({'text': text[0], 'pred': sent, 'rewrite': text[1]},ensure_ascii=False) + '\n')

print(rouge_scorces(save_rewrite_path))
torch.cuda.empty_cache()