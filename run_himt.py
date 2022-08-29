from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

"""BERT finetuning runner."""

import csv
import os
import logging
import argparse
import random
from tqdm import tqdm, trange

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

from my_bert.tokenization import BertTokenizer
from my_bert.mm_modeling import HierarchicalTransformer_TopN_Attention_v2
from my_bert.optimization import BertAdam
from my_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE

from sklearn.metrics import precision_recall_fscore_support

from _image_features_reader import ImageFeaturesH5Reader
from torch.nn import CrossEntropyLoss


def macro_f1(y_true, y_pred):
    preds = np.argmax(y_pred, axis=-1)
    true = y_true
    p_macro, r_macro, f_macro, support_macro \
      = precision_recall_fscore_support(true, preds, average='macro')
    # f_macro = 2*p_macro*r_macro/(p_macro+r_macro)
    return p_macro, r_macro, f_macro


logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
    datefmt='%m/%d/%Y %H:%M:%S',
    level=logging.INFO)
logger = logging.getLogger(__name__)


class InputExample(object):
    """A single training/test example for simple sequence classification."""
    def __init__(self, guid, text_a, text_b=None, label=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


class InputFeatures(object):
    """A single set of features of data."""
    def __init__(self, input_ids, input_mask, segment_ids, label_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id


class MultimodalExample(object):
    def __init__(self, guid, context, target, img_id, label_id):
        self.guid = guid
        self.text_a = context
        self.text_b = target
        self.img_id = img_id
        self.label_id = label_id


class MultimodalFeatures(object):
    def __init__(self, context_ids, context_mask, context_segment_ids,
                 target_ids, target_mask, target_segment_ids, img_features,
                 img_mask, img_sematic_ids, combine_mask, label_id):
        self.context_ids = context_ids
        self.context_mask = context_mask
        self.context_segment_ids = context_segment_ids
        self.target_ids = target_ids
        self.target_mask = target_mask
        self.target_segment_ids = target_segment_ids
        self.img_features = img_features
        self.img_mask = img_mask
        self.img_sematic_ids = img_sematic_ids
        self.combine_mask = combine_mask
        self.label_id = label_id


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""
    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_test_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r", encoding='utf-8') as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                lines.append(line)
            return lines


class AbmsaProcessor(DataProcessor):
    """Processor for the MRPC data set (GLUE version)."""
    def get_train_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {}".format(os.path.join(data_dir,
                                                        "train.tsv")))
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

    def get_labels(self):
        """See base class."""
        return ["0", "1", "2"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, i)
            text_a = line[3].lower()
            text_b = line[4].lower()
            img_id = line[2]
            label = line[1]
            examples.append(
                MultimodalExample(guid=guid,
                                  context=text_a,
                                  target=text_b,
                                  img_id=img_id,
                                  label_id=label))
        return examples


def readembedding(wordbag_path, embedding_path):
    embedding = []
    wordbag = dict()
    embedding.append(np.zeros(300))
    with open(embedding_path, 'r', encoding='utf-8') as f:
        lines = f.read().splitlines()
        for line in lines:
            embedding.append([float(num) for num in line.split()])

    assert np.shape(embedding)[1] == 300
    with open(wordbag_path, 'r', encoding='utf-8') as f:
        lines = f.read().splitlines()
        for (i, line) in enumerate(lines):
            wordbag[line] = i

    return torch.tensor(embedding, dtype=torch.float), wordbag


def convert_mmexamples_to_mmfeatures(examples, label_list, args, tokenizer,
                                     wordbag, img_features_reader, mode):
    label_map = {label: i for i, label in enumerate(label_list)}
    features = []
    # if mode == 'Train':
    #     random.shuffle(examples)
    #     len_example = len(examples)
    #     examples = examples[:int(0.75 * len_example)]
    for (ex_index, example) in enumerate(examples):
        tokens_a = tokenizer.tokenize(example.text_a)
        tokens_b = tokenizer.tokenize(example.text_b)

        _truncate_seq_pair(tokens_a, tokens_b, args.max_seq_length - 3)

        tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
        segment_ids = [0] * len(tokens)

        # tokens      += tokens_b + ["[SEP]"]
        # segment_ids += [1] * (len(tokens_b) + 1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1] * len(input_ids)

        padding = [0] * (args.max_seq_length - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding

        assert len(input_ids) == args.max_seq_length
        assert len(input_mask) == args.max_seq_length
        assert len(segment_ids) == args.max_seq_length

        target_tokens = tokens_b
        if len(target_tokens) > args.max_target_length:
            target_tokens = target_tokens[:args.max_target_length]

        target_tokens = ["[CLS]"] + target_tokens + ["[SEP]"] + tokens_a + ["[SEP]"]
        target_input_ids = tokenizer.convert_tokens_to_ids(target_tokens)
        target_input_mask = [1] * len(target_tokens)
        target_segment_ids = [0] * len(target_tokens)

        padding = [0] * (args.max_target_length - len(target_input_ids))
        target_input_ids += padding
        target_input_mask += padding
        target_segment_ids += padding

        img_features = img_features_reader[example.img_id][0][1:]
        img_features_num = len(img_features)
        img_mask = [1] * img_features_num

        img_semantics = img_features_reader[example.img_id][4].split('/')[:36]
        # print(img_semantics)
        img_semantic_list = []
        for text in img_semantics:
            words = text.split('_')
            now_id_list = []
            if len(words) > 3:
                attr = wordbag[words[0]]
            else:
                attr = 0
            for word in words[-3:]:
                try:
                    word_id = wordbag[word]
                    now_id_list.append([attr, word_id])
                except:
                    print(text)
            assert len(now_id_list) == 3
            img_semantic_list.append(now_id_list)


        combine_mask = [1] + target_input_mask
        # combine_mask = [1] + img_mask


        label_id = label_map[example.label_id]
        if ex_index < 1:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("tokens: %s" % " ".join([str(x) for x in tokens]))
            logger.info("input_ids: %s" % " ".join([str(x)
                                                    for x in input_ids]))
            logger.info("input_mask: %s" %
                        " ".join([str(x) for x in input_mask]))
            logger.info("segment_ids: %s" %
                        " ".join([str(x) for x in segment_ids]))
            logger.info("target tokens: %s" %
                        " ".join([str(x) for x in target_tokens]))
            logger.info("target mask: %s" %
                        " ".join([str(x) for x in target_input_mask]))
            logger.info("target segment ids: %s" %
                        " ".join([str(x) for x in target_segment_ids]))
            logger.info("image feature size: {}".format(
                np.shape(img_features)))
            logger.info("image semantic: {}".format("/".join(img_semantics)))
            logger.info("label: %s (id = %d)" % (example.label_id, label_id))

        features.append(
            MultimodalFeatures(context_ids=input_ids,
                               context_mask=input_mask,
                               context_segment_ids=segment_ids,
                               target_ids=target_input_ids,
                               target_mask=target_input_mask,
                               target_segment_ids=target_segment_ids,
                               img_features=img_features,
                               img_mask=img_mask,
                               img_sematic_ids=img_semantic_list,
                               combine_mask=combine_mask,
                               label_id=label_id))
    return features


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def get_DataLoader(args,
                   examples,
                   label_list,
                   tokenizer,
                   wordbag,
                   mode='Train'):
    img_features_reader = ImageFeaturesH5Reader(args.img_path + mode + '.lmdb')
    features = convert_mmexamples_to_mmfeatures(examples, label_list, args,
                                                tokenizer, wordbag,
                                                img_features_reader, mode)
    all_input_ids = torch.tensor([f.context_ids for f in features],
                                 dtype=torch.long)
    all_input_mask = torch.tensor([f.context_mask for f in features],
                                  dtype=torch.long)
    all_segment_ids = torch.tensor([f.context_segment_ids for f in features],
                                   dtype=torch.long)
    all_target_ids = torch.tensor([f.target_ids for f in features],
                                  dtype=torch.long)
    all_target_mask = torch.tensor([f.target_mask for f in features],
                                   dtype=torch.long)
    all_target_segment = torch.tensor([f.target_segment_ids for f in features],
                                      dtype=torch.long)
    all_img_features = torch.tensor([f.img_features for f in features],
                                    dtype=torch.float)
    all_img_mask = torch.tensor([f.img_mask for f in features],
                                dtype=torch.long)
    all_img_semantics = torch.tensor([f.img_sematic_ids for f in features],
                                     dtype=torch.long)
    all_combine_mask = torch.tensor([f.combine_mask for f in features],
                                    dtype=torch.long)
    all_label_ids = torch.tensor([f.label_id for f in features],
                                 dtype=torch.long)

    dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids,
                            all_target_ids, all_target_mask,
                            all_target_segment, all_img_features, all_img_mask,
                            all_img_semantics, all_combine_mask, all_label_ids)
    sampler = None
    dataloader = None
    if mode == 'Train':
        sampler = RandomSampler(dataset)
        dataloader = DataLoader(dataset,
                                sampler=sampler,
                                batch_size=args.train_batch_size)
    else:
        sampler = SequentialSampler(dataset)
        dataloader = DataLoader(dataset,
                                sampler=sampler,
                                batch_size=args.eval_batch_size)
    return dataloader


def accuracy(out, labels):
    outputs = np.argmax(out, axis=1)
    return np.sum(outputs == labels)


def warmup_linear(x, warmup=0.002):
    if x < warmup:
        return x / warmup
    return 1.0 - x


def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument(
        "--data_dir",
        default='../absa_data/twitter2015',
        type=str,
        required=True,
        help=
        "The input data dir. Should contain the .tsv files (or other data files) for the task."
    )
    parser.add_argument(
        "--img_path",
        default="",
        type=str,
        required=True)
    parser.add_argument(
        "--bert_model",
        default=None,
        type=str,
        required=True,
        help="Bert pre-trained model selected in the list: bert-base-uncased, "
        "bert-large-uncased, bert-base-cased, bert-large-cased, bert-base-multilingual-uncased, "
        "bert-base-multilingual-cased, bert-base-chinese.")
    parser.add_argument("--task_name",
                        default='twitter',
                        type=str,
                        required=True,
                        help="The name of the task to train.")
    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        required=True,
        help=
        "The output directory where the model predictions and checkpoints will be written."
    )

    ## Other parameters
    parser.add_argument(
        "--max_seq_length",
        default=64,
        type=int,
        help=
        "The maximum total input sequence length after WordPiece tokenization. \n"
        "Sequences longer than this will be truncated, and sequences shorter \n"
        "than this will be padded.")
    parser.add_argument("--max_target_length", default=16, type=int)
    parser.add_argument("--do_train",
                        action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval",
                        action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument(
        "--do_lower_case",
        action='store_true',
        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--train_batch_size",
                        default=32,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size",
                        default=16,
                        type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--learning_rate",
                        default=2e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs",
                        default=8.0,
                        type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument(
        "--warmup_proportion",
        default=0.1,
        type=float,
        help=
        "Proportion of training to perform linear learning rate warmup for. "
        "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--no_gate", default=True, type=bool)
    parser.add_argument("--r_loss", default=0.1, type=float)
    parser.add_argument("--no_cuda",
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--seed',
                        type=int,
                        default=58,
                        help="random seed for initialization")
    parser.add_argument(
        '--gradient_accumulation_steps',
        type=int,
        default=1,
        help=
        "Number of updates steps to accumulate before performing a backward/update pass."
    )
    parser.add_argument(
        '--fp16',
        action='store_true',
        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument(
        '--loss_scale',
        type=float,
        default=0,
        help=
        "Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
        "0 (default value): dynamic loss scaling.\n"
        "Positive power of 2: static loss scaling value.\n")
    parser.add_argument('--bertlayer',
                        action='store_true',
                        help='whether to add another bert layer')

    parser.add_argument('--wordbag_path',
                        default="../bottom-up-attention/mywordbag.txt")
    parser.add_argument('--embedding_path',
                        default="../bottom-up-attention/myembedding.txt")

    parser.add_argument('--projectionlayer', default=1, type=int)
    parser.add_argument('--cmtlayer', default=1, type=int)

    args = parser.parse_args()

    if args.bertlayer:
        print("add another bert layer")
    else:
        print("pre-trained bert without additional bert layer")

    if args.task_name == "twitter2015" and args.bertlayer:
        args.seed += 22

    processors = {"twitter": AbmsaProcessor, "twitter2015": AbmsaProcessor}

    num_labels_task = {"twitter": 3, "twitter2015": 3}

    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available()
                              and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')
    logger.info(
        "device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".
        format(device, n_gpu, bool(args.local_rank != -1), args.fp16))

    if args.gradient_accumulation_steps < 1:
        raise ValueError(
            "Invalid gradient_accumulation_steps parameter: {}, should be >= 1"
            .format(args.gradient_accumulation_steps))

    args.train_batch_size = int(args.train_batch_size /
                                args.gradient_accumulation_steps)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    if not args.do_train and not args.do_eval:
        raise ValueError(
            "At least one of `do_train` or `do_eval` must be True.")

    if os.path.exists(args.output_dir) and os.listdir(
            args.output_dir) and args.do_train:
        raise ValueError(
            "Output directory ({}) already exists and is not empty.".format(
                args.output_dir))
    os.makedirs(args.output_dir, exist_ok=True)

    task_name = args.task_name.lower()

    if task_name not in processors:
        raise ValueError("Task not found: %s" % (task_name))

    processor = processors[task_name]()
    num_labels = num_labels_task[task_name]
    label_list = processor.get_labels()

    embedding, wordbag = readembedding(args.wordbag_path, args.embedding_path)
    embedding = torch.nn.Embedding.from_pretrained(embeddings=embedding)

    tokenizer = BertTokenizer.from_pretrained(args.bert_model,
                                              do_lower_case=args.do_lower_case)

    train_examples = None
    num_train_steps = None
    if args.do_train:
        train_examples = processor.get_train_examples(args.data_dir)
        num_train_steps = int(
            len(train_examples) / args.train_batch_size /
            args.gradient_accumulation_steps * args.num_train_epochs)

    # Prepare model
    model = HierarchicalTransformer_TopN_Attention_v2.from_pretrained(
        args.bert_model,
        cache_dir=PYTORCH_PRETRAINED_BERT_CACHE /
        'distributed_{}'.format(args.local_rank),
        num_labels=num_labels,
        embedding=embedding,
        TAT_num=args.cmtlayer,
        MFT_num=args.projectionlayer)
    
    param_sum = sum(p.numel() for p in model.parameters())
    print(param_sum)

    if args.fp16:
        model.half()
    model.to(device)
    if args.local_rank != -1:
        try:
            from apex.parallel import DistributedDataParallel as DDP
        except ImportError:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training."
            )

        model = DDP(model)
    elif n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Prepare optimizer
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [{
        'params':
        [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
        'weight_decay':
        0.01
    }, {
        'params':
        [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
        'weight_decay':
        0.0
    }]
    t_total = num_train_steps
    if args.local_rank != -1:
        t_total = t_total // torch.distributed.get_world_size()
    if args.fp16:
        try:
            from apex.optimizers import FP16_Optimizer
            from apex.optimizers import FusedAdam
        except ImportError:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training."
            )

        optimizer = FusedAdam(optimizer_grouped_parameters,
                              lr=args.learning_rate,
                              bias_correction=False,
                              max_grad_norm=1.0)
        if args.loss_scale == 0:
            optimizer = FP16_Optimizer(optimizer, dynamic_loss_scale=True)
        else:
            optimizer = FP16_Optimizer(optimizer,
                                       static_loss_scale=args.loss_scale)

    else:
        optimizer = BertAdam(optimizer_grouped_parameters,
                             lr=args.learning_rate,
                             warmup=args.warmup_proportion,
                             t_total=t_total)

    global_step = 0
    nb_tr_steps = 0
    tr_loss = 0
    output_model_file = os.path.join(args.output_dir, "pytorch_model.bin")
    if args.do_train:
        train_dataloader = get_DataLoader(args, train_examples, label_list,
                                          tokenizer, wordbag)
        #'''
        eval_examples = processor.get_dev_examples(args.data_dir)
        eval_dataloader = get_DataLoader(args,
                                         eval_examples,
                                         label_list,
                                         tokenizer,
                                         wordbag,
                                         mode='Dev')

        max_acc = 0.0
        max_f1 = 0.0
        #'''
        logger.info("*************** Running training ***************")
        for train_idx in trange(int(args.num_train_epochs), desc="Epoch"):

            logger.info("********** Epoch: " + str(train_idx) + " **********")
            logger.info("  Num examples = %d", len(train_examples))
            logger.info("  Batch size = %d", args.train_batch_size)
            logger.info("  Num steps = %d", num_train_steps)
            model.train()
            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0
            for step, batch in enumerate(
                    tqdm(train_dataloader, desc="Iteration")):
                batch = tuple(t.to(device) for t in batch)
                input_ids, input_mask, segment_ids, target_ids, target_mask, target_segment, img_features, img_mask, img_semantic_ids, combine_mask, label_ids = batch

                logits, r_loss = model(
                    context_ids=input_ids,
                    context_mask=input_mask,
                    context_segment_ids=segment_ids,
                    target_ids=target_ids,
                    target_mask=target_mask,
                    target_segment_ids=target_segment,
                    img_feature=img_features,
                    img_mask=img_mask,
                    img_sematic_ids=img_semantic_ids,
                    combine_mask=combine_mask,
                    copy_flag=True if train_idx == 0 and step == 0 else False)

                loss_fct = CrossEntropyLoss()

                loss = loss_fct(logits.view(-1, 3),
                                label_ids.view(-1)) + args.r_loss * r_loss

                if args.fp16:
                    optimizer.backward(loss)
                else:
                    loss.backward()

                tr_loss += loss.item()
                nb_tr_examples += input_ids.size(0)
                nb_tr_steps += 1
                if (step + 1) % args.gradient_accumulation_steps == 0:
                    # modify learning rate with special warm up BERT uses
                    lr_this_step = args.learning_rate * warmup_linear(
                        global_step / t_total, args.warmup_proportion)
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = lr_this_step
                    optimizer.step()
                    optimizer.zero_grad()
                    global_step += 1

            logger.info("***** Running evaluation on Dev Set*****")
            logger.info("  Num examples = %d", len(eval_examples))
            logger.info("  Batch size = %d", args.eval_batch_size)
            model.eval()
            eval_loss, eval_accuracy = 0, 0
            nb_eval_steps, nb_eval_examples = 0, 0

            true_label_list = []
            pred_label_list = []

            for batch in tqdm(eval_dataloader, desc="Evaluating"):
                batch = tuple(t.to(device) for t in batch)
                input_ids, input_mask, segment_ids, target_ids, target_mask, target_segment, img_features, img_mask, img_semantic_ids, combine_mask, label_ids = batch

                with torch.no_grad():
                    # logits = model(input_ids, segment_ids, input_mask)
                    logits, _ = model(context_ids=input_ids,
                                      context_mask=input_mask,
                                      context_segment_ids=segment_ids,
                                      target_ids=target_ids,
                                      target_mask=target_mask,
                                      target_segment_ids=target_segment,
                                      img_feature=img_features,
                                      img_mask=img_mask,
                                      img_sematic_ids=img_semantic_ids,
                                      combine_mask=combine_mask,
                                      copy_flag=True if train_idx == 0
                                      and step == 0 else False)

                logits = logits.detach().cpu().numpy()
                label_ids = label_ids.to('cpu').numpy()
                true_label_list.append(label_ids)
                pred_label_list.append(logits)
                tmp_eval_accuracy = accuracy(logits, label_ids)

                eval_accuracy += tmp_eval_accuracy

                nb_eval_examples += input_ids.size(0)
                nb_eval_steps += 1

            eval_accuracy = eval_accuracy / nb_eval_examples
            loss = tr_loss / nb_tr_steps if args.do_train else None
            true_label = np.concatenate(true_label_list)
            pred_outputs = np.concatenate(pred_label_list)
            precision, recall, F_score = macro_f1(true_label, pred_outputs)
            result = {
                'eval_accuracy': eval_accuracy,
                'f_score': F_score,
                'global_step': global_step,
                'loss': loss
            }

            logger.info("***** Dev Eval results *****")
            for key in sorted(result.keys()):
                logger.info("  %s = %s", key, str(result[key]))

            if F_score >= max_f1:
                # Save a trained model
                model_to_save = model.module if hasattr(
                    model, 'module') else model  # Only save the model it-self
                if args.do_train:
                    torch.save(model_to_save.state_dict(), output_model_file)
                max_f1 = F_score

    # Load a trained model that you have fine-tuned

    model_state_dict = torch.load(output_model_file)
    model = HierarchicalTransformer_TopN_Attention_v2.from_pretrained(
        args.bert_model,
        state_dict=model_state_dict,
        num_labels=num_labels,
        embedding=embedding,
        TAT_num=args.cmtlayer,
        MFT_num=args.projectionlayer)

    model.to(device)

    if args.do_eval and (args.local_rank == -1
                         or torch.distributed.get_rank() == 0):
        eval_examples = processor.get_test_examples(args.data_dir)
        logger.info("***** Running evaluation on Test Set *****")
        logger.info("  Num examples = %d", len(eval_examples))
        logger.info("  Batch size = %d", args.eval_batch_size)
        eval_dataloader = get_DataLoader(args,
                                         eval_examples,
                                         label_list,
                                         tokenizer,
                                         wordbag,
                                         mode='Test')

        model.eval()
        eval_loss, eval_accuracy = 0, 0
        nb_eval_steps, nb_eval_examples = 0, 0

        true_label_list = []
        pred_label_list = []

        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            batch = tuple(t.to(device) for t in batch)
            input_ids, input_mask, segment_ids, target_ids, target_mask, target_segment, img_features, img_mask, img_semantic_ids, combine_mask, label_ids = batch
            with torch.no_grad():
                logits, _ = model(context_ids=input_ids,
                                  context_mask=input_mask,
                                  context_segment_ids=segment_ids,
                                  target_ids=target_ids,
                                  target_mask=target_mask,
                                  target_segment_ids=target_segment,
                                  img_feature=img_features,
                                  img_mask=img_mask,
                                  img_sematic_ids=img_semantic_ids,
                                  combine_mask=combine_mask,
                                  copy_flag=False)

            logits = logits.detach().cpu().numpy()
            label_ids = label_ids.to('cpu').numpy()

            true_label_list.append(label_ids)
            pred_label_list.append(logits)
            tmp_eval_accuracy = accuracy(logits, label_ids)

            eval_accuracy += tmp_eval_accuracy

            nb_eval_examples += input_ids.size(0)
            nb_eval_steps += 1

        eval_accuracy = eval_accuracy / nb_eval_examples
        loss = tr_loss / nb_tr_steps if args.do_train else None
        true_label = np.concatenate(true_label_list)
        pred_outputs = np.concatenate(pred_label_list)

        precision, recall, F_score = macro_f1(true_label, pred_outputs)
        result = {
            'eval_loss': eval_loss,
            'eval_accuracy': eval_accuracy,
            'precision': precision,
            'recall': recall,
            'f_score': F_score,
            'global_step': global_step,
            'loss': loss
        }

        pred_label = np.argmax(pred_outputs, axis=-1)
        fout_p = open(os.path.join(args.output_dir, "pred.txt"), 'w')
        fout_t = open(os.path.join(args.output_dir, "true.txt"), 'w')

        for i in range(len(pred_label)):
            attstr = str(pred_label[i])
            fout_p.write(attstr + '\n')
        for i in range(len(true_label)):
            attstr = str(true_label[i])
            fout_t.write(attstr + '\n')

        fout_p.close()
        fout_t.close()

        output_eval_file = os.path.join(args.output_dir, "eval_results.txt")
        with open(output_eval_file, "w") as writer:
            logger.info("***** Test Eval results *****")
            for key in sorted(result.keys()):
                logger.info("  %s = %s", key, str(result[key]))
                writer.write("%s = %s\n" % (key, str(result[key])))


if __name__ == "__main__":
    main()
