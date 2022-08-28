from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

"""PyTorch BERT model."""

import os
import copy
import json
import math
import logging
import tarfile
import tempfile
import shutil

import torch
from torch import nn
from torch.nn import CrossEntropyLoss

from .file_utils import cached_path

import torch.nn.functional as F
from torch.autograd import Variable

logger = logging.getLogger(__name__)

PRETRAINED_MODEL_ARCHIVE_MAP = {
    'bert-base-uncased':
    "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased.tar.gz",
    'bert-large-uncased':
    "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-uncased.tar.gz",
    'bert-base-cased':
    "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-cased.tar.gz",
    'bert-large-cased':
    "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-cased.tar.gz",
    'bert-base-multilingual-uncased':
    "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-multilingual-uncased.tar.gz",
    'bert-base-multilingual-cased':
    "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-multilingual-cased.tar.gz",
    'bert-base-chinese':
    "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-chinese.tar.gz",
}
CONFIG_NAME = 'bert_config.json'
WEIGHTS_NAME = 'pytorch_model.bin'


def gelu(x):
    """Implementation of the gelu activation function.
        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


def swish(x):
    return x * torch.sigmoid(x)


def get_mask(inputs, seq_len, max_seq_len=None):
    """
    Get mask matrix
    * inputs [tensor]: tensor corresponding to seq_len (batch_size * max_seq_len * input_size)
    * seq_len [tensor]: sequence length vector
    * max_seq_len [int]: max sequence length
    - mask [tensor]: mask matrix for each sample (batch_size * max_seq_len)
    """
    seq_len = seq_len.type_as(inputs.data)
    max_seq_len = inputs.size(1) if max_seq_len is None else max_seq_len
    query = torch.arange(0, max_seq_len,
                         device=inputs.device).unsqueeze(1).float()
    mask = torch.lt(query, seq_len.unsqueeze(0)).float().transpose(0, 1)
    return mask


ACT2FN = {"gelu": gelu, "relu": torch.nn.functional.relu, "swish": swish}


class ImgEnhancedLSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size, cell_size, img_size):
        super(ImgEnhancedLSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.cell_size = cell_size
        self.img_size = img_size
        self.drop_out = nn.Dropout(0.1)

        self.forget_gate = nn.Sequential(
            nn.Linear(self.input_size + self.hidden_size + self.img_size,
                      self.cell_size), nn.Sigmoid())

        self.input_gate = nn.Sequential(
            nn.Linear(self.input_size + self.hidden_size + self.img_size,
                      self.cell_size), nn.Sigmoid())

        self.output_gate = nn.Sequential(
            nn.Linear(self.input_size + self.hidden_size + self.img_size,
                      self.cell_size), nn.Sigmoid())

        self.cell_state = nn.Sequential(
            nn.Linear(self.hidden_size + self.input_size + self.img_size,
                      self.cell_size), nn.Tanh())

        self.noleaner = nn.Tanh()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self, inputs, hidden_state, cell_state, img_state):
        now_batch_size, now_input_size = inputs.size()
        #print(inputs.size())
        #print(hidden_state.size())
        #print(img_state.size())
        combine_state = torch.cat((inputs, hidden_state, img_state), dim=-1)

        f_gate = self.forget_gate(combine_state)
        i_gate = self.input_gate(combine_state)
        o_gate = self.output_gate(combine_state)

        candidate_cell = self.cell_state(combine_state)
        now_cell_state = f_gate * cell_state + i_gate * candidate_cell
        now_hidden_state = o_gate * self.noleaner(now_cell_state)
        #now_hidden_state = self.drop_out(now_hidden_state)

        return now_hidden_state, now_cell_state



class TargetGuidedAttention(nn.Module):
    def __init__(self, hidden_dim=300):
        super(TargetGuidedAttention, self).__init__()
        self.W = nn.Linear(768, 300)
        self.hidden_dim = hidden_dim

    def forward(self, query, kv):
        '''
        query [batchsize, 768]
        kv    [batchsize, 36, topn, 300]
        '''
        query = self.W(query)  #[batchsize, 300]
        query = query.unsqueeze(1).unsqueeze(1)  #[batchsize, 1, 1, 300]
        query = query.permute(0, 1, 3, 2)
        smi = torch.div(torch.matmul(kv, query), math.sqrt(
            self.hidden_dim)).squeeze()  #[batchsize, 36, topn]
        # print('smi----', smi.shape)
        if smi.dim() == 2:
            smi = smi.unsqueeze(-1)
        score = F.softmax(smi, dim=-1)

        # import pdb; pdb.set_trace()

        score = score.unsqueeze(-1).permute(0, 1, 3,
                                            2)  #[batchsize, 36, 1, topn]

        output = torch.matmul(score, kv).squeeze()  #[batchsize, 36, 300]

        return output


# for NAACL visual attention part
class Attention(nn.Module):
    def __init__(self,
                 embed_dim,
                 hidden_dim=None,
                 n_head=1,
                 score_function='scaled_dot_product',
                 dropout=0.1):
        ''' Attention Mechanism
        :param embed_dim:
        :param hidden_dim:
        :param n_head: num of head (Multi-Head Attention)
        :param score_function: scaled_dot_product / mlp (concat) / bi_linear (general dot)
        '''
        super(Attention, self).__init__()
        if hidden_dim is None:
            hidden_dim = embed_dim // n_head
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.n_head = n_head
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        self.score_function = score_function
        self.w_kx = nn.Parameter(
            torch.FloatTensor(n_head, embed_dim, hidden_dim))
        self.w_qx = nn.Parameter(
            torch.FloatTensor(n_head, embed_dim, hidden_dim))
        self.proj = nn.Linear(n_head * hidden_dim, embed_dim)
        if score_function == 'mlp':
            self.weight = nn.Parameter(torch.Tensor(hidden_dim * 2, 1))
        elif self.score_function == 'bi_linear':
            self.weight = nn.Parameter(torch.Tensor(hidden_dim, hidden_dim))
        else:
            self.register_parameter('weight', None)

    def forward(self, k, q, memory_len):
        if len(q.shape) == 2:  # q_len missing
            q = torch.unsqueeze(q, dim=1)
        if len(k.shape) == 2:  # k_len missing
            k = torch.unsqueeze(k, dim=1)
        mb_size = k.shape[0]  # ?
        k_len = k.shape[1]
        q_len = q.shape[1]
        # k: (?, k_len, embed_dim,)
        # q: (?, q_len, embed_dim,)
        # kx: (n_head, ?*k_len, embed_dim) -> (n_head*?, k_len, hidden_dim)
        # qx: (n_head, ?*q_len, embed_dim) -> (n_head*?, q_len, hidden_dim)
        # score: (n_head*?, q_len, k_len,)
        # output: (?, q_len, embed_dim,)
        kx = k.repeat(self.n_head, 1,
                      1).view(self.n_head, -1,
                              self.embed_dim)  # (n_head, ?*k_len, embed_dim)
        qx = q.repeat(self.n_head, 1,
                      1).view(self.n_head, -1,
                              self.embed_dim)  # (n_head, ?*q_len, embed_dim)
        kx = torch.bmm(kx, self.w_kx).view(
            -1, k_len, self.hidden_dim)  # (n_head*?, k_len, hidden_dim)
        qx = torch.bmm(qx, self.w_qx).view(
            -1, q_len, self.hidden_dim)  # (n_head*?, q_len, hidden_dim)
        if self.score_function == 'scaled_dot_product':
            kt = kx.permute(0, 2, 1)
            qkt = torch.bmm(qx, kt)
            score = torch.div(qkt, math.sqrt(self.hidden_dim))
        elif self.score_function == 'mlp':
            kxx = torch.unsqueeze(kx, dim=1).expand(-1, q_len, -1, -1)
            qxx = torch.unsqueeze(qx, dim=2).expand(-1, -1, k_len, -1)
            kq = torch.cat((kxx, qxx),
                           dim=-1)  # (n_head*?, q_len, k_len, hidden_dim*2)
            score = torch.tanh(torch.matmul(kq, self.weight).squeeze(dim=-1))
        elif self.score_function == 'bi_linear':
            qw = torch.matmul(qx, self.weight)
            kt = kx.permute(0, 2, 1)
            score = torch.tanh(torch.bmm(qw, kt))
        else:
            raise RuntimeError('invalid score_function')
        score = F.softmax(score, dim=-1)
        attentions = torch.squeeze(score, dim=1)
        # print(attentions[:2])
        # create mask based on the sentence lengths
        mask = Variable(torch.ones(attentions.size())).to(self.device)
        for i, l in enumerate(memory_len):
            if l < k_len:
                mask[i, l:] = 0
        # apply mask and renormalize attention scores (weights)
        masked = attentions * mask
        # print(masked[:2])
        # print(masked.shape)
        _sums = masked.sum(-1)  # sums per row
        attentions = torch.div(masked, _sums.view(_sums.size(0), 1))
        # print(attentions[:2])

        score = torch.unsqueeze(attentions, dim=1)

        output = torch.bmm(score, kx)  # (n_head*?, k_len, hidden_dim)
        output = torch.cat(torch.split(output, mb_size, dim=0),
                           dim=-1)  # (?, k_len, n_head*hidden_dim)
        output = self.proj(output)  # (?, k_len, embed_dim)
        return output


class SelfAttention(Attention):
    '''q is a parameter'''
    def __init__(self,
                 embed_dim,
                 hidden_dim=None,
                 n_head=1,
                 score_function='scaled_dot_product',
                 q_len=1,
                 dropout=0.1):
        super(SelfAttention, self).__init__(embed_dim, hidden_dim, n_head,
                                            score_function, dropout)
        self.q_len = q_len
        self.q = nn.Parameter(torch.FloatTensor(q_len, embed_dim))

    def forward(self, k, **kwargs):
        mb_size = k.shape[0]
        q = self.q.expand(mb_size, -1, -1)
        return super(SelfAttention, self).forward(k, q)


class BertConfig(object):
    """Configuration class to store the configuration of a `BertModel`.
    """
    def __init__(self,
                 vocab_size_or_config_json_file,
                 hidden_size=768,
                 num_hidden_layers=12,
                 num_attention_heads=12,
                 intermediate_size=3072,
                 hidden_act="gelu",
                 hidden_dropout_prob=0.1,
                 attention_probs_dropout_prob=0.1,
                 max_position_embeddings=512,
                 type_vocab_size=2,
                 initializer_range=0.02):
        """Constructs BertConfig.

        Args:
            vocab_size_or_config_json_file: Vocabulary size of `inputs_ids` in `BertModel`.
            hidden_size: Size of the encoder layers and the pooler layer.
            num_hidden_layers: Number of hidden layers in the Transformer encoder.
            num_attention_heads: Number of attention heads for each attention layer in
                the Transformer encoder.
            intermediate_size: The size of the "intermediate" (i.e., feed-forward)
                layer in the Transformer encoder.
            hidden_act: The non-linear activation function (function or string) in the
                encoder and pooler. If string, "gelu", "relu" and "swish" are supported.
            hidden_dropout_prob: The dropout probabilitiy for all fully connected
                layers in the embeddings, encoder, and pooler.
            attention_probs_dropout_prob: The dropout ratio for the attention
                probabilities.
            max_position_embeddings: The maximum sequence length that this model might
                ever be used with. Typically set this to something large just in case
                (e.g., 512 or 1024 or 2048).
            type_vocab_size: The vocabulary size of the `token_type_ids` passed into
                `BertModel`.
            initializer_range: The sttdev of the truncated_normal_initializer for
                initializing all weight matrices.
        """
        if isinstance(vocab_size_or_config_json_file, str):
            with open(vocab_size_or_config_json_file, "r",
                      encoding='utf-8') as reader:
                json_config = json.loads(reader.read())
            for key, value in json_config.items():
                self.__dict__[key] = value
        elif isinstance(vocab_size_or_config_json_file, int):
            self.vocab_size = vocab_size_or_config_json_file
            self.hidden_size = hidden_size
            self.num_hidden_layers = num_hidden_layers
            self.num_attention_heads = num_attention_heads
            self.hidden_act = hidden_act
            self.intermediate_size = intermediate_size
            self.hidden_dropout_prob = hidden_dropout_prob
            self.attention_probs_dropout_prob = attention_probs_dropout_prob
            self.max_position_embeddings = max_position_embeddings
            self.type_vocab_size = type_vocab_size
            self.initializer_range = initializer_range
        else:
            raise ValueError(
                "First argument must be either a vocabulary size (int)"
                "or the path to a pretrained model config file (str)")

    @classmethod
    def from_dict(cls, json_object):
        """Constructs a `BertConfig` from a Python dictionary of parameters."""
        config = BertConfig(vocab_size_or_config_json_file=-1)
        for key, value in json_object.items():
            config.__dict__[key] = value
        return config

    @classmethod
    def from_json_file(cls, json_file):
        """Constructs a `BertConfig` from a json file of parameters."""
        with open(json_file, "r", encoding='utf-8') as reader:
            text = reader.read()
        return cls.from_dict(json.loads(text))

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


try:
    from apex.normalization.fused_layer_norm import FusedLayerNorm as BertLayerNorm
except ImportError:
    print(
        "Better speed can be achieved with apex installed from https://www.github.com/nvidia/apex."
    )

    class BertLayerNorm(nn.Module):
        def __init__(self, hidden_size, eps=1e-12):
            print('**********layer Norm**********')
            """Construct a layernorm module in the TF style (epsilon inside the square root).
            """
            super(BertLayerNorm, self).__init__()
            self.weight = nn.Parameter(torch.ones(hidden_size))
            self.bias = nn.Parameter(torch.zeros(hidden_size))
            self.variance_epsilon = eps

        def forward(self, x):
            u = x.mean(-1, keepdim=True)
            s = (x - u).pow(2).mean(-1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.variance_epsilon)
            return self.weight * x + self.bias


class BertEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings.
    """
    def __init__(self, config):
        super(BertEmbeddings, self).__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size,
                                            config.hidden_size)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings,
                                                config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size,
                                                  config.hidden_size)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_ids, token_type_ids=None):
        seq_length = input_ids.size(1)
        position_ids = torch.arange(seq_length,
                                    dtype=torch.long,
                                    device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        words_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = words_embeddings + position_embeddings + token_type_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class BertSelfAttention(nn.Module):
    def __init__(self, config):
        super(BertSelfAttention, self).__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" %
                (config.hidden_size, config.num_attention_heads))
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size /
                                       config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads,
                                       self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer,
                                        key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(
            self.attention_head_size)
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (
            self.all_head_size, )
        context_layer = context_layer.view(*new_context_layer_shape)
        return context_layer


class BertImageEmbeddings(nn.Module):
    """Construct the embeddings from image, spatial location (omit now) and token_type embeddings.
    """
    def __init__(self, config):
        super(BertImageEmbeddings, self).__init__()

        self.image_embeddings = nn.Linear(2048, config.hidden_size)
        self.image_location_embeddings = nn.Linear(5, config.hidden_size)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_ids, input_loc):
        img_embeddings = self.image_embeddings(input_ids)
        loc_embeddings = self.image_location_embeddings(input_loc)
        embeddings = self.LayerNorm(img_embeddings + loc_embeddings)
        embeddings = self.dropout(embeddings)

        return embeddings


class BertCoAttention(nn.Module):
    def __init__(self, config):
        super(BertCoAttention, self).__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" %
                (config.hidden_size, config.num_attention_heads))
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size /
                                       config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads,
                                       self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, s1_hidden_states, s2_hidden_states, s2_attention_mask):
        mixed_query_layer = self.query(s1_hidden_states)
        mixed_key_layer = self.key(s2_hidden_states)
        mixed_value_layer = self.value(s2_hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer,
                                        key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(
            self.attention_head_size)
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        attention_scores = attention_scores + s2_attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (
            self.all_head_size, )
        context_layer = context_layer.view(*new_context_layer_shape)
        return context_layer


class BertSelfOutput(nn.Module):
    def __init__(self, config):
        super(BertSelfOutput, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertAttention(nn.Module):
    def __init__(self, config):
        super(BertAttention, self).__init__()
        self.self = BertSelfAttention(config)
        self.output = BertSelfOutput(config)

    def forward(self, input_tensor, attention_mask):
        self_output = self.self(input_tensor, attention_mask)
        attention_output = self.output(self_output, input_tensor)
        return attention_output


class BertCrossAttention(nn.Module):
    def __init__(self, config):
        super(BertCrossAttention, self).__init__()
        self.self = BertCoAttention(config)
        self.output = BertSelfOutput(config)

    def forward(self, s1_input_tensor, s2_input_tensor, s2_attention_mask):
        s1_cross_output = self.self(s1_input_tensor, s2_input_tensor,
                                    s2_attention_mask)
        attention_output = self.output(s1_cross_output, s1_input_tensor)
        return attention_output


class BertIntermediate(nn.Module):
    def __init__(self, config):
        super(BertIntermediate, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        self.intermediate_act_fn = ACT2FN[config.hidden_act] \
            if isinstance(config.hidden_act, str) else config.hidden_act

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class BertOutput(nn.Module):
    def __init__(self, config):
        super(BertOutput, self).__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class DecoderOutput(nn.Module):
    def __init__(self, config):
        super(DecoderOutput, self).__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertLayer(nn.Module):
    def __init__(self, config):
        super(BertLayer, self).__init__()
        self.attention = BertAttention(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(self, hidden_states, attention_mask):
        attention_output = self.attention(hidden_states, attention_mask)
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output


class BertLayerForAttention(nn.Module):
    def __init__(self, config):
        super(BertLayer, self).__init__()
        self.attention = BertAttention(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(self, hidden_states, attention_mask, get_attention_mask):
        attention_output = self.attention(hidden_states, attention_mask)
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output


class BertCrossAttentionLayer(nn.Module):
    def __init__(self, config):
        super(BertCrossAttentionLayer, self).__init__()
        self.attention = BertCrossAttention(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(self, s1_hidden_states, s2_hidden_states, s2_attention_mask):
        attention_output = self.attention(s1_hidden_states, s2_hidden_states,
                                          s2_attention_mask)
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output


class BertEncoder(nn.Module):
    def __init__(self, config):
        super(BertEncoder, self).__init__()
        layer = BertLayer(config)
        self.layer = nn.ModuleList(
            [copy.deepcopy(layer) for _ in range(config.num_hidden_layers)])

    def forward(self,
                hidden_states,
                attention_mask,
                output_all_encoded_layers=True):
        all_encoder_layers = []
        for layer_module in self.layer:
            hidden_states = layer_module(hidden_states, attention_mask)
            if output_all_encoded_layers:
                all_encoder_layers.append(hidden_states)
        if not output_all_encoded_layers:
            all_encoder_layers.append(hidden_states)
        return all_encoder_layers


class BertEncoderForAttention(nn.Module):
    def __init__(self, config):
        super(BertEncoder, self).__init__()
        layer = BertLayer(config)
        self.layer = nn.ModuleList(
            [copy.deepcopy(layer) for _ in range(config.num_hidden_layers)])

    def forward(self,
                hidden_states,
                attention_mask,
                output_all_encoded_layers=True):
        all_encoder_layers = []
        for layer_module in self.layer:
            hidden_states = layer_module(hidden_states, attention_mask)
            if output_all_encoded_layers:
                all_encoder_layers.append(hidden_states)
        if not output_all_encoded_layers:
            all_encoder_layers.append(hidden_states)
        return all_encoder_layers


class MultimodalDecoderBlock(nn.Module):
    def __init__(self, config):
        super(MultimodalDecoderBlock, self).__init__()
        self.self_attention = BertAttention(config)
        self.mid_intermediate = BertIntermediate(config)
        self.encoder_decoder_attention = BertCoAttention(config)
        self.top_intermediate = BertIntermediate(config)
        self.output = DecoderOutput(config)

    def forward(self, encoder_hidden_states, hidden_states,
                encoder_attention_mask, attention_mask):
        self_attention_output = self.self_attention(hidden_states,
                                                    encoder_attention_mask)
        #mid_intermediate_output = self.mid_intermediate(self_attention_output)
        encoder_decoder_attention_output = self.encoder_decoder_attention(
            hidden_states, encoder_hidden_states, attention_mask)
        top_intermediate_output = self.top_intermediate(
            encoder_decoder_attention_output)
        output = self.output(top_intermediate_output, self_attention_output)

        return output


class MultimodalityFusionBlock(nn.Module):
    def __init__(self, config):
        super(MultimodalityFusionBlock, self).__init__()
        self.self_attention = BertAttention(config)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-12)
        self.cross_attention = BertCoAttention(config)
        self.intermediate = BertIntermediate(config)
        self.output = DecoderOutput(config)

    def forward(self, query_hidden_state, kv_hidden_state,
                query_attention_mask, kv_attention_mask):
        self_attention_output = self.self_attention(query_hidden_state,
                                                    query_attention_mask)
        self_attention_output = self.LayerNorm(self_attention_output +
                                               query_hidden_state)
        cross_modal_output = self.cross_attention(query_hidden_state,
                                                  kv_hidden_state,
                                                  kv_attention_mask)
        intermediate_output = self.intermediate(cross_modal_output)
        output = self.output(intermediate_output, self_attention_output)

        return output


class MultimodalityFusionLayer(nn.Module):
    def __init__(self, config, layernum=1):
        super(MultimodalityFusionLayer, self).__init__()
        block = MultimodalityFusionBlock(config)

        self.mfl = nn.ModuleList(
            [copy.deepcopy(block) for _ in range(layernum)])

    def forward(self, query_hidden_state, kv_hidden_state,
                query_attention_mask, kv_attention_mask):
        all_encoder_layers = []
        for block in self.mfl:
            query_hidden_state = block(query_hidden_state, kv_hidden_state,
                                       query_attention_mask, kv_attention_mask)
            all_encoder_layers.append(query_hidden_state)
        return all_encoder_layers[-1]


class MultimodalEncoderDecoder(nn.Module):
    def __init__(self, config, EncoderLayerNum=2, DecoderLayerNum=3):
        super(MultimodalEncoderDecoder, self).__init__()
        encoder_block = BertCrossAttentionLayer(config)
        #encoder_block = BertLayer(config)
        decoder_block = MultimodalDecoderBlock(config)
        self.encoder = nn.ModuleList(
            [copy.deepcopy(encoder_block) for _ in range(EncoderLayerNum)])
        self.decoder = nn.ModuleList(
            [copy.deepcopy(decoder_block) for _ in range(DecoderLayerNum)])

    def forward(self, vis_hidden, context_hidden, vis_attention_mask,
                context_attention_mask):

        extended_attention_mask = context_attention_mask.unsqueeze(
            1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(
            dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        all_encoder_layers = []
        for encoder_block in self.encoder:
            vis_hidden = encoder_block(vis_hidden, context_hidden,
                                       extended_attention_mask)
            #vis_hidden = encoder_block(vis_hidden, vis_attention_mask)
            all_encoder_layers.append(vis_hidden)
        encoder_output_layer = all_encoder_layers[-1]

        all_decoder_layers = []
        for decoder_block in self.decoder:
            context_hidden = decoder_block(encoder_output_layer,
                                           context_hidden,
                                           extended_attention_mask,
                                           vis_attention_mask)
            all_decoder_layers.append(context_hidden)

        return all_decoder_layers[-1]


class MultimodalEncoder(nn.Module):
    def __init__(self, config):
        super(MultimodalEncoder, self).__init__()
        layer = BertLayer(config)
        self.layer = nn.ModuleList([copy.deepcopy(layer) for _ in range(1)])

    def forward(self,
                hidden_states,
                attention_mask,
                output_all_encoded_layers=True):
        all_encoder_layers = []
        for layer_module in self.layer:
            hidden_states = layer_module(hidden_states, attention_mask)
            if output_all_encoded_layers:
                all_encoder_layers.append(hidden_states)
        if not output_all_encoded_layers:
            all_encoder_layers.append(hidden_states)
        return all_encoder_layers


class SemanticAttention(nn.Module):
    def __init__(self, config):
        super(SemanticAttention, self).__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" %
                (config.hidden_size, config.num_attention_heads))
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size /
                                       config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads,
                                       self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, query_hidden_states, key_hidden_states,
                value_hidden_states, kv_attention_mask):
        mixed_query_layer = self.query(query_hidden_states)
        mixed_key_layer = self.key(key_hidden_states)
        mixed_value_layer = self.value(value_hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer,
                                        key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(
            self.attention_head_size)
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        attention_scores = attention_scores + kv_attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (
            self.all_head_size, )
        context_layer = context_layer.view(*new_context_layer_shape)
        return context_layer


class SemanticEncoderLayer(nn.Module):
    def __init__(self, config):
        super(SemanticEncoderLayer, self).__init__()
        self.attention = SemanticAttention(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(self, query_hidden_states, key_hidden_states,
                value_hidden_states, kv_attention_mask):
        attention_output = self.attention(query_hidden_states,
                                          key_hidden_states,
                                          value_hidden_states,
                                          kv_attention_mask)
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output


class SemanticDecoderLayer(nn.Module):
    def __init__(self, config):
        super(SemanticDecoderLayer, self).__init__()
        self.self_attention = BertAttention(config)
        self.encoder_decoder_attention = BertCrossAttention(config)
        self.intermediate = BertIntermediate(config)
        self.layer_output = BertOutput(config)

    def forward(self, encoder_hidden_states, hidden_states,
                encoder_attention_mask, attention_mask):
        self_attention_output = self.self_attention(hidden_states,
                                                    attention_mask)
        ed_attention = self.encoder_decoder_attention(self_attention_output,
                                                      encoder_hidden_states,
                                                      encoder_attention_mask)
        intermediate_output = self.intermediate(ed_attention)
        output = self.layer_output(intermediate_output, ed_attention)
        return output


class SemanticEncoder(nn.Module):
    def __init__(self, config, layer_num=2):
        super(SemanticEncoder, self).__init__()
        encoder_block = SemanticEncoderLayer(config)
        self.encoder = nn.ModuleList(
            [copy.deepcopy(encoder_block) for _ in range(layer_num)])

    def forward(self, encoder_q, encoder_k, encoder_v, encoder_kv_mask):
        for encoder_block in self.encoder:
            encoder_q = encoder_block(encoder_q, encoder_k, encoder_v,
                                      encoder_kv_mask)
        return encoder_q


class MultimodalSemanticEncoderDecoder(nn.Module):
    def __init__(self, config, EncoderLayerNum=2, DecoderLayerNum=3):
        super(MultimodalSemanticEncoderDecoder, self).__init__()
        encoder_block = SemanticEncoderLayer(config)
        decoder_block = SemanticDecoderLayer(config)
        self.encoder = nn.ModuleList(
            [copy.deepcopy(encoder_block) for _ in range(EncoderLayerNum)])
        self.decoder = nn.ModuleList(
            [copy.deepcopy(decoder_block) for _ in range(DecoderLayerNum)])

    def forward(self, encoder_q, encoder_k, encoder_v, decoder_q,
                encoder_kv_mask, decoder_q_mask, decoder_kv_mask):
        for encoder_block in self.encoder:
            encoder_q = encoder_block(encoder_q, encoder_k, encoder_v,
                                      encoder_kv_mask)
        encoder_output = encoder_q
        for decoder_block in self.decoder:
            decoder_q = decoder_block(encoder_output, decoder_q,
                                      decoder_kv_mask, decoder_q_mask)
        output = decoder_q
        return decoder_q


class BertSemanticCrossEncoder(nn.Module):
    def __init__(self, config, layer_num=2):
        super(BertSemanticCrossEncoder, self).__init__()
        layer = SemanticAttentionLayer(config)
        #layer = BertCrossAttentionLayer(config)
        self.layer = nn.ModuleList(
            [copy.deepcopy(layer) for _ in range(layer_num - 1)])

    def forward(self,
                query_hidden_states,
                key_hidden_states,
                value_hidden_states,
                kv_attention_mask,
                output_all_encoded_layers=True):
        all_encoder_layer = []
        for layer_module in self.layer:
            query_hidden_states = self.semantic_layer(query_hidden_states,
                                                      key_hidden_states,
                                                      value_hidden_states,
                                                      kv_attention_mask)
            all_encoder_layer.append(query_hidden_states)
        return all_encoder_layer[-1]


class BertCrossEncoder(nn.Module):
    def __init__(self, config, layer_num=1):
        super(BertCrossEncoder, self).__init__()
        layer = BertCrossAttentionLayer(config)
        self.layer = nn.ModuleList(
            [copy.deepcopy(layer) for _ in range(layer_num)])

    def forward(self,
                s1_hidden_states,
                s2_hidden_states,
                s2_attention_mask,
                output_all_encoded_layers=True):
        all_encoder_layers = []
        for layer_module in self.layer:
            s1_hidden_states = layer_module(s1_hidden_states, s2_hidden_states,
                                            s2_attention_mask)
            if output_all_encoded_layers:
                all_encoder_layers.append(s1_hidden_states)
        if not output_all_encoded_layers:
            all_encoder_layers.append(s1_hidden_states)
        return all_encoder_layers


class BertPooler(nn.Module):
    def __init__(self, config):
        super(BertPooler, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class BertText1Pooler(nn.Module):
    def __init__(self, config):
        super(BertText1Pooler, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token of text.
        first_token_tensor = hidden_states[:, 1]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class BertPredictionHeadTransform(nn.Module):
    def __init__(self, config):
        super(BertPredictionHeadTransform, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.transform_act_fn = ACT2FN[config.hidden_act] \
            if isinstance(config.hidden_act, str) else config.hidden_act
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-12)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


class BertLMPredictionHead(nn.Module):
    def __init__(self, config, bert_model_embedding_weights):
        super(BertLMPredictionHead, self).__init__()
        self.transform = BertPredictionHeadTransform(config)

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        self.decoder = nn.Linear(bert_model_embedding_weights.size(1),
                                 bert_model_embedding_weights.size(0),
                                 bias=False)
        self.decoder.weight = bert_model_embedding_weights
        self.bias = nn.Parameter(
            torch.zeros(bert_model_embedding_weights.size(0)))

    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states) + self.bias
        return hidden_states


class BertOnlyMLMHead(nn.Module):
    def __init__(self, config, bert_model_embedding_weights):
        super(BertOnlyMLMHead, self).__init__()
        self.predictions = BertLMPredictionHead(config,
                                                bert_model_embedding_weights)

    def forward(self, sequence_output):
        prediction_scores = self.predictions(sequence_output)
        return prediction_scores


class BertOnlyNSPHead(nn.Module):
    def __init__(self, config):
        super(BertOnlyNSPHead, self).__init__()
        self.seq_relationship = nn.Linear(config.hidden_size, 2)

    def forward(self, pooled_output):
        seq_relationship_score = self.seq_relationship(pooled_output)
        return seq_relationship_score


class BertPreTrainingHeads(nn.Module):
    def __init__(self, config, bert_model_embedding_weights):
        super(BertPreTrainingHeads, self).__init__()
        self.predictions = BertLMPredictionHead(config,
                                                bert_model_embedding_weights)
        self.seq_relationship = nn.Linear(config.hidden_size, 2)

    def forward(self, sequence_output, pooled_output):
        prediction_scores = self.predictions(sequence_output)
        seq_relationship_score = self.seq_relationship(pooled_output)
        return prediction_scores, seq_relationship_score


class PreTrainedBertModel(nn.Module):
    """ An abstract class to handle weights initialization and
        a simple interface for dowloading and loading pretrained models.
    """
    def __init__(self, config, *inputs, **kwargs):
        super(PreTrainedBertModel, self).__init__()
        if not isinstance(config, BertConfig):
            raise ValueError(
                "Parameter config in `{}(config)` should be an instance of class `BertConfig`. "
                "To create a model from a Google pretrained model use "
                "`model = {}.from_pretrained(PRETRAINED_MODEL_NAME)`".format(
                    self.__class__.__name__, self.__class__.__name__))
        self.config = config

    def init_bert_weights(self, module):
        """ Initialize the weights.
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0,
                                       std=self.config.initializer_range)
        elif isinstance(module, BertLayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    @classmethod
    def from_pretrained(cls,
                        pretrained_model_name,
                        state_dict=None,
                        cache_dir=None,
                        *inputs,
                        **kwargs):
        """
        Instantiate a PreTrainedBertModel from a pre-trained model file or a pytorch state dict.
        Download and cache the pre-trained model file if needed.

        Params:
            pretrained_model_name: either:
                - a str with the name of a pre-trained model to load selected in the list of:
                    . `bert-base-uncased`
                    . `bert-large-uncased`
                    . `bert-base-cased`
                    . `bert-large-cased`
                    . `bert-base-multilingual-uncased`
                    . `bert-base-multilingual-cased`
                    . `bert-base-chinese`
                - a path or url to a pretrained model archive containing:
                    . `bert_config.json` a configuration file for the model
                    . `pytorch_model.bin` a PyTorch dump of a BertForPreTraining instance
            cache_dir: an optional path to a folder in which the pre-trained models will be cached.
            state_dict: an optional state dictionnary (collections.OrderedDict object) to use instead of Google pre-trained models
            *inputs, **kwargs: additional input for the specific Bert class
                (ex: num_labels for BertForSequenceClassification)
        """
        print(
            "*************************load pretrained model******************")
        if pretrained_model_name in PRETRAINED_MODEL_ARCHIVE_MAP:
            archive_file = PRETRAINED_MODEL_ARCHIVE_MAP[pretrained_model_name]
        else:
            archive_file = pretrained_model_name
        # redirect to the cache, if necessary
        try:
            resolved_archive_file = cached_path(archive_file,
                                                cache_dir=cache_dir)
        except FileNotFoundError:
            logger.error(
                "Model name '{}' was not found in model name list ({}). "
                "We assumed '{}' was a path or url but couldn't find any file "
                "associated to this path or url.".format(
                    pretrained_model_name,
                    ', '.join(PRETRAINED_MODEL_ARCHIVE_MAP.keys()),
                    archive_file))
            return None
        if resolved_archive_file == archive_file:
            logger.info("loading archive file {}".format(archive_file))
        else:
            logger.info("loading archive file {} from cache at {}".format(
                archive_file, resolved_archive_file))
        tempdir = None
        if os.path.isdir(resolved_archive_file):
            serialization_dir = resolved_archive_file
        else:
            # Extract archive to temp dir
            tempdir = tempfile.mkdtemp()
            logger.info("extracting archive file {} to temp dir {}".format(
                resolved_archive_file, tempdir))
            with tarfile.open(resolved_archive_file, 'r:gz') as archive:
                archive.extractall(tempdir)
            serialization_dir = tempdir
        # Load config
        config_file = os.path.join(serialization_dir, CONFIG_NAME)
        config = BertConfig.from_json_file(config_file)
        logger.info("Model config {}".format(config))
        # Instantiate model.
        model = cls(config, *inputs, **kwargs)
        if state_dict is None:
            weights_path = os.path.join(serialization_dir, WEIGHTS_NAME)
            state_dict = torch.load(weights_path)

        old_keys = []
        new_keys = []
        for key in state_dict.keys():
            new_key = None
            if 'gamma' in key:
                new_key = key.replace('gamma', 'weight')
            if 'beta' in key:
                new_key = key.replace('beta', 'bias')
            if new_key:
                old_keys.append(key)
                new_keys.append(new_key)
        for old_key, new_key in zip(old_keys, new_keys):
            state_dict[new_key] = state_dict.pop(old_key)

        missing_keys = []
        unexpected_keys = []
        error_msgs = []
        # copy state_dict so _load_from_state_dict can modify it
        metadata = getattr(state_dict, '_metadata', None)
        state_dict = state_dict.copy()
        if metadata is not None:
            state_dict._metadata = metadata

        def load(module, prefix=''):
            #print('Note that prefix is: '+prefix)
            local_metadata = {} if metadata is None else metadata.get(
                prefix[:-1], {})
            module._load_from_state_dict(state_dict, prefix, local_metadata,
                                         True, missing_keys, unexpected_keys,
                                         error_msgs)
            for name, child in module._modules.items():
                if child is not None:
                    load(child, prefix + name + '.')

        load(model, prefix='' if hasattr(model, 'bert') else 'bert.')
        #if len(missing_keys) > 0:
        #logger.info("Weights of {} not initialized from pretrained model: {}".format(
        #model.__class__.__name__, missing_keys))
        if len(unexpected_keys) > 0:
            logger.info(
                "Weights from pretrained model not used in {}: {}".format(
                    model.__class__.__name__, unexpected_keys))
        if tempdir:
            # Clean up temp dir
            shutil.rmtree(tempdir)
        return model


class BertModel(PreTrainedBertModel):
    """BERT model ("Bidirectional Embedding Representations from a Transformer").

    Params:
        config: a BertConfig class instance with the configuration to build a new model

    Inputs:
        `input_ids`: a torch.LongTensor of shape [batch_size, sequence_length]
            with the word token indices in the vocabulary(see the tokens preprocessing logic in the scripts
            `extract_features.py`, `run_classifier.py` and `run_squad.py`)
        `token_type_ids`: an optional torch.LongTensor of shape [batch_size, sequence_length] with the token
            types indices selected in [0, 1]. Type 0 corresponds to a `sentence A` and type 1 corresponds to
            a `sentence B` token (see BERT paper for more details).
        `attention_mask`: an optional torch.LongTensor of shape [batch_size, sequence_length] with indices
            selected in [0, 1]. It's a mask to be used if the input sequence length is smaller than the max
            input sequence length in the current batch. It's the mask that we typically use for attention when
            a batch has varying length sentences.
        `output_all_encoded_layers`: boolean which controls the content of the `encoded_layers` output as described below. Default: `True`.

    Outputs: Tuple of (encoded_layers, pooled_output)
        `encoded_layers`: controled by `output_all_encoded_layers` argument:
            - `output_all_encoded_layers=True`: outputs a list of the full sequences of encoded-hidden-states at the end
                of each attention block (i.e. 12 full sequences for BERT-base, 24 for BERT-large), each
                encoded-hidden-state is a torch.FloatTensor of size [batch_size, sequence_length, hidden_size],
            - `output_all_encoded_layers=False`: outputs only the full sequence of hidden-states corresponding
                to the last attention block of shape [batch_size, sequence_length, hidden_size],
        `pooled_output`: a torch.FloatTensor of size [batch_size, hidden_size] which is the output of a
            classifier pretrained on top of the hidden state associated to the first character of the
            input (`CLF`) to train on the Next-Sentence task (see BERT's paper).

    Example usage:
    ```python
    # Already been converted into WordPiece token ids
    input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])
    input_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])
    token_type_ids = torch.LongTensor([[0, 0, 1], [0, 1, 0]])

    config = modeling.BertConfig(vocab_size_or_config_json_file=32000, hidden_size=768,
        num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072)

    model = modeling.BertModel(config=config)
    all_encoder_layers, pooled_output = model(input_ids, token_type_ids, input_mask)
    ```
    """
    def __init__(self, config):
        super(BertModel, self).__init__(config)
        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)
        self.pooler = BertPooler(config)
        self.apply(self.init_bert_weights)

    def forward(self,
                input_ids,
                token_type_ids=None,
                attention_mask=None,
                output_all_encoded_layers=True):
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        # We create a 3D attention mask from a 2D tensor mask.
        # Sizes are [batch_size, 1, 1, to_seq_length]
        # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
        # this attention mask is more simple than the triangular masking of causal attention
        # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = extended_attention_mask.to(
            dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        embedding_output = self.embeddings(input_ids, token_type_ids)
        encoded_layers = self.encoder(
            embedding_output,
            extended_attention_mask,
            output_all_encoded_layers=output_all_encoded_layers)
        sequence_output = encoded_layers[-1]
        pooled_output = self.pooler(sequence_output)
        if not output_all_encoded_layers:
            encoded_layers = encoded_layers[-1]
        return encoded_layers, pooled_output


class BertModelForAttentionScore(PreTrainedBertModel):
    def __init__(self, config):
        super(BertModelForAttentionScore, self).__init__(config)
        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)
        self.pooler = BertPooler(config)
        self.apply(self.init_bert_weights)

    def forward(self,
                input_ids,
                token_type_ids=None,
                attention_mask=None,
                output_all_encoded_layers=True):
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        # We create a 3D attention mask from a 2D tensor mask.
        # Sizes are [batch_size, 1, 1, to_seq_length]
        # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
        # this attention mask is more simple than the triangular masking of causal attention
        # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = extended_attention_mask.to(
            dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        embedding_output = self.embeddings(input_ids, token_type_ids)
        encoded_layers = self.encoder(
            embedding_output,
            extended_attention_mask,
            output_all_encoded_layers=output_all_encoded_layers)
        sequence_output = encoded_layers[-1]
        pooled_output = self.pooler(sequence_output)
        if not output_all_encoded_layers:
            encoded_layers = encoded_layers[-1]
        return encoded_layers, pooled_output


class BertForPreTraining(PreTrainedBertModel):
    """BERT model with pre-training heads.
    This module comprises the BERT model followed by the two pre-training heads:
        - the masked language modeling head, and
        - the next sentence classification head.

    Params:
        config: a BertConfig class instance with the configuration to build a new model.

    Inputs:
        `input_ids`: a torch.LongTensor of shape [batch_size, sequence_length]
            with the word token indices in the vocabulary(see the tokens preprocessing logic in the scripts
            `extract_features.py`, `run_classifier.py` and `run_squad.py`)
        `token_type_ids`: an optional torch.LongTensor of shape [batch_size, sequence_length] with the token
            types indices selected in [0, 1]. Type 0 corresponds to a `sentence A` and type 1 corresponds to
            a `sentence B` token (see BERT paper for more details).
        `attention_mask`: an optional torch.LongTensor of shape [batch_size, sequence_length] with indices
            selected in [0, 1]. It's a mask to be used if the input sequence length is smaller than the max
            input sequence length in the current batch. It's the mask that we typically use for attention when
            a batch has varying length sentences.
        `masked_lm_labels`: masked language modeling labels: torch.LongTensor of shape [batch_size, sequence_length]
            with indices selected in [-1, 0, ..., vocab_size]. All labels set to -1 are ignored (masked), the loss
            is only computed for the labels set in [0, ..., vocab_size]
        `next_sentence_label`: next sentence classification loss: torch.LongTensor of shape [batch_size]
            with indices selected in [0, 1].
            0 => next sentence is the continuation, 1 => next sentence is a random sentence.

    Outputs:
        if `masked_lm_labels` and `next_sentence_label` are not `None`:
            Outputs the total_loss which is the sum of the masked language modeling loss and the next
            sentence classification loss.
        if `masked_lm_labels` or `next_sentence_label` is `None`:
            Outputs a tuple comprising
            - the masked language modeling logits of shape [batch_size, sequence_length, vocab_size], and
            - the next sentence classification logits of shape [batch_size, 2].

    Example usage:
    ```python
    # Already been converted into WordPiece token ids
    input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])
    input_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])
    token_type_ids = torch.LongTensor([[0, 0, 1], [0, 1, 0]])

    config = BertConfig(vocab_size_or_config_json_file=32000, hidden_size=768,
        num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072)

    model = BertForPreTraining(config)
    masked_lm_logits_scores, seq_relationship_logits = model(input_ids, token_type_ids, input_mask)
    ```
    """
    def __init__(self, config):
        super(BertForPreTraining, self).__init__(config)
        self.bert = BertModel(config)
        self.cls = BertPreTrainingHeads(
            config, self.bert.embeddings.word_embeddings.weight)
        self.apply(self.init_bert_weights)

    def forward(self,
                input_ids,
                token_type_ids=None,
                attention_mask=None,
                masked_lm_labels=None,
                next_sentence_label=None):
        sequence_output, pooled_output = self.bert(
            input_ids,
            token_type_ids,
            attention_mask,
            output_all_encoded_layers=False)
        prediction_scores, seq_relationship_score = self.cls(
            sequence_output, pooled_output)

        if masked_lm_labels is not None and next_sentence_label is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-1)
            masked_lm_loss = loss_fct(
                prediction_scores.view(-1, self.config.vocab_size),
                masked_lm_labels.view(-1))
            next_sentence_loss = loss_fct(seq_relationship_score.view(-1, 2),
                                          next_sentence_label.view(-1))
            total_loss = masked_lm_loss + next_sentence_loss
            return total_loss
        else:
            return prediction_scores, seq_relationship_score


class BertForSequenceClassification(PreTrainedBertModel):
    """BERT model for classification.
    This module is composed of the BERT model with a linear layer on top of
    the pooled output.

    Params:
        `config`: a BertConfig class instance with the configuration to build a new model.
        `num_labels`: the number of classes for the classifier. Default = 2.

    Inputs:
        `input_ids`: a torch.LongTensor of shape [batch_size, sequence_length]
            with the word token indices in the vocabulary(see the tokens preprocessing logic in the scripts
            `extract_features.py`, `run_classifier.py` and `run_squad.py`)
        `token_type_ids`: an optional torch.LongTensor of shape [batch_size, sequence_length] with the token
            types indices selected in [0, 1]. Type 0 corresponds to a `sentence A` and type 1 corresponds to
            a `sentence B` token (see BERT paper for more details).
        `attention_mask`: an optional torch.LongTensor of shape [batch_size, sequence_length] with indices
            selected in [0, 1]. It's a mask to be used if the input sequence length is smaller than the max
            input sequence length in the current batch. It's the mask that we typically use for attention when
            a batch has varying length sentences.
        `labels`: labels for the classification output: torch.LongTensor of shape [batch_size]
            with indices selected in [0, ..., num_labels].

    Outputs:
        if `labels` is not `None`:
            Outputs the CrossEntropy classification loss of the output with the labels.
        if `labels` is `None`:
            Outputs the classification logits of shape [batch_size, num_labels].

    Example usage:
    ```python
    # Already been converted into WordPiece token ids
    input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])
    input_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])
    token_type_ids = torch.LongTensor([[0, 0, 1], [0, 1, 0]])

    config = BertConfig(vocab_size_or_config_json_file=32000, hidden_size=768,
        num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072)

    num_labels = 2

    model = BertForSequenceClassification(config, num_labels)
    logits = model(input_ids, token_type_ids, input_mask)
    ```
    """
    def __init__(self, config, num_labels=2):
        super(BertForSequenceClassification, self).__init__(config)
        self.num_labels = num_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, num_labels)
        self.apply(self.init_bert_weights)

    def forward(self,
                input_ids,
                token_type_ids=None,
                attention_mask=None,
                labels=None):
        seq_output, pooled_output = self.bert(input_ids,
                                              token_type_ids,
                                              attention_mask,
                                              output_all_encoded_layers=False)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            return loss
        else:
            return logits, seq_output



class HierarchicalTransformer_TopN_Attention_v2(PreTrainedBertModel):
    def __init__(self,
                 config,
                 num_labels=3,
                 embedding=None,
                 TAT_num=1,
                 MFT_num=1):
        super(HierarchicalTransformer_TopN_Attention_v2, self).__init__(config)
        self.num_labels = num_labels
        self.embedding = embedding
        self.bert = BertModel(config)
        self.target_bert = BertModel(config)

        self.TargetGuidedAttention = TargetGuidedAttention()
        self.Wr = nn.Linear(16, 64)
        self.mse = torch.nn.MSELoss()

        self.vismap2text = nn.Linear(2048 + 300, config.hidden_size)
        self.left_TAT = BertCrossEncoder(config, layer_num=TAT_num)
        self.right_TAT = BertCrossEncoder(config, layer_num=TAT_num)
        self.left_MFT = MultimodalityFusionLayer(config, layernum=MFT_num)
        self.right_MFT = MultimodalityFusionLayer(config, layernum=MFT_num)

        self.comb_attention = MultimodalEncoder(config)
        self.pooler = BertPooler(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, num_labels)

        self.apply(self.init_bert_weights)

    def forward(self, context_ids, context_mask, context_segment_ids,
                target_ids, target_mask, target_segment_ids, img_feature,
                img_mask, img_sematic_ids, combine_mask, copy_flag):
        if copy_flag:
            print(
                "*************************copy bert's weights to s2_bert******************"
            )
            self.target_bert = copy.deepcopy(self.bert)
        context_output, _ = self.bert(context_ids,
                                      context_segment_ids,
                                      context_mask,
                                      output_all_encoded_layers=False)
        target_output, target_cls_output = self.target_bert(
            target_ids,
            target_mask,
            target_segment_ids,
            output_all_encoded_layers=False)
        # print('!!!!!!!!!!!!!!!!!!!!')
        # print(img_sematic_ids.shape)
        img_sematic = self.embedding(
            img_sematic_ids)  #[batch_size, 36, top_n, 2, 300]
        img_sematic = torch.mean(img_sematic, dim=3,
                                 keepdim=False)  #[batch_size, 36, top_n, 300]
        img_sematic = self.TargetGuidedAttention(
            target_cls_output, img_sematic)  #[batch_size, 36, 300]
        combined_img_feature = torch.cat((img_sematic, img_feature),
                                         dim=-1)  #[batch_size, 36, 2348]
        v2t_img_feature = self.vismap2text(combined_img_feature)
        '''
        ----------------------------------------------------------------------------------------------
        '''
        extended_img_mask = img_mask.unsqueeze(1).unsqueeze(2)
        extended_img_mask = extended_img_mask.to(dtype=next(
            self.parameters()).dtype)  # fp16 compatibility
        extended_img_mask = (1.0 - extended_img_mask) * -10000.0

        extended_context_mask = context_mask.unsqueeze(1).unsqueeze(2)
        extended_context_mask = extended_context_mask.to(
            dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_context_mask = (1.0 - extended_context_mask) * -10000.0

        extended_target_mask = target_mask.unsqueeze(1).unsqueeze(2)
        extended_target_mask = extended_target_mask.to(dtype=next(
            self.parameters()).dtype)  # fp16 compatibility
        extended_target_mask = (1.0 - extended_target_mask) * -10000.0

        extended_attention_mask = combine_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(
            dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        '''
        ----------------------------------------------------------------------------------------------
        '''

        target_aware_context = self.left_TAT(target_output, context_output,
                                             extended_context_mask)[-1]
        target_aware_image = self.right_TAT(target_output, v2t_img_feature,
                                            extended_img_mask)[-1]

        reconstruct_from_context = self.Wr(
            target_aware_context.permute(0, 2, 1)).permute(0, 2, 1)
        reconstruct_from_image = self.Wr(target_aware_image.permute(
            0, 2, 1)).permute(0, 2, 1)

        r_loss = (self.mse(reconstruct_from_context, context_output) +
                  self.mse(reconstruct_from_image, context_output)) / 2

        multimodality_fused_image = self.left_MFT(target_aware_context,
                                                  target_aware_image,
                                                  extended_target_mask,
                                                  extended_target_mask)
        multimodality_fused_context = self.right_MFT(target_aware_image,
                                                     target_aware_context,
                                                     extended_target_mask,
                                                     extended_target_mask)

        cls_multimodality_fused_image = multimodality_fused_image[:,
                                                                  0].unsqueeze(
                                                                      1)

        combine_representation = torch.cat(
            (cls_multimodality_fused_image, multimodality_fused_context),
            dim=1)

        combine_output = self.comb_attention(combine_representation,
                                             extended_attention_mask)[-1]

        output = self.pooler(combine_output)
        logits = self.classifier(output)

        return logits, r_loss


class ZOL_HIMT(PreTrainedBertModel):
    def __init__(self,
                 config,
                 num_labels=3,
                 embedding=None,
                 TAT_num=1,
                 MFT_num=1):
        super(ZOL_HIMT, self).__init__(config)
        self.num_labels = num_labels
        self.embedding = embedding
        self.bert = BertModel(config)
        self.target_bert = BertModel(config)

        self.TargetGuidedAttention = TargetGuidedAttention()
        self.Wr = nn.Linear(16, 128)
        self.mse = torch.nn.MSELoss()

        self.vismap2text = nn.Linear(2348, config.hidden_size)
        self.left_TAT = BertCrossEncoder(config, layer_num=TAT_num)
        self.right_TAT = BertCrossEncoder(config, layer_num=TAT_num)
        self.left_MFT = MultimodalityFusionLayer(config, layernum=MFT_num)
        self.right_MFT = MultimodalityFusionLayer(config, layernum=MFT_num)

        self.comb_attention = MultimodalEncoder(config)
        self.pooler = BertPooler(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, num_labels)

        self.apply(self.init_bert_weights)

    def forward(self, context_ids, context_mask, context_segment_ids,
                target_ids, target_mask, target_segment_ids, img_feature,
                img_mask, img_sematic_ids, combine_mask, copy_flag):
        if copy_flag:
            print(
                "*************************copy bert's weights to s2_bert******************"
            )
            self.target_bert = copy.deepcopy(self.bert)
        context_output, _ = self.bert(context_ids,
                                      context_segment_ids,
                                      context_mask,
                                      output_all_encoded_layers=False)
        target_output, target_cls_output = self.target_bert(
            target_ids,
            target_mask,
            target_segment_ids,
            output_all_encoded_layers=False)

        img_sematic = self.embedding(
            img_sematic_ids)  #[batch_size, 144, top_n, 2, 300]
        img_sematic = torch.mean(img_sematic, dim=3,
                                 keepdim=False)  #[batch_size, 36, top_n, 300]
        img_sematic = self.TargetGuidedAttention(
            target_cls_output, img_sematic)  #[batch_size, 36, 300]
        now_batch_size = img_feature.size()[0]
        img_feature = img_feature.view(now_batch_size, -1, 2048)
        combined_img_feature = torch.cat((img_sematic, img_feature),
                                         dim=-1)  #[batch_size, 36, 2348]
        v2t_img_feature = self.vismap2text(
            combined_img_feature)  #[batch_size, 4, 36, 2048]
        '''
        ----------------------------------------------------------------------------------------------
        '''
        extended_img_mask = img_mask.unsqueeze(1).unsqueeze(2)
        extended_img_mask = extended_img_mask.to(dtype=next(
            self.parameters()).dtype)  # fp16 compatibility
        extended_img_mask = (1.0 - extended_img_mask) * -10000.0

        extended_context_mask = context_mask.unsqueeze(1).unsqueeze(2)
        extended_context_mask = extended_context_mask.to(
            dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_context_mask = (1.0 - extended_context_mask) * -10000.0

        extended_target_mask = target_mask.unsqueeze(1).unsqueeze(2)
        extended_target_mask = extended_target_mask.to(dtype=next(
            self.parameters()).dtype)  # fp16 compatibility
        extended_target_mask = (1.0 - extended_target_mask) * -10000.0

        extended_attention_mask = combine_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(
            dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        '''
        ----------------------------------------------------------------------------------------------
        '''

        target_aware_context = self.left_TAT(target_output, context_output,
                                             extended_context_mask)[-1]
        target_aware_image = self.right_TAT(target_output, v2t_img_feature,
                                            extended_img_mask)[-1]

        reconstruct_from_context = self.Wr(
            target_aware_context.permute(0, 2, 1)).permute(0, 2, 1)
        reconstruct_from_image = self.Wr(target_aware_image.permute(
            0, 2, 1)).permute(0, 2, 1)

        r_loss = (self.mse(reconstruct_from_context, context_output) +
                  self.mse(reconstruct_from_image, context_output)) / 2

        multimodality_fused_image = self.left_MFT(target_aware_context,
                                                  target_aware_image,
                                                  extended_target_mask,
                                                  extended_target_mask)
        multimodality_fused_context = self.right_MFT(target_aware_image,
                                                     target_aware_context,
                                                     extended_target_mask,
                                                     extended_target_mask)

        cls_multimodality_fused_image = multimodality_fused_image[:,
                                                                  0].unsqueeze(
                                                                      1)

        combine_representation = torch.cat(
            (cls_multimodality_fused_image, multimodality_fused_context),
            dim=1)

        combine_output = self.comb_attention(combine_representation,
                                             extended_attention_mask)[-1]

        output = self.pooler(combine_output)
        logits = self.classifier(output)

        return logits, r_loss