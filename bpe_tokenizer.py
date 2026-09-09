import json
from collections import Counter
from pathlib import Path

class BPETokenizer:
   end_of_word = '</w>'#表示单词结束的标记
   unknown_token = '<unk>'#表示未知单词的标记

   def __init__(self):
      self.merge_rules = []
      self.token_to_id = {}
      self.id_to_token = {}
   @property#属性式访问
   def vocab_size(self):
      return len(self.token_to_id)
   def count_tokens(self, text):
    pairs = Counter()
    for token in text:
        for left, right in zip(token, token[1:]):
            pairs[(left, right)] += 1
    return pairs
   def merge_tokens(self,tokens, pairs):
     new_tokens = []
     index=0
     while index<len(tokens):
          can_merge = (index+1<len(tokens)and tokens[index]==pairs[0]and tokens[index + 1] == pairs[1])
          if can_merge:
            new_tokens.append(pairs[0]+pairs[1])
            index += 2
          else:
            new_tokens.append(tokens[index])
            index += 1
     return new_tokens
   def merge_pair(self, text, pairs):
    return [self.merge_tokens(token, pairs) for token in text]
   def train(self, text, num_steps):
    self.merge_rules = []
    self.token_to_id = {}
    self.id_to_token = {}
    pairs = [list(word)+[self.end_of_word] for word in text.split()]
    vocabulary={self.unknown_token,self.end_of_word}
    for word in text.split():
        for char in word:
            vocabulary.add(char)
    for step in range(num_steps):
        pair_counts = self.count_tokens(pairs)
        if not pair_counts:
            break
        best_pair = max(pair_counts, key=pair_counts.get)
        best_count = pair_counts[best_pair]
        pairs=self.merge_pair(pairs, best_pair)
        self.merge_rules.append(best_pair)
        merge_token = best_pair[0]+best_pair[1]
        vocabulary.add(merge_token)
        print(f"第{step+1}步：",best_pair,"出现次数：",best_count)
    self.token_to_id = {token: i for i, token in enumerate(vocabulary)}
    self.id_to_token = {i: token for i, token in enumerate(vocabulary)}
    print("训练完成")
    print("词汇表大小：",self.vocab_size)
    return pairs
   def encode_word(self, word):
    tokens=(list(word)+[self.end_of_word])
    for pair in self.merge_rules:
        tokens=self.merge_tokens(tokens,pair)
    return tokens
   def tokenize(self, text):
    tokens=[]
    for word in text.split():
       word_tokens=self.encode_word(word)
       tokens.extend(word_tokens)
    return tokens
   def encode(self, text):
      if not self.token_to_id:raise RuntimeError("Tokenizer还没有训练或加载")
      tokens=self.tokenize(text)
      unknown_id=self.token_to_id.get(self.unknown_token)
      token_ids=[self.token_to_id.get(token,unknown_id) for token in tokens]
      return token_ids
   def decode(self, token_ids):
    if not self.id_to_token:raise RuntimeError("Tokenizer还没有训练或加载")
    tokens=[self.id_to_token[token_id] for token_id in token_ids]
    text="".join(tokens)
    text=text.replace(self.end_of_word," ")
    return text.rstrip()
   def save(self, path):
    file_path=Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    data={
       "merge_rules": [
                [left, right]
                for left, right
                in self.merge_rules
            ],
            "token_to_id": self.token_to_id
        }
    with open(file_path, "w",encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"tokenizer已保存到 {file_path}")
   @classmethod
   def load(cls, path):
    file_path=Path(path)
    with open(file_path, "r",encoding="utf-8") as f:
        data = json.load(f)
    tokenizer = cls()
    tokenizer.merge_rules = [tuple(rule) for rule in data["merge_rules"]]
    tokenizer.token_to_id = {token: i for token, i in data["token_to_id"].items()}
    tokenizer.id_to_token = {token_id: token for token, token_id in tokenizer.token_to_id.items()}
    return tokenizer
if __name__ == "__main__":
    train_text=("low lower lowest ""low lower low")
    tokenizer=BPETokenizer()
    tokenizer.train(train_text, num_steps=50)
    print("\n学习到的合并规则：")
    for index, rule in enumerate(tokenizer.merge_rules):
       print(f"规则{index}：",rule,"→",rule[0] + rule[1])
    print("\n词表：")
    for token, token_id in (tokenizer.token_to_id.items()):
        print(repr(token),"→",token_id)
    test_text = "low lower lowest"
    tokens = tokenizer.tokenize(test_text)
    token_ids = tokenizer.encode(test_text)
    decoded_text = tokenizer.decode(token_ids)
    print("\n原始文本：", test_text)
    print("BPE Tokens：", tokens)
    print("整数ID：", token_ids)
    print("解码结果：", decoded_text)
    print(
        "编码解码是否一致：",
        decoded_text == test_text
    )
    tokenizer.save(
        "checkpoints/bpe_tokenizer.json"
    )
    loaded_tokenizer = BPETokenizer.load(
        "checkpoints/bpe_tokenizer.json"
    )
    loaded_ids = loaded_tokenizer.encode(
        test_text
    )
    loaded_text = loaded_tokenizer.decode(
        loaded_ids
    )
    print("\n重新加载后的ID：", loaded_ids)
    print("重新加载后的文本：", loaded_text)
    print(
        "保存加载是否正确：",
        loaded_text == test_text
    )


