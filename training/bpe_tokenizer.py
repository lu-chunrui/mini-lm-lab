END_OF_WORD = "</w>"
from collections import Counter

def count_pairs(text):
    pairs = Counter()
    for tokens in text:
        for left,right in zip(tokens,tokens[1:]):
            pairs[(left,right)]+=1
    return pairs
raw_text="low lower low"
sequences=[list(token) + [END_OF_WORD] for token in raw_text.split()]
print("初始序列：")
print(sequences)

pair_counts = count_pairs(sequences)
print("\n相邻token对出现次数：")
for pair, count in pair_counts.items():
    print(pair, count)

best_pair = max(pair_counts,key=pair_counts.get)
print("\n出现次数最多的token对：")
print(best_pair)
print("出现次数：")
print(pair_counts[best_pair])

def merge_pairs(text,pair):
    new_text = []
    for tokens in text:
        new_tokens = []
        index=0
        while index<len(tokens):
            can_merge=(index+1<len(tokens) and tokens[index]==pair[0] and tokens[index+1]==pair[1])
            if can_merge:
               merged_token=tokens[index]+tokens[index+1]
               new_tokens.append(merged_token)
               index+=2
            else:
               new_tokens.append(tokens[index])
               index+=1
        new_text.append(new_tokens)
    return new_text
def train_bpe(text,num_merges):
    text = [list(token) + [END_OF_WORD] for token in text.split()]
    merges = []
    for _ in range(num_merges):
        pair_counts = count_pairs(text)
        if not pair_counts:
            break
        best_pair = max(pair_counts,key=pair_counts.get)
        text = merge_pairs(text,best_pair)
        merges.append(best_pair)
        print(f"第{_ + 1}次合并：",best_pair,"出现次数：",pair_counts[best_pair])
        print("合并结果：", text)
    return merges, text
def encode_word(word,merge_rules):
    tokens=list(word)+[END_OF_WORD]
    for pair in merge_rules:
       tokens=merge_pairs([tokens],pair)[0]
    return tokens

def encode_text(text,merge_rules):
    encode_tokens=[]
    for word in text.split():
        word_tokens=encode_word(word,merge_rules)
        encode_tokens.extend(word_tokens)
    return encode_tokens

def decode_text(encode_tokens):
    text="".join(encode_tokens)
    text=text.replace(END_OF_WORD," ")
    return text.rstrip()

if __name__ == "__main__":
    training_text = (
        "low lower lowest "
        "low lower low"
    )
    print("训练文本：")
    print(training_text)
    merge_rules, final_sequences = train_bpe(text=training_text,num_merges=8)

    print("\n============================")
    print("学习到的合并规则")
    print("============================")
    for rule_index, rule in enumerate( merge_rules,start=1):
         print(f"规则{rule_index}：",rule,"→",rule[0] + rule[1])
    print("\n最终训练序列：")
    print(final_sequences)
    test_word = "lower"
    encoded_word = encode_word( word=test_word, merge_rules=merge_rules)
    print("\n============================")
    print("单词编码测试")
    print("============================")
    print("原始单词：", test_word)
    print("编码结果：", encoded_word)
    test_text = "low lower lowest"
    encoded_tokens = encode_text(text=test_text,merge_rules=merge_rules)
    print("\n============================")
    print("完整文本编码测试")
    print("============================")
    print("原始文本：", test_text)
    print("BPE Tokens：", encoded_tokens)
    # 测试解码
    decoded_text = decode_text(
        encoded_tokens
)
    print("\n============================")
    print("解码测试")
    print("============================")
    print("解码结果：", decoded_text)
    print("编码解码是否一致：",decoded_text == test_text)
