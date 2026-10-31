from tokenizers import Tokenizer
t = Tokenizer.from_file(r"E:\Parentheses\parentheses-0.9\tokenizer\tokenizer.json")
p1 = "The history of the Vijayanagara Empire in South India began when"
p2 = "Question: If a train travels 180 km in 3 hours, how far will it travel in 5 hours at the same speed?\nAnswer: <think>"
print("P1 len:", len(t.encode(p1).ids))
print("P2 len:", len(t.encode(p2).ids))
