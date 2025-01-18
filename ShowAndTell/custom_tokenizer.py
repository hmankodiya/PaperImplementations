
class VocabTokenizer:
    def __init__(self, documents, max_length=MAX_LENGTH):
        self.documents = documents
        self.max_length = max_length
        self.special_tokens = [
            "<|startoftext|>",
            "<|endoftext|>",
            "<|pad|>",
            "<|unkn|>",
        ]
        self.tokens = self.special_tokens + sorted(
            list(
                set(
                    [
                        token
                        for document in self.documents
                        for token in word_tokenize(preprocess_text(document))
                    ]
                )
            )
        )
        self.size = len(self.tokens)

        self.words2index = {token: i for i, token in enumerate(self.tokens)}
        self.index2words = {i: token for i, token in enumerate(self.tokens)}

    def encode(self, document):
        return (
            [self.words2index["<start>"]]
            + [
                self.words2index.get(token, self.words2index["<unkn>"])
                for token in word_tokenize(preprocess_text(document))
            ]
            + [self.words2index["<end>"]]
        )

    def decode_indexes(self, indexes):
        return " ".join(
            [
                self.index2words.get(index, self.words2index["<unkn>"])
                for index in indexes
            ]
        )

